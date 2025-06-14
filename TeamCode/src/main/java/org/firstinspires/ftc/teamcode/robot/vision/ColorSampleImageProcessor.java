package org.firstinspires.ftc.teamcode.robot.vision;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.*;

/**
 * Encapsulates the core image-processing steps for color-based blob detection and pose estimation.
 * Can be unit-tested independently of the FTC SDK.
 */
public class ColorSampleImageProcessor {
    private final List<ColorRange> ranges;
    private final Mat kernelClean;
    private final Mat erodeElem;
    private final Mat dilateElem;
    private final Size blurSize;
    private final Mat cameraMatrix;
    private final MatOfDouble distCoeffs;

    private Mat roiMat_userColorSpace;

    public ColorSampleImageProcessor(
            List<ColorRange> ranges,
            Mat kernelClean,
            Mat erodeElem,
            Mat dilateElem,
            Size blurSize,
            Mat cameraMatrix,
            MatOfDouble distCoeffs
    ) {
        this.ranges = ranges;
        this.kernelClean = kernelClean;
        this.erodeElem = erodeElem;
        this.dilateElem = dilateElem;
        this.blurSize = blurSize;
        this.cameraMatrix = cameraMatrix;
        this.distCoeffs = distCoeffs;
    }

    /**
     * Preprocess ROI: convert color space, blur, and create binary mask.
     */
    public Mat createMask(Mat roiBgr, ColorSpace space) {

        switch (space) {
            case HSV:
                Imgproc.cvtColor(roiBgr, roiMat_userColorSpace, Imgproc.COLOR_RGB2HSV);
                break;
            case YCrCb:
                Imgproc.cvtColor(roiBgr, roiMat_userColorSpace, Imgproc.COLOR_RGB2YCrCb);
                break;
            case RGB:
            default:
                Imgproc.cvtColor(roiBgr, roiMat_userColorSpace, Imgproc.COLOR_RGBA2RGB);
        }
        if (blurSize != null) {
            Imgproc.GaussianBlur(roiMat_userColorSpace, roiMat_userColorSpace, blurSize, 0);
        }

        Mat mask = new Mat(roiBgr.rows(), roiBgr.cols(), CvType.CV_8U, new Scalar(0)); // Start with empty mask
        
        Mat tempMask = new Mat();
        //renderImage(userSpace, "userSpace.png");
        for (int i = 0; i < ranges.size(); i++) {
            ColorRange range = ranges.get(i);
            Core.inRange(roiMat_userColorSpace, range.min, range.max, tempMask);
            //renderImage(tempMask, "tempMask" + i + ".png");
            if (i == 0) {
                // For first range, just copy to the mask
                tempMask.copyTo(mask);
            } else { 
                // For subsequent ranges, OR with the existing mask
                Core.bitwise_or(mask, tempMask, mask);
            }
        }
        tempMask.release();

        if (erodeElem != null) Imgproc.erode(mask, mask, erodeElem);
        if (dilateElem != null) Imgproc.dilate(mask, mask, dilateElem);

        return mask;
    }

    /**
     * Cleans up a binary mask via morphological ops.
     */
    public Mat cleanMask(Mat mask) {
        Mat closed = new Mat();
        Imgproc.morphologyEx(mask, closed, Imgproc.MORPH_CLOSE, kernelClean, new Point(-1,-1), 2);
        Mat opened = new Mat();
        Imgproc.morphologyEx(closed, opened, Imgproc.MORPH_OPEN, kernelClean, new Point(-1,-1), 2);
        closed.release();
        return opened;
    }

    /**
     * Segments blobs via watershed and returns marker image.
     */
    public Mat computeMarkers(Mat clean) {
        Mat dist = new Mat();
        Imgproc.distanceTransform(clean, dist, Imgproc.DIST_L2, 5);
        Core.MinMaxLocResult mm = Core.minMaxLoc(dist);

        Mat fg = new Mat();
        Imgproc.threshold(dist, fg, 0.8 * mm.maxVal, 255, Imgproc.THRESH_BINARY);
        fg.convertTo(fg, CvType.CV_8U);

        Mat bg = new Mat();
        Imgproc.dilate(clean, bg, kernelClean, new Point(-1,-1), 3);

        Mat unknown = new Mat();
        Core.subtract(bg, fg, unknown);

        //renderImage(fg, "fg.png");
        //renderImage(bg, "bg.png");

        Mat markers = new Mat();
        Imgproc.connectedComponents(fg, markers);
        //renderImage(markers, "markers_0.png");
        Core.add(markers, Scalar.all(1), markers);
        markers.setTo(Scalar.all(0), unknown);

        //Mat bgrCopy = bgr.clone();
        Mat bgrCopy = new Mat();
        Imgproc.cvtColor(roiMat_userColorSpace, bgrCopy, Imgproc.COLOR_HSV2BGR);  // Ensure CV_8UC3
        Imgproc.watershed(bgrCopy, markers);
        
        dist.release(); 
        fg.release(); 
        bg.release(); 
        unknown.release(); 
        bgrCopy.release();
        return markers;
    }

    /**
     * Finds blobs (rotated boxes) from labeled markers.
     */
    public List<Blob> findBlobs(Mat markers, Rect roi) {
        Core.MinMaxLocResult mm = Core.minMaxLoc(markers);
        int maxLabel = (int) mm.maxVal;
        List<Blob> blobs = new ArrayList<>();

        for (int lab = 2; lab <= maxLabel; lab++) {
            Mat region = new Mat();
            Core.compare(markers, new Scalar(lab), region, Core.CMP_EQ);
            region.convertTo(region, CvType.CV_8U, 255);

            List<MatOfPoint> contours = new ArrayList<>();
            Mat hier = new Mat();
            Imgproc.findContours(region, contours, hier, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
            hier.release();

            for (MatOfPoint contour : contours) {
                if (Imgproc.contourArea(contour) > 500) {
                    Core.add(contour, new Scalar(roi.x, roi.y), contour);
                    blobs.add(new BlobImpl(contour));
                }
            }
            region.release();
        }
        return blobs;
    }

    /**
     * Estimates 3D pose of each blob.
     */
    public void estimatePose(List<Blob> blobs) {
        
        for (Blob b : blobs) {
            RotatedRect rect = Imgproc.minAreaRect(new MatOfPoint2f(b.getContour().toArray()));
            Point[] pts = new Point[4]; rect.points(pts);

            Point3[] obj = new Point3[]{
                new Point3(-1.75, -0.75, 0), new Point3(1.75, -0.75, 0),
                new Point3(1.75, 0.75, 0), new Point3(-1.75, 0.75, 0)
            };
            MatOfPoint3f objPts = new MatOfPoint3f(obj);
            MatOfPoint2f imgPts = new MatOfPoint2f(pts);

            Mat rvec = new Mat(), tvec = new Mat();
            Calib3d.solvePnP(objPts, imgPts, cameraMatrix, distCoeffs, rvec, tvec);
            double x = tvec.get(0,0)[0];
            double y = tvec.get(1,0)[0];
            double z = tvec.get(2,0)[0];
            Mat R = new Mat(); 
            Calib3d.Rodrigues(rvec, R);
            double yaw = Math.toDegrees(Math.atan2(R.get(1,0)[0], R.get(0,0)[0]));
            b.setPosition(x,y,z,yaw);
        }
    }

    /**
     * Full pipeline: mask -> clean -> markers -> blobs -> pose
     */
    public List<Blob> process(Mat roiBgr, Rect roi) {
        ColorSpace space = ranges.get(0).colorSpace;
        Mat roiMat = roiBgr.submat(roi);
        roiMat_userColorSpace = roiMat.clone();
        //renderImage(roiMat_userColorSpace, "roiMat_userColorSpace.png");
        Mat raw = createMask(roiMat, space);
        //renderImage(raw, "raw.png");
        Mat clean = cleanMask(raw);
        //renderImage(clean, "mask_clean.png");
        Mat markers = computeMarkers(clean);
        //renderImage(markers, "markers.png");
        List<Blob> blobs = findBlobs(markers, roi);
        estimatePose(blobs);

        raw.release(); 
        clean.release(); 
        markers.release();
        roiMat_userColorSpace.release();
        roiMat.release();
        return blobs;
    }

    private int renderBoxContour(Mat inputBgr, int idx, Point[] pts) {

        MatOfPoint box = new MatOfPoint(pts);
        Imgproc.drawContours(inputBgr,
                Collections.singletonList(box), 0, new Scalar(0, 255, 0), 2);
        renderImage(inputBgr, "/detected_sample_" + (idx++) + ".png");

        return idx;
    }

    private void renderImage(Mat input, String filename) {
        File debugOutputDir = new File("build/outputs");
        if (!debugOutputDir.exists()) debugOutputDir.mkdirs();

        // If input might be markers or other non-visual format
        Mat visualOutput = new Mat();
        if (input.type() == CvType.CV_32S) {
            // Convert markers to visualization
            input.convertTo(visualOutput, CvType.CV_8U, 10); // Scale for better visibility
            Imgproc.applyColorMap(visualOutput, visualOutput, Imgproc.COLORMAP_JET);
        } else {
            // Use original image
            input.copyTo(visualOutput);
        }

        Imgcodecs.imwrite(debugOutputDir + "/" + filename, visualOutput);
        visualOutput.release();
    }
}

