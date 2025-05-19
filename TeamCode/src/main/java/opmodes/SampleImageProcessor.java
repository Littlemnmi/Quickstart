package opmodes;


import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.*;

/**
 * Separates out core OpenCV image processing for color-based sample detection.
 * Methods can be tested independently via JUnit by supplying Mat inputs.
 */
public class SampleImageProcessor {
    private final Mat kernelClean;

    private final double targetAspectRatio = 3.5/1.5;

    private final double aspectRatioTol = 0.5;

    private Mat cameraMatrix;
    private MatOfDouble distCoeffs;

    private static final double fx = 238.722;
    private static final double fy = 238.722;

    private static final double cx = 323.204;
    private static final double cy = 228.638;

    public List<DetectedSample> samples = new ArrayList<>();


    public SampleImageProcessor() {
        // 7×7 elliptical kernel for morphological operations
        this.kernelClean = Imgproc.getStructuringElement(
                Imgproc.MORPH_ELLIPSE,
                new Size(7, 7)
        );

        cameraMatrix = Mat.zeros(3,3, CvType.CV_64F);
        cameraMatrix.put(0, 0, fx);
        cameraMatrix.put(0, 1, 0);
        cameraMatrix.put(0, 2, cx);
        cameraMatrix.put(1, 0, 0);
        cameraMatrix.put(1, 1, fy);
        cameraMatrix.put(1, 2, cy);
        cameraMatrix.put(2, 0, 0);
        cameraMatrix.put(2, 1, 0);
        cameraMatrix.put(2, 2, 1);

        distCoeffs = new MatOfDouble(
                0.154576,   // k1
                -1.19143,   // k2
                0,   // p1 (tangential)
                0,   // p2 (tangential)
                2.06105  // k3
        );
    }

    /**
     * Produce a single-channel mask of the given color in an HSV image.
     */
    public Mat getColorMask(Mat hsv, CameraColor color) {
        Mat mask = new Mat();
        Mat lower = new Mat(), upper = new Mat();

        switch (color) {
            case RED:
                Core.inRange(hsv, new Scalar(0, 100, 100), new Scalar(10, 255, 255), lower);
                Core.inRange(hsv, new Scalar(160, 100, 100), new Scalar(180, 255, 255), upper);
                Core.bitwise_or(lower, upper, mask);
                break;
            case BLUE:
                Core.inRange(hsv, new Scalar(75, 100, 100), new Scalar(145, 255, 255), mask);
                break;
            case YELLOW:
                Core.inRange(hsv, new Scalar(20, 100, 100), new Scalar(35, 255, 255), mask);
                break;
            default:
                throw new IllegalArgumentException("Color not recognized: " + color);
        }

        lower.release();
        upper.release();
        return mask;
    }

    /**
     * Clean up a binary mask via morphological closing then opening.
     */
    public Mat cleanMask(Mat mask) {
        Mat cleaned = new Mat();
        Imgproc.morphologyEx(mask, cleaned, Imgproc.MORPH_CLOSE, kernelClean, new Point(-1, -1), 2);
        Mat cleaned2 = new Mat();
        Imgproc.morphologyEx(cleaned, cleaned2, Imgproc.MORPH_OPEN,  kernelClean, new Point(-1, -1), 2);
        return cleaned2;
    }

    /**
     * Compute watershed markers from a cleaned mask.
     * Returns a 32S Mat where each region has a unique marker ID.
     */
    public Mat computeMarkers(Mat maskClean, Mat inputBgr) {
        // Distance transform
        Mat dist = new Mat();
        Imgproc.distanceTransform(maskClean, dist, Imgproc.DIST_L2, 5);
        Core.MinMaxLocResult mmr = Core.minMaxLoc(dist);

        // Foreground (sure internal regions)
        Mat fg = new Mat();
        Imgproc.threshold(dist, fg, 0.8 * mmr.maxVal, 255, Imgproc.THRESH_BINARY);
        fg.convertTo(fg, CvType.CV_8U);

        // Background (sure external regions)
        Mat bg = new Mat();
        Imgproc.dilate(maskClean, bg, kernelClean, new Point(-1, -1), 3);

        // Unknown region = bg - fg
        Mat unknown = new Mat();
        Core.subtract(bg, fg, unknown);

        //renderImage(fg, "fg.png");
        //renderImage(bg, "bg.png");

        // Label foreground components
        Mat markers = new Mat();
        Imgproc.connectedComponents(fg, markers);
        //renderImage(markers, "markers_0.png");
        Core.add(markers, Scalar.all(1), markers);        // ensure background != 0
        markers.setTo(Scalar.all(0), unknown);           // mark unknown as 0


        // Apply watershed to segment touching regions
        Imgproc.watershed(inputBgr, markers);

        dist.release();
        fg.release();
        bg.release();
        unknown.release();
        return markers;
    }

    /**
     * From labeled markers, extract rotated bounding boxes for regions above minArea.
     */
    public List<RotatedRect> findBoundingBoxes(Mat markers, double minArea) {
        Core.MinMaxLocResult mm = Core.minMaxLoc(markers);
        int maxLabel = (int) mm.maxVal;
        List<RotatedRect> boxes = new ArrayList<>();

        for (int label = 2; label <= maxLabel; label++) {
            Mat region = new Mat();
            Core.compare(markers, new Scalar(label), region, Core.CMP_EQ);
            region.convertTo(region, CvType.CV_8U, 255);

            List<MatOfPoint> contours = new ArrayList<>();
            Imgproc.findContours(region, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
            for (MatOfPoint cnt : contours) {
                double area = Imgproc.contourArea(cnt);
                if (area > minArea) {
                    RotatedRect rect = Imgproc.minAreaRect(new MatOfPoint2f(cnt.toArray()));
                    boxes.add(rect);
                }
                cnt.release();
            }
            region.release();
        }
        return boxes;
    }

    /**
     * Full pipeline: HSV conversion, masking, cleaning, watershed, and box extraction.
     */
    public List<DetectedSample> detectSamples(Mat inputBgr, CameraColor color, double minArea) {
        samples.clear();
        Mat hsv = new Mat();
        Imgproc.cvtColor(inputBgr, hsv, Imgproc.COLOR_BGR2HSV);

        Mat maskClean = cleanMask(getColorMask(hsv, color));
        //renderImage(maskClean, "mask_clean_" + color.name().toLowerCase() + ".png");

        Mat markers   = computeMarkers(maskClean, inputBgr);

        //renderImage(markers, "markers_" + color.name().toLowerCase() + ".png");

        List<RotatedRect> boxes = findBoundingBoxes(markers, minArea);

        int idx = 0;
        for (RotatedRect rect : boxes) {
            double w = rect.size.width, h = rect.size.height;
            if (w == 0 || h == 0) continue;
            double ratio = Math.max(w, h) / Math.min(w, h);

            // draw only if aspect ratio matches
            if (Math.abs(ratio - targetAspectRatio) < aspectRatioTol) {

                Point[] pts = new Point[4];
                rect.points(pts);

                //for debug
                //idx = renderBoxContour(inputBgr, idx, pts);

                // Pose estimation
                List<Point3> objPts = Arrays.asList(
                        new Point3(-1.75, -0.75, 0),
                        new Point3(1.75, -0.75, 0),
                        new Point3(1.75, 0.75, 0),
                        new Point3(-1.75, 0.75, 0)
                );
                List<Point> imgPts = new ArrayList<>(Arrays.asList(pts));

                Mat rvec = new Mat(), tvec = new Mat();
                Calib3d.solvePnP(
                        new MatOfPoint3f(objPts.toArray(new Point3[0])),
                        new MatOfPoint2f(imgPts.toArray(new Point[0])),
                        cameraMatrix, distCoeffs,
                        rvec, tvec
                );

                double distance = Core.norm(tvec);
                Mat R = new Mat();
                Calib3d.Rodrigues(rvec, R);
                double yaw = Math.atan2(R.get(1, 0)[0], R.get(0, 0)[0]);

                samples.add(new DetectedSample(rect, distance, yaw));
            }
        }

        hsv.release();
        maskClean.release();
        markers.release();
        return samples;
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

