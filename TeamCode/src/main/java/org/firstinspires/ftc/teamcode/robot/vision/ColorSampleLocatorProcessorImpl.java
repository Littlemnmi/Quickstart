package org.firstinspires.ftc.teamcode.robot.vision;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.util.Log;

import androidx.annotation.ColorInt;

import com.qualcomm.robotcore.util.SortOrder;

import org.firstinspires.ftc.robotcore.internal.camera.calibration.CameraCalibration;
import org.firstinspires.ftc.vision.VisionProcessor;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.calib3d.Calib3d;

import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;

public class ColorSampleLocatorProcessorImpl extends ColorSampleLocatorProcessor implements VisionProcessor
{
    private ImageRegion roiImg;
    private Rect roi;

    private int contourCode;


    private final Paint boundingRectPaint;
    private final Paint roiPaint;
    private final Paint contourPaint;
    private final boolean drawContours;

    private final Mat erodeElement;
    private final Mat dilateElement;
    private final Size blurElement;

    private final Object lockFilters = new Object();
    private final List<BlobFilter> filters = new ArrayList<>();
    private final Mat kernelClean;

    private Mat cameraMatrix;
    private MatOfDouble distCoeffs;

    private static final double fx = 238.722;
    private static final double fy = 238.722;

    private static final double cx = 323.204;
    private static final double cy = 228.638;

    private ColorSampleImageProcessor processor;

    private volatile BlobSort sort;

    private volatile ArrayList<Blob> userBlobs = new ArrayList<>();

    ColorSampleLocatorProcessorImpl(List<ColorRange> colorRange, ImageRegion roiImg, ContourMode contourMode,
                                  int erodeSize, int dilateSize, boolean drawContours, int blurSize,
                                  @ColorInt int boundingBoxColor, @ColorInt int roiColor, @ColorInt int contourColor)
    {

        this.roiImg = roiImg;
        this.drawContours = drawContours;

        if (blurSize > 0)
        {
            // enforce Odd blurSize
            blurElement = new Size(blurSize | 0x01, blurSize | 0x01);
        }
        else
        {
            blurElement = null;
        }

        if (contourMode == ContourMode.EXTERNAL_ONLY)
        {
            contourCode = Imgproc.RETR_EXTERNAL;
        }
        else
        {
            contourCode = Imgproc.RETR_LIST;
        }

        if (erodeSize > 0)
        {
            erodeElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(erodeSize, erodeSize));
        }
        else
        {
            erodeElement = null;
        }

        if (dilateSize > 0)
        {
            dilateElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(dilateSize, dilateSize));
        }
        else
        {
            dilateElement = null;
        }

        boundingRectPaint = new Paint();
        boundingRectPaint.setAntiAlias(true);
        boundingRectPaint.setStrokeCap(Paint.Cap.BUTT);
        boundingRectPaint.setColor(boundingBoxColor);

        roiPaint = new Paint();
        roiPaint.setAntiAlias(true);
        roiPaint.setStrokeCap(Paint.Cap.BUTT);
        roiPaint.setColor(roiColor);

        contourPaint = new Paint();
        contourPaint.setStyle(Paint.Style.STROKE);
        contourPaint.setColor(contourColor);


        // 7×7 elliptical kernel for morphological operations
        this.kernelClean = Imgproc.getStructuringElement(
                Imgproc.MORPH_ELLIPSE,
                new Size(7, 7)
        );

        cameraMatrix = configureCameraMatrix();

        distCoeffs = configureDistCoeffs();

        // Create the processor with proper parameters
        processor = new ColorSampleImageProcessor(
                colorRange,
                kernelClean,  // kernel for clean
                erodeElement,     // erode element
                dilateElement,     // dilate element
                blurElement,                                                       // blur size
                cameraMatrix,                 // camera matrix
                distCoeffs,                                            // distortion coeffs
                contourCode
        );
    }

    public static Mat configureCameraMatrix() {
        Mat cameraMatrix = Mat.zeros(3,3, CvType.CV_64F);
        cameraMatrix.put(0, 0, fx);
        cameraMatrix.put(0, 1, 0);
        cameraMatrix.put(0, 2, cx);
        cameraMatrix.put(1, 0, 0);
        cameraMatrix.put(1, 1, fy);
        cameraMatrix.put(1, 2, cy);
        cameraMatrix.put(2, 0, 0);
        cameraMatrix.put(2, 1, 0);
        cameraMatrix.put(2, 2, 1);
        return cameraMatrix;
    }

    public static MatOfDouble configureDistCoeffs() {
        return new MatOfDouble(
                0.154576,   // k1
                -1.19143,   // k2
                0,   // p1 (tangential)
                0,   // p2 (tangential)
                2.06105  // k3
        );
    }

    @Override
    public void init(int width, int height, CameraCalibration calibration)
    {
        roi = roiImg.asOpenCvRect(width, height);
    }

    @Override
    public Object processFrame(Mat frame, long captureTimeNanos)
    {

        List<Blob> blobs = processor.process(frame, roi);

        // Apply filters.
        synchronized (lockFilters)
        {
            for (BlobFilter filter : filters)
            {
                switch (filter.criteria)
                {
                    case BY_CONTOUR_AREA:
                        Util.filterByArea(filter.minValue, filter.maxValue, blobs);
                        break;
                    case BY_DENSITY:
                        Util.filterByDensity(filter.minValue, filter.maxValue, blobs);
                        break;
                    case BY_ASPECT_RATIO:
                        Util.filterByAspectRatio(filter.minValue, filter.maxValue, blobs);
                        break;
                }
            }
        }

        // Apply sorting.
        BlobSort sort = this.sort; // Put the field into a local variable for thread safety.
        if (sort != null)
        {
            switch (sort.criteria)
            {
                case BY_CONTOUR_AREA:
                    Util.sortByArea(sort.sortOrder, blobs);
                    break;
                case BY_DENSITY:
                    Util.sortByDensity(sort.sortOrder, blobs);
                    break;
                case BY_ASPECT_RATIO:
                    Util.sortByAspectRatio(sort.sortOrder, blobs);
                    break;
            }
        }
        else
        {
            // Apply a default sort by area
            Util.sortByArea(SortOrder.DESCENDING, blobs);
        }


        // Deep copy this to prevent concurrent modification exception
        userBlobs = new ArrayList<>(blobs);

        return blobs;
    }

    @Override
    public void onDrawFrame(Canvas canvas, int onscreenWidth, int onscreenHeight, float scaleBmpPxToCanvasPx, float scaleCanvasDensity, Object userContext)
    {
        ArrayList<Blob> blobs = (ArrayList<Blob>) userContext;

        contourPaint.setStrokeWidth(scaleCanvasDensity * 4);
        boundingRectPaint.setStrokeWidth(scaleCanvasDensity * 10);
        roiPaint.setStrokeWidth(scaleCanvasDensity * 10);

        android.graphics.Rect gfxRect = makeGraphicsRect(roi, scaleBmpPxToCanvasPx);

        for (Blob blob : blobs)
        {
            if (drawContours)
            {
                Path path = new Path();

                Point[] contourPts = blob.getContourPoints();

                path.moveTo((float) (contourPts[0].x) * scaleBmpPxToCanvasPx, (float)(contourPts[0].y) * scaleBmpPxToCanvasPx);
                for (int i = 1; i < contourPts.length; i++)
                {
                    path.lineTo((float) (contourPts[i].x) * scaleBmpPxToCanvasPx, (float) (contourPts[i].y) * scaleBmpPxToCanvasPx);
                }
                path.close();

                canvas.drawPath(path, contourPaint);
            }

            /*
             * Draws a rotated rect by drawing each of the 4 lines individually
             */
            Point[] rotRectPts = new Point[4];
            blob.getBoxFit().points(rotRectPts);

            for(int i = 0; i < 4; ++i)
            {
                canvas.drawLine(
                        (float) (rotRectPts[i].x)*scaleBmpPxToCanvasPx, (float) (rotRectPts[i].y)*scaleBmpPxToCanvasPx,
                        (float) (rotRectPts[(i+1)%4].x)*scaleBmpPxToCanvasPx, (float) (rotRectPts[(i+1)%4].y)*scaleBmpPxToCanvasPx,
                        boundingRectPaint
                        );
            }
        }

        canvas.drawLine(gfxRect.left, gfxRect.top, gfxRect.right, gfxRect.top, roiPaint);
        canvas.drawLine(gfxRect.right, gfxRect.top, gfxRect.right, gfxRect.bottom, roiPaint);
        canvas.drawLine(gfxRect.right, gfxRect.bottom, gfxRect.left, gfxRect.bottom, roiPaint);
        canvas.drawLine(gfxRect.left, gfxRect.bottom, gfxRect.left, gfxRect.top, roiPaint);
    }

    private android.graphics.Rect makeGraphicsRect(Rect rect, float scaleBmpPxToCanvasPx)
    {
        int left = Math.round(rect.x * scaleBmpPxToCanvasPx);
        int top = Math.round(rect.y * scaleBmpPxToCanvasPx);
        int right = left + Math.round(rect.width * scaleBmpPxToCanvasPx);
        int bottom = top + Math.round(rect.height * scaleBmpPxToCanvasPx);

        return new android.graphics.Rect(left, top, right, bottom);
    }

    @Override
    public void addFilter(BlobFilter filter)
    {
        synchronized (lockFilters)
        {
            filters.add(filter);
        }
    }

    @Override
    public void removeFilter(BlobFilter filter)
    {
        synchronized (lockFilters)
        {
            filters.remove(filter);
        }
    }

    @Override
    public void removeAllFilters()
    {
        synchronized (lockFilters)
        {
            filters.clear();
        }
    }

    @Override
    public void setSort(BlobSort sort)
    {
        this.sort = sort;
    }

    @Override
    public List<Blob> getBlobs()
    {
        return userBlobs;
    }

    
}
