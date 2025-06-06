package opmodes;

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

import java.util.ArrayList;
import java.util.List;

class ColorSampleLocatorProcessorImpl extends ColorSampleLocatorProcessor implements VisionProcessor
{
    private List<ColorRange> colorRange;
    private ImageRegion roiImg;
    private Rect roi;
    private int frameWidth;
    private int frameHeight;
    private Mat roiMat;
    private Mat roiMat_userColorSpace;
    private final int contourCode;

    private Mat mask = new Mat();

    private final Paint boundingRectPaint;
    private final Paint roiPaint;
    private final Paint contourPaint;
    private final boolean drawContours;
    private final @ColorInt int boundingBoxColor;
    private final @ColorInt int roiColor;
    private final @ColorInt int contourColor;

    private final Mat erodeElement;
    private final Mat dilateElement;
    private final Size blurElement;

    private final Object lockFilters = new Object();
    private final List<BlobFilter> filters = new ArrayList<>();
    private final Mat kernelClean;
    private volatile BlobSort sort;

    private volatile ArrayList<Blob> userBlobs = new ArrayList<>();

    ColorSampleLocatorProcessorImpl(List<ColorRange> colorRange, ImageRegion roiImg, ContourMode contourMode,
                                  int erodeSize, int dilateSize, boolean drawContours, int blurSize,
                                  @ColorInt int boundingBoxColor, @ColorInt int roiColor, @ColorInt int contourColor)
    {
        // 7×7 elliptical kernel for morphological operations
        this.kernelClean = Imgproc.getStructuringElement(
                Imgproc.MORPH_ELLIPSE,
                new Size(7, 7)
        );
        this.colorRange = colorRange;
        this.roiImg = roiImg;
        this.drawContours = drawContours;
        this.boundingBoxColor = boundingBoxColor;
        this.roiColor = roiColor;
        this.contourColor = contourColor;

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
    }

    @Override
    public void init(int width, int height, CameraCalibration calibration)
    {
        frameWidth = width;
        frameHeight = height;

        roi = roiImg.asOpenCvRect(width, height);
    }

    @Override
    public Object processFrame(Mat frame, long captureTimeNanos)
    {
        filterAndProcessROI(frame);

        List<Blob> blobs = retrieveBlobsFromMask();

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

     /**
     * Compute watershed markers from a cleaned mask.
     * Returns a 32S Mat where each region has a unique marker ID.
     */
    private Mat computeMarkers(Mat maskClean, Mat inputBgr) {
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

        // Clone the BGR image so we don’t accidentally free the caller’s Mat
        Mat bgrForWatershed = inputBgr.clone();
        Imgproc.watershed(bgrForWatershed, markers);

        dist.release();
        fg.release();
        bg.release();
        unknown.release();
        bgrForWatershed.release();
        return markers;
    }

    /**
     * From labeled markers, extract rotated bounding boxes for regions above minArea.
     */
    public List<Blob> findBlobs(Mat markers) {
        Core.MinMaxLocResult mm = Core.minMaxLoc(markers);
        int maxLabel = (int) mm.maxVal;
        List<RotatedRect> boxes = new ArrayList<>();

        List<Blob> blobs = new ArrayList<>();
        
        for (int label = 2; label <= maxLabel; label++) {
            Mat region = new Mat();
            Core.compare(markers, new Scalar(label), region, Core.CMP_EQ);
            region.convertTo(region, CvType.CV_8U, 255);

            List<MatOfPoint> contours = new ArrayList<>();
            Mat hierarchy = new Mat();
            Imgproc.findContours(region, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
            hierarchy.release();
            for (MatOfPoint contour : contours) {
                if (!contour.empty() && Imgproc.contourArea(contour) > 100) {
                    Core.add(contour, new Scalar(roi.x, roi.y), contour);
                    blobs.add(new BlobImpl(contour));
                }
                
            }
            region.release();
        }
        return blobs;
    }

    /**
     * Clean up a binary mask via morphological closing then opening.
     */
    private Mat cleanMask(Mat mask) {
        Mat cleaned = new Mat();
        Imgproc.morphologyEx(mask, cleaned, Imgproc.MORPH_CLOSE, kernelClean, new Point(-1, -1), 2);
        Mat cleaned2 = new Mat();
        Imgproc.morphologyEx(cleaned, cleaned2, Imgproc.MORPH_OPEN,  kernelClean, new Point(-1, -1), 2);
        cleaned.release();
        return cleaned2;
    }

    private List<Blob> retrieveBlobsFromMask() {
        /*ArrayList<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(mask, contours, hierarchy, contourCode, Imgproc.CHAIN_APPROX_SIMPLE);
        hierarchy.release();

        List<Blob> blobs = new ArrayList<>();
        for (MatOfPoint contour : contours)
        {
            Core.add(contour, new Scalar(roi.x, roi.y), contour);
            blobs.add(new BlobImpl(contour));
        }
        return blobs;*/

        Mat maskClean = cleanMask(mask);

        // Compute markers for watershed
        Mat markers = computeMarkers(maskClean, roiMat_userColorSpace);

        //renderImage(markers, "markers_" + color.name().toLowerCase() + ".png");

        List<Blob> blobs = findBlobs(markers);
        maskClean.release();
        markers.release();
        return blobs;
    }

    private void filterAndProcessROI(Mat frame) {
        if (roiMat == null)
        {
            roiMat = frame.submat(roi);
            roiMat_userColorSpace = roiMat.clone();
        }

        if (colorRange.get(0).colorSpace == ColorSpace.YCrCb)
        {
            Imgproc.cvtColor(roiMat, roiMat_userColorSpace, Imgproc.COLOR_RGB2YCrCb);
        }
        else if (colorRange.get(0).colorSpace == ColorSpace.HSV)
        {
            Imgproc.cvtColor(roiMat, roiMat_userColorSpace, Imgproc.COLOR_RGB2HSV);
        }
        else if (colorRange.get(0).colorSpace == ColorSpace.RGB)
        {
            Imgproc.cvtColor(roiMat, roiMat_userColorSpace, Imgproc.COLOR_RGBA2RGB);
        }

        if (blurElement != null)
        {
            Imgproc.GaussianBlur(roiMat_userColorSpace, roiMat_userColorSpace, blurElement, 0);
        }

        createColorRangeMask();
        

        if (erodeElement != null)
        {
            Imgproc.erode(mask, mask, erodeElement);
        }

        if (dilateElement != null)
        {
            Imgproc.dilate(mask, mask, dilateElement);
        }
    }

    private void createColorRangeMask() {
        // Process multiple color ranges by combining their masks
        mask.release(); // Release any previously used mask
        mask = new Mat(roiMat.rows(), roiMat.cols(), CvType.CV_8U, new Scalar(0)); // Start with empty mask
        
        Mat tempMask = new Mat();
        
        for (int i = 0; i < colorRange.size(); i++) {
            ColorRange range = colorRange.get(i);
            Core.inRange(roiMat_userColorSpace, range.min, range.max, tempMask);
            
            if (i == 0) {
                // For first range, just copy to the mask
                tempMask.copyTo(mask);
            } else { 
                // For subsequent ranges, OR with the existing mask
                Core.bitwise_or(mask, tempMask, mask);
            }
        }
        
        tempMask.release();
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

    class BlobImpl extends Blob
    {
        private MatOfPoint contour;
        private Point[] contourPts;
        private int area = -1;
        private double density = -1;
        private double aspectRatio = -1;
        private RotatedRect rect;
        private double x = -1;
        private double y = -1;
        private double z = -1;
        private double angle = -1;

        BlobImpl(MatOfPoint contour)
        {
            this.contour = contour;
        }

        @Override
        public MatOfPoint getContour()
        {
            return contour;
        }

        @Override
        public Point[] getContourPoints()
        {
            if (contourPts == null)
            {
                contourPts = contour.toArray();
            }

            return contourPts;
        }

        @Override
        public int getContourArea()
        {
            if (area < 0)
            {
                area = Math.max(1, (int) Imgproc.contourArea(contour));  //  Fix zero area issue
            }

            return area;
        }

        @Override
        public double getDensity()
        {
            Point[] contourPts = getContourPoints();

            if (density < 0)
            {
                // Compute the convex hull of the contour
                MatOfInt hullMatOfInt = new MatOfInt();
                Imgproc.convexHull(contour, hullMatOfInt);

                // The convex hull calculation tells us the INDEX of the points which
                // which were passed in eariler which form the convex hull. That's all
                // well and good, but now we need filter out that original list to find
                // the actual POINTS which form the convex hull
                Point[] hullPoints = new Point[hullMatOfInt.rows()];
                List<Integer> hullContourIdxList = hullMatOfInt.toList();

                for (int i = 0; i < hullContourIdxList.size(); i++)
                {
                    hullPoints[i] = contourPts[hullContourIdxList.get(i)];
                }

                double hullArea = Math.max(1.0,Imgproc.contourArea(new MatOfPoint(hullPoints)));  //  Fix zero area issue

                density = getContourArea() / hullArea;
            }
            return density;
        }

        @Override
        public double getAspectRatio()
        {
            if (aspectRatio < 0)
            {
                RotatedRect r = getBoxFit();

                double longSize  = Math.max(1, Math.max(r.size.width, r.size.height));
                double shortSize = Math.max(1, Math.min(r.size.width, r.size.height));

                aspectRatio = longSize / shortSize;
            }

            return aspectRatio;
        }

        @Override
        public RotatedRect getBoxFit()
        {
            if (rect == null)
            {
                rect = Imgproc.minAreaRect(new MatOfPoint2f(getContourPoints()));
            }
            return rect;
        }

        @Override
        public double getX()
        {
            return x;
        }

        @Override
        public double getY()
        {
            return y;
        }

        @Override
        public double getZ()
        {
            return z;
        }

        @Override
        public double getAngle()
        {
            return angle;
        }
    }
}
