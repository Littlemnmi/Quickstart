package org.firstinspires.ftc.teamcode.robot.vision;

import android.graphics.Color;

import androidx.annotation.ColorInt;

import com.qualcomm.robotcore.util.SortOrder;

import org.firstinspires.ftc.vision.VisionProcessor;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * The {@link ColorSampleLocatorProcessor} finds "blobs" of a user-specified color
 * in the image. You can restrict the search area to a specified Region
 * of Interest (ROI).
 */
public abstract class ColorSampleLocatorProcessor implements VisionProcessor
{
    /**
     * Class supporting construction of a {@link ColorSampleLocatorProcessor}
     */
    public static class Builder
    {
        private List<ColorRange> colorRange = new ArrayList<>();
        private ContourMode contourMode;
        private ImageRegion imageRegion;
        private int erodeSize = -1;
        private int dilateSize = -1;
        private boolean drawContours = false;
        private int blurSize = -1;
        private int boundingBoxColor = Color.rgb(255, 120, 31);
        private int roiColor = Color.rgb(255, 255, 255);
        private int contourColor = Color.rgb(3, 227, 252);

        /**
         * Sets whether to draw the contour outline for the detected
         * blobs on the camera preview. This can be helpful for debugging
         * thresholding.
         * @param drawContours whether to draw contours on the camera preview
         * @return Builder object, to allow for method chaining
         */
        public Builder setDrawContours(boolean drawContours)
        {
            this.drawContours = drawContours;
            return this;
        }

        /**
         * Set the color used to draw the "best fit" bounding boxes for blobs
         * @param color Android color int
         * @return Builder object, to allow for method chaining
         */
        public Builder setBoxFitColor(@ColorInt int color)
        {
            this.boundingBoxColor = color;
            return this;
        }

        /**
         * Set the color used to draw the ROI on the camera preview
         * @param color Android color int
         * @return Builder object, to allow for method chaining
         */
        public Builder setRoiColor(@ColorInt int color)
        {
            this.roiColor = color;
            return this;
        }

        /**
         * Set the color used to draw blob contours on the camera preview
         * @param color Android color int
         * @return Builder object, to allow for method chaining
         */
        public Builder setContourColor(@ColorInt int color)
        {
            this.contourColor = color;
            return this;
        }

        /**
         * Set the color range used to find blobs
         * @param colorRange the color range used to find blobs
         * @return Builder object, to allow for method chaining
         */
        public Builder setTargetColorRange(ColorRange colorRange)
        {
            this.colorRange.add(colorRange);
            return this;
        }

        /**
         * Set the contour mode which will be used when generating
         * the results provided by {@link #getBlobs()}
         * @param contourMode contour mode which will be used when generating
         *                    the results provided by {@link #getBlobs()}
         * @return Builder object, to allow for method chaining
         */
        public Builder setContourMode(ContourMode contourMode)
        {
            this.contourMode = contourMode;
            return this;
        }

        /**
         * Set the Region of Interest on which to perform blob detection
         * @param roi region of interest
         * @return Builder object, to allow for method chaining
         */
        public Builder setRoi(ImageRegion roi)
        {
            this.imageRegion = roi;
            return this;
        }

        /**
         * Set the size of the blur kernel. Blurring can improve
         * color thresholding results by smoothing color variation.
         * @param blurSize size of the blur kernel
         *                 0 to disable
         * @return Builder object, to allow for method chaining
         */
        public Builder setBlurSize(int blurSize)
        {
            this.blurSize = blurSize;
            return this;
        }

        /**
         * Set the size of the Erosion operation performed after applying
         * the color threshold. Erosion eats away at the mask, reducing
         * noise by eliminating super small areas, but also reduces the
         * contour areas of everything a little bit.
         * @param erodeSize size of the Erosion operation
         *                  0 to disable
         * @return Builder object, to allow for method chaining
         */
        public Builder setErodeSize(int erodeSize)
        {
            this.erodeSize = erodeSize;
            return this;
        }

        /**
         * Set the size of the Dilation operation performed after applying
         * the Erosion operation. Dilation expands mask areas, making up
         * for shrinkage caused during erosion, and can also clean up results
         * by closing small interior gaps in the mask.
         * @param dilateSize the size of the Dilation operation performed
         *                   0 to disable
         * @return Builder object, to allow for method chaining
         */
        public Builder setDilateSize(int dilateSize)
        {
            this.dilateSize = dilateSize;
            return this;
        }

        /**
         * Construct a {@link ColorSampleLocatorProcessor} object using previously
         * set parameters
         * @return a {@link  ColorSampleLocatorProcessor} object which can be attached
         * to your {@link org.firstinspires.ftc.vision.VisionPortal}
         */
        public ColorSampleLocatorProcessor build()
        {
            if (colorRange == null)
            {
                throw new IllegalArgumentException("You must set a color range!");
            }

            if (contourMode == null)
            {
                throw new IllegalArgumentException("You must set a contour mode!");
            }

            return new ColorSampleLocatorProcessorImpl(colorRange, imageRegion, contourMode, erodeSize, dilateSize, drawContours, blurSize, boundingBoxColor, roiColor, contourColor);
        }
    }

    /**
     * Determines what you get in {@link #getBlobs()}
     */
    public enum ContourMode
    {
        /**
         * Only return blobs from external contours
         */
        EXTERNAL_ONLY,

        /**
         * Return blobs which may be from nested contours
         */
        ALL_FLATTENED_HIERARCHY
    }

    /**
     * The criteria used for filtering and sorting.
     */
    public enum BlobCriteria
    {
        BY_CONTOUR_AREA,
        BY_DENSITY,
        BY_ASPECT_RATIO,
    }

    /**
     * Class describing how to filter blobs.
     */
    public static class BlobFilter {
        public final BlobCriteria criteria;
        public final double minValue;
        public final double maxValue;

        public BlobFilter(BlobCriteria criteria, double minValue,  double maxValue)
        {
            this.criteria = criteria;
            this.minValue = minValue;
            this.maxValue = maxValue;
        }
    }

    /**
     * Class describing how to sort blobs.
     */
    public static class BlobSort
    {
        public final BlobCriteria criteria;
        public final SortOrder sortOrder;

        public BlobSort(BlobCriteria criteria, SortOrder sortOrder)
        {
            this.criteria = criteria;
            this.sortOrder = sortOrder;
        }
    }

    
    /**
     * Add a filter.
     */
    public abstract void addFilter(BlobFilter filter);

    /**
     * Remove a filter.
     */
    public abstract void removeFilter(BlobFilter filter);

    /**
     * Remove all filters.
     */
    public abstract void removeAllFilters();

    /**
     * Sets the sort.
     */
    public abstract void setSort(BlobSort sort);

    /**
     * Get the results of the most recent blob analysis
     * @return results of the most recent blob analysis
     */
    public abstract List<Blob> getBlobs();

    /**
     * Utility class for post-processing results from {@link #getBlobs()}
     */
    public static class Util
    {
        /**
         * Remove from a List of Blobs those which fail to meet an area criteria
         * @param minArea minimum area
         * @param maxArea maximum area
         * @param blobs List of Blobs to operate on
         */
        public static void filterByArea(double minArea, double maxArea, List<Blob> blobs)
        {
            ArrayList<Blob> toRemove = new ArrayList<>();

            for(Blob b : blobs)
            {
                if (b.getContourArea() > maxArea || b.getContourArea() < minArea)
                {
                    toRemove.add(b);
                }
            }

            blobs.removeAll(toRemove);
        }

        /**
         * Sort a list of Blobs based on area
         * @param sortOrder sort order
         * @param blobs List of Blobs to operate on
         */
        public static void sortByArea(SortOrder sortOrder, List<Blob> blobs)
        {
            blobs.sort(new Comparator<Blob>()
            {
                public int compare(Blob c1, Blob c2)
                {
                    int tmp = (int)Math.signum(c2.getContourArea() - c1.getContourArea());

                    if (sortOrder == SortOrder.ASCENDING)
                    {
                        tmp = -tmp;
                    }

                    return tmp;
                }
            });
        }

        /**
         * Remove from a List of Blobs those which fail to meet a density criteria
         * @param minDensity minimum density
         * @param maxDensity maximum desnity
         * @param blobs List of Blobs to operate on
         */
        public static void filterByDensity(double minDensity, double maxDensity, List<Blob> blobs)
        {
            ArrayList<Blob> toRemove = new ArrayList<>();

            for(Blob b : blobs)
            {
                if (b.getDensity() > maxDensity || b.getDensity() < minDensity)
                {
                    toRemove.add(b);
                }
            }

            blobs.removeAll(toRemove);
        }

        /**
         * Sort a list of Blobs based on density
         * @param sortOrder sort order
         * @param blobs List of Blobs to operate on
         */
        public static void sortByDensity(SortOrder sortOrder, List<Blob> blobs)
        {
            blobs.sort(new Comparator<Blob>()
            {
                public int compare(Blob c1, Blob c2)
                {
                    int tmp = (int)Math.signum(c2.getDensity() - c1.getDensity());

                    if (sortOrder == SortOrder.ASCENDING)
                    {
                        tmp = -tmp;
                    }

                    return tmp;
                }
            });
        }

        /**
         * Remove from a List of Blobs those which fail to meet an aspect ratio criteria
         * @param minAspectRatio minimum aspect ratio
         * @param maxAspectRatio maximum aspect ratio
         * @param blobs List of Blobs to operate on
         */
        public static void filterByAspectRatio(double minAspectRatio, double maxAspectRatio, List<Blob> blobs)
        {
            ArrayList<Blob> toRemove = new ArrayList<>();

            for(Blob b : blobs)
            {
                if (b.getAspectRatio() > maxAspectRatio || b.getAspectRatio() < minAspectRatio)
                {
                    toRemove.add(b);
                }
            }

            blobs.removeAll(toRemove);
        }

        /**
         * Sort a list of Blobs based on aspect ratio
         * @param sortOrder sort order
         * @param blobs List of Blobs to operate on
         */
        public static void sortByAspectRatio(SortOrder sortOrder, List<Blob> blobs)
        {
            blobs.sort(new Comparator<Blob>()
            {
                public int compare(Blob c1, Blob c2)
                {
                    int tmp = (int)Math.signum(c2.getAspectRatio() - c1.getAspectRatio());

                    if (sortOrder == SortOrder.ASCENDING)
                    {
                        tmp = -tmp;
                    }

                    return tmp;
                }
            });
        }
    }
}
