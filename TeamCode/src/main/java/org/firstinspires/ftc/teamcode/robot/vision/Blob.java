package org.firstinspires.ftc.teamcode.robot.vision;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;

/**
 * Class describing a Blob of color found inside the image
 */
public abstract class Blob
{
    /**
     * Get the OpenCV contour for this blob
     * @return OpenCV contour
     */
    public abstract MatOfPoint getContour();

    /**
     * Get the contour points for this blob
     *
     * @return contour points for this blob
     */
    public abstract Point[] getContourPoints();

    /**
     * Get the area enclosed by this blob's contour
     * @return area enclosed by this blob's contour
     */
    public abstract int getContourArea();

    /**
     * Get the density of this blob, i.e. ratio of
     * contour area to convex hull area
     * @return density of this blob
     */
    public abstract double getDensity();

    /**
     * Get the aspect ratio of this blob, i.e. the ratio
     * of longer side of the bounding box to the shorter side
     * @return aspect ratio of this blob
     */
    public abstract double getAspectRatio();

    /**
     * Get a "best fit" bounding box for this blob
     * @return "best fit" bounding box for this blob
     */
    public abstract RotatedRect getBoxFit();

    /**
     * Get the x translation of the center point of this blob
     * relative to the center of the image
     * @return x translation of the center point of this blob (inchs)
     */
    public abstract double getX();

    /**
     * Get the y translation of the center point of this blob
     * relative to the center of the image
     * @return y translation of the center point of this blob (inchs)
     */
    public abstract double getY();

    /**
     * Get the z distance of the center point of this blob
     * relative to the camera
     * @return z distance of the center point of this blob (inches)
     */
    public abstract double getZ();

    /**
     * Get the angle of this blob relative to the camera
     * @return angle of this blob relative to the camera (degrees)
     */
    public abstract double getAngle();
    
    /**
     * Set the position of this blob in the world
     * @param x x translation of the center point of this blob (inches)
     * @param y y translation of the center point of this blob (inches)
     * @param z z distance of the center point of this blob (inches)
     * @param angle angle of this blob relative to the camera (degrees)
     */
    public abstract void setPosition(double x, double y, double z, double angle);

}

