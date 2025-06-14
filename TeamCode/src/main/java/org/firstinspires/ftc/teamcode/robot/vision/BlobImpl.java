package org.firstinspires.ftc.teamcode.robot.vision;

import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint2f;
import org.opencv.imgproc.Imgproc;

import java.util.List;

public class BlobImpl extends Blob
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

    public BlobImpl(MatOfPoint contour)
    {
        this.contour = contour;
    }

    @Override
    public MatOfPoint getContour()
    {
        return contour;
    }

    @Override
    public org.opencv.core.Point[] getContourPoints()
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
    public void setPosition(double x, double y, double z, double angle)
    {
        this.x = x;
        this.y = y;
        this.z = z;
        this.angle = angle;
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
