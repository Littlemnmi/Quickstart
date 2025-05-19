package opmodes;

import org.opencv.core.RotatedRect;

public class DetectedSample {
    public RotatedRect boundingBox;
    public double distanceInches;
    public double angleDegrees; // orientation relative to camera

    public DetectedSample(RotatedRect rect, double distance, double angle) {
        this.boundingBox = rect;
        this.distanceInches = distance;
        this.angleDegrees = angle;
    }
}
