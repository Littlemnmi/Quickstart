package opmodes;

import android.annotation.SuppressLint;
import android.util.Size;

import com.qualcomm.robotcore.eventloop.opmode.LinearOpMode;
import com.qualcomm.robotcore.eventloop.opmode.TeleOp;

//import org.firstinspires.ftc.teamcode.robot.camera.CameraColor;
//import org.firstinspires.ftc.teamcode.robot.camera.DetectedSample;
//import org.firstinspires.ftc.teamcode.robot.camera.SampleDetectionPipeline;
//import org.firstinspires.ftc.teamcode.robot.camera.SampleImageProcessor;
import org.firstinspires.ftc.vision.VisionPortal;
import org.firstinspires.ftc.robotcore.external.hardware.camera.WebcamName;
//import org.firstinspires.ftc.teamcode.robot.camera.SampleDetectionProcessor;

import org.opencv.core.RotatedRect;

import java.util.List;
import java.util.Arrays;
import java.util.ArrayList;

@TeleOp(name="Sample Detection", group="Test")
public class SampleDetectionOpMode extends LinearOpMode {

    private VisionPortal visionPortal;
    private SampleDetectionProcessor processor;

    @SuppressLint("DefaultLocale")
    @Override
    public void runOpMode() {
        processor = new SampleDetectionProcessor(CameraColor.RED);

        ColorSampleLocatorProcessor colorLocator = new ColorSampleLocatorProcessor.Builder()
            .setTargetColorRange(ColorRange.BLUE_HSV)        // use a predefined color match
            .setContourMode(ColorSampleLocatorProcessor.ContourMode.EXTERNAL_ONLY)    // exclude blobs inside blobs
            .setRoi(ImageRegion.asUnityCenterCoordinates(-0.95, 0.95, 0.95, -0.95))  
            .setDrawContours(true)                        // Show contours on the Stream Preview
            .setBlurSize(5)                               // Smooth the transitions between different colors in image
            .build();
        colorLocator.addFilter(new ColorSampleLocatorProcessor.BlobFilter(ColorSampleLocatorProcessor.BlobCriteria.BY_CONTOUR_AREA, 1000, 20000)); // Filter out very small blobs
        //colorLocator.addFilter(new ColorSampleLocatorProcessor.BlobFilter(ColorSampleLocatorProcessor.BlobCriteria.BY_ASPECT_RATIO, 0.5, 2.0)); // Filter out blobs that are too wide or too narrow
        // Build the VisionPortal
        visionPortal = new VisionPortal.Builder()                
                .addProcessor(colorLocator)
                .setCameraResolution(new Size(640, 480))
                .setStreamFormat(VisionPortal.StreamFormat.MJPEG)
                .setCamera(hardwareMap.get(WebcamName.class, "Webcam 1"))
                .build();


        telemetry.addLine("Ready! Press Start.");
        telemetry.update();


        while (opModeIsActive() || opModeInInit()) {
             // Read the current list
            List<ColorSampleLocatorProcessor.Blob> blobs = colorLocator.getBlobs();

            telemetry.addLine(" Area  Density  Aspect Center");

            // Display the size (area) and center location for each Blob.
            for(ColorSampleLocatorProcessor.Blob b : blobs)
            {
                RotatedRect boxFit = b.getBoxFit();
                telemetry.addLine(String.format("%5d  %4.2f   %5.2f  (%3d,%3d)",
                          b.getContourArea(), b.getDensity(), b.getAspectRatio(), (int) boxFit.center.x, (int) boxFit.center.y));
            }


            // Access processed sample data
            List<DetectedSample> samples = processor.getDetectedSamples();

            for (DetectedSample sample : samples) {
                telemetry.addData("Sample", String.format("(%.1f,%.1f) %.1f\" %.1f°",
                        sample.boundingBox.center.x, sample.boundingBox.center.y,
                        sample.distanceInches, sample.angleDegrees));
            }


            telemetry.update();
            sleep(1000);
        }
    }
}

