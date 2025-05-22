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

import org.firstinspires.ftc.vision.opencv.ColorBlobLocatorProcessor;
import org.firstinspires.ftc.vision.opencv.ColorRange;
import org.firstinspires.ftc.vision.opencv.ImageRegion;
import org.opencv.core.RotatedRect;

import java.util.List;

@TeleOp(name="Sample Detection", group="Test")
public class SampleDetectionOpMode extends LinearOpMode {

    private VisionPortal visionPortal;
    private SampleDetectionProcessor processor;

    @SuppressLint("DefaultLocale")
    @Override
    public void runOpMode() {
        processor = new SampleDetectionProcessor(CameraColor.RED);

        ColorBlobLocatorProcessor colorLocator = new ColorBlobLocatorProcessor.Builder()
            .setTargetColorRange(ColorRange.BLUE)         // use a predefined color match
            .setContourMode(ColorBlobLocatorProcessor.ContourMode.EXTERNAL_ONLY)    // exclude blobs inside blobs
            .setRoi(ImageRegion.asUnityCenterCoordinates(-0.5, 0.5, 0.5, -0.5))  // search central 1/4 of camera view
            .setDrawContours(true)                        // Show contours on the Stream Preview
            .setBlurSize(5)                               // Smooth the transitions between different colors in image
            .build();

        // Build the VisionPortal
        visionPortal = new VisionPortal.Builder()                
                .addProcessor(processor)
                .setCameraResolution(new Size(640, 480))
                .setStreamFormat(VisionPortal.StreamFormat.MJPEG)
                .setCamera(hardwareMap.get(WebcamName.class, "Webcam 1"))
                .build();


        telemetry.addLine("Ready! Press Start.");
        telemetry.update();


        while (opModeIsActive() || opModeInInit()) {
             // Read the current list
            List<ColorBlobLocatorProcessor.Blob> blobs = colorLocator.getBlobs();
            ColorBlobLocatorProcessor.Util.filterByArea(1000, 20000, blobs);  // filter out very small blobs.

            telemetry.addLine(" Area  Density  Aspect Center");

            // Display the size (area) and center location for each Blob.
            for(ColorBlobLocatorProcessor.Blob b : blobs)
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

