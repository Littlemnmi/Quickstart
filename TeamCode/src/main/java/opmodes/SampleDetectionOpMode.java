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

@TeleOp(name="Sample Detection", group="Test")
public class SampleDetectionOpMode extends LinearOpMode {

    private VisionPortal visionPortal;
    private SampleDetectionProcessor processor;

    @SuppressLint("DefaultLocale")
    @Override
    public void runOpMode() {
        processor = new SampleDetectionProcessor(CameraColor.BLUE);

        // Build the VisionPortal
        visionPortal = new VisionPortal.Builder()                
                .addProcessor(processor)
                .setCameraResolution(new Size(640, 480))
                .setStreamFormat(VisionPortal.StreamFormat.MJPEG)
                .setCamera(hardwareMap.get(WebcamName.class, "Webcam 1"))
                .build();


        telemetry.addLine("Ready! Press Start.");
        telemetry.update();
        waitForStart();

        while (opModeIsActive()) {
            telemetry.addLine("opmode is active.");
            // Access processed sample data
            SampleImageProcessor pipeline = processor.getPipeline();

            for (DetectedSample sample : pipeline.samples) {
                telemetry.addData("Sample", String.format("(%.1f,%.1f) %.1f\" %.1f°",
                        sample.boundingBox.center.x, sample.boundingBox.center.y,
                        sample.distanceInches, sample.angleDegrees));
            }


            telemetry.update();
            sleep(5000);
        }
    }
}

