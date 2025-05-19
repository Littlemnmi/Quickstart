package opmodes;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;

import org.firstinspires.ftc.robotcore.internal.camera.calibration.CameraCalibration;
import org.firstinspires.ftc.vision.VisionProcessor;
import org.opencv.core.*;


public class SampleDetectionProcessor implements VisionProcessor {

    private SampleImageProcessor pipeline = new SampleImageProcessor();

    // Store the last processed frame for drawing (optional)
    private Mat lastFrame = null;

    private CameraColor targetColor;

    public SampleDetectionProcessor(CameraColor color) {

        targetColor = color;

    }


    @Override
    public void init(int width, int height, CameraCalibration calibration) {

    }

    @Override
    public Object processFrame(Mat input, long captureTimeNanos) {
        if (lastFrame != null) {
            lastFrame.release();
        }
        lastFrame = input.clone(); // Save a copy if you want to draw later

        pipeline.detectSamples(input, targetColor, 500);

        // You can return detection results (optional)
        return pipeline;
    }

    @Override
    public void onDrawFrame(Canvas canvas, int onscreenWidth, int onscreenHeight, float scaleBmpPxToCanvasPx, float scaleCanvasDensity, Object userContext) {
        if (canvas == null) return;

        Paint greenPaint = new Paint();
        greenPaint.setColor(Color.GREEN);
        greenPaint.setStyle(Paint.Style.STROKE);
        greenPaint.setStrokeWidth(4);

        Paint textPaint = new Paint();
        textPaint.setColor(android.graphics.Color.WHITE);
        textPaint.setTextSize(30);

        // Draw yellow samples
        for (DetectedSample sample : pipeline.samples) {
            drawSample(canvas, sample, greenPaint, textPaint, "Y: " + String.format("%.0f\"", sample.distanceInches));
        }
        
    }


    private void drawSample(Canvas canvas,
                            DetectedSample sample,
                            Paint boxPaint,
                            Paint textPaint,
                            String label) {
        // 1) Get the 4 vertices of the rotated rectangle
        org.opencv.core.Point[] cvPts = new org.opencv.core.Point[4];
        sample.boundingBox.points(cvPts);

        // 2) Convert to Android PointF and build a Path
        Path boxPath = new Path();
        boxPath.moveTo((float)cvPts[0].x, (float)cvPts[0].y);
        for (int i = 1; i < 4; i++) {
            boxPath.lineTo((float)cvPts[i].x, (float)cvPts[i].y);
        }
        boxPath.close();

        // 3) Draw the rotated rectangle
        canvas.drawPath(boxPath, boxPaint);

        // 4) Draw the label at the top-left corner of the box (with a little offset)
        float textX = (float)cvPts[0].x;
        float textY = (float)cvPts[0].y - 10;  // 10px above the corner
        canvas.drawText(label, textX, textY, textPaint);
    }

    public SampleImageProcessor getPipeline() {
        return pipeline;
    }
}

