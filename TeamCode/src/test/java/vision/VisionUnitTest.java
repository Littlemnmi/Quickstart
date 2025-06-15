package vision;
import static org.junit.Assert.*;

import org.firstinspires.ftc.teamcode.robot.vision.*;
import org.junit.BeforeClass;
import org.junit.Test;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.InputStream;
import java.util.List;
import java.util.Locale;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import java.util.ArrayList;

public class VisionUnitTest {

    private static Mat kernelClean = null;
    private static Mat erodeElem = null; //Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
    private static Mat dilateElem = null; //Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
    private static Size blurSize = null; //new Size(5, 5);
    private static double minBlobArea = 500; // Minimum area for a blob to be considered valid
    private static double minBlobAspectRatio = 0.5; // Minimum aspect ratio for a blob to be considered valid
    private static double maxBlobAspectRatio = 2.0; // Maximum aspect ratio for a blob to be considered valid
    private static Mat cameraMatrix = null; // Default camera matrix
    private static MatOfDouble distCoeffs = null; // Default distortion coefficients

    @BeforeClass
    public static void setup() {
        try {
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
            kernelClean = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(7, 7));
            cameraMatrix = ColorSampleLocatorProcessorImpl.configureCameraMatrix();
            distCoeffs = ColorSampleLocatorProcessorImpl.configureDistCoeffs();
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Warning: OpenCV native library not loaded. Some tests may fail.");
        }
    }

    private Mat loadImage(String name) throws Exception {
        InputStream is = getClass().getClassLoader().getResourceAsStream(name);

        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        byte[] data = new byte[1024];
        int nRead;
        while ((nRead = is.read(data, 0, data.length)) != -1) {
            buffer.write(data, 0, nRead);
        }
        buffer.flush();
        byte[] imageBytes = buffer.toByteArray();
        is.close();

        Mat bgr = Imgcodecs.imdecode(new MatOfByte(imageBytes), Imgcodecs.IMREAD_COLOR);
        // Convert BGR → RGB
        Mat rgb = new Mat();
        Imgproc.cvtColor(bgr, rgb, Imgproc.COLOR_BGR2RGB);
        bgr.release();
        return rgb;
    }

    private void writeAnnotatedImage(
            List<Blob> samples,
            Mat img,
            String outputFileName) {
        // Convert RGB -> BGR
        Mat bgr = new Mat();
        Imgproc.cvtColor(img, bgr, Imgproc.COLOR_BGR2RGB);
        // 1) Draw each rotated rect in its respective color
        drawRotatedRects(samples, bgr, new Scalar(0, 255, 0));
        // 2) Ensure output directory exists (optional)
        File outDir = new File("build/outputs");
        if (!outDir.exists()) {
            outDir.mkdirs();
        }

        // 3) Write the annotated image
        String fullPath = new File(outDir, outputFileName + ".png").getAbsolutePath();
        boolean ok = Imgcodecs.imwrite(fullPath, bgr);
        if (!ok) {
            System.err.println("Failed to write image to " + fullPath);
        }
    }

    private void drawRotatedRects(
            List<Blob> samples,
            Mat img,
            Scalar color) {
        for (Blob s : samples) {
            // Get the 4 corners of the RotatedRect
            //Point[] pts = s.getContourPoints();
            Point[] pts = new Point[4];
            s.getBoxFit().points(pts);

            // Draw the 4 sides
            for (int i = 0; i < 4; i++) {
                Point p1 = pts[i];
                Point p2 = pts[(i + 1) % 4];
                Imgproc.line(img, p1, p2, color, 3);
            }
        }
    }


    private void printBlobs(List<Blob> samples) {
        System.out.println("Area   Aspect   Center   Position");
        for (Blob b : samples) {
            System.out.println(String.format("%5d  %5.2f  (%3d,%3d) (%4.2f,%4.2f, %4.2f,%4.2f)",
                          b.getContourArea(),  b.getAspectRatio(), (int) b.getBoxFit().center.x, (int) b.getBoxFit().center.y, b.getX(), b.getY(), b.getZ(), b.getAngle()));
        }
    }



    @Test
    public void testBlobDetection1() throws Exception {
        Mat img = loadImage("image_1.jfif");
        
        // Create color ranges for YELLOW detection
        List<ColorRange> yellowRanges = new ArrayList<>();
        yellowRanges.add(ColorRange.YELLOW_HSV); // Use predefined yellow HSV range
        
        // Create the processor with proper parameters
        ColorSampleImageProcessor processor = new ColorSampleImageProcessor(
            yellowRanges,
            kernelClean,  // kernel for clean
            erodeElem,     // erode element
            dilateElem,     // dilate element
            blurSize,                                                       // blur size
            cameraMatrix,                 // camera matrix
            distCoeffs,                                           // distortion coeffs
            Imgproc.RETR_EXTERNAL
        );
        
        // Process the image
        List<Blob> blobs = processor.process(img, new Rect(0, 0, img.width(), img.height()));
        
        assertFalse("No blobs detected", blobs.isEmpty());
        printBlobs(blobs);  // You'll need to update this method to handle Blob objects
        writeAnnotatedImage(blobs, img, "annotated_image_1");  // You'll need to update this too
    }



    @Test
    public void testBlobDetection2() throws Exception {
        Mat img = loadImage("image_2.jfif");
        // Create color ranges for YELLOW detection
        List<ColorRange> redRanges = new ArrayList<>();
        redRanges.add(ColorRange.RED_HSV_LOWER);
        redRanges.add(ColorRange.RED_HSV_UPPER);

        // Create the processor with proper parameters
        ColorSampleImageProcessor processor = new ColorSampleImageProcessor(
                redRanges,
                kernelClean,  // kernel for clean
                erodeElem,     // erode element
                dilateElem,     // dilate element
                blurSize,                                                       // blur size
                cameraMatrix,                 // camera matrix
                distCoeffs,                                            // distortion coeffs
                Imgproc.RETR_EXTERNAL
        );

        // Process the image
        List<Blob> blobs = processor.process(img, new Rect(0, 0, img.width(), img.height()));

        assertFalse("No blobs detected", blobs.isEmpty());
        printBlobs(blobs);  // You'll need to update this method to handle Blob objects
        writeAnnotatedImage(blobs, img, "annotated_image_2");  // You'll need to update this too

    }


    @Test
    public void testBlobDetection3() throws Exception {
        Mat img = loadImage("image_3.jfif");
        // Create color ranges for YELLOW detection
        List<ColorRange> blueRanges = new ArrayList<>();
        blueRanges.add(ColorRange.BLUE_HSV);

        // Create the processor with proper parameters
        ColorSampleImageProcessor processor = new ColorSampleImageProcessor(
                blueRanges,
                kernelClean,  // kernel for clean
                erodeElem,     // erode element
                dilateElem,     // dilate element
                blurSize,                                                       // blur size
                cameraMatrix,                 // camera matrix
                distCoeffs,                                            // distortion coeffs
                Imgproc.RETR_EXTERNAL
        );

        // Process the image
        List<Blob> blobs = processor.process(img, new Rect(0, 0, img.width(), img.height()));

        assertFalse("No blobs detected", blobs.isEmpty());
        printBlobs(blobs);  // You'll need to update this method to handle Blob objects
        writeAnnotatedImage(blobs, img, "annotated_image_3");  // You'll need to update this too

    }

}
