package opmodes;
import org.opencv.core.Scalar;

/**
 * An {@link ColorRange represents a 3-channel minimum/maximum
 * range for a given color space}
 */
public class ColorRange
{
    protected final ColorSpace colorSpace;
    protected final Scalar min;
    protected final Scalar max;

    // -----------------------------------------------------------------------------
    // DEFAULT OPTIONS
    // -----------------------------------------------------------------------------

    public static final ColorRange BLUE = new ColorRange(
            ColorSpace.YCrCb,
            new Scalar( 16,   0, 155),
            new Scalar(255, 127, 255)
    );

    public static final ColorRange RED = new ColorRange(
            ColorSpace.YCrCb,
            new Scalar( 32, 176,  0),
            new Scalar(255, 255, 132)
    );

    public static final ColorRange YELLOW = new ColorRange(
            ColorSpace.YCrCb,
            new Scalar( 32, 128,   0),
            new Scalar(255, 170, 120)
    );

    public static final ColorRange GREEN = new ColorRange(
            ColorSpace.YCrCb,
            new Scalar( 32,   0,   0),
            new Scalar(255, 120, 133)
    );


    // Red lower range (0-10 degrees in hue circle)
    public static final ColorRange RED_HSV_LOWER = new ColorRange(
        ColorSpace.HSV,
        new Scalar(0, 100, 100),
        new Scalar(10, 255, 255)
    );

    // Red upper range (160-180 degrees in hue circle)
    public static final ColorRange RED_HSV_UPPER = new ColorRange(
        ColorSpace.HSV,
        new Scalar(160, 100, 100),
        new Scalar(180, 255, 255)
    );

    public static final ColorRange BLUE_HSV = new ColorRange(
        ColorSpace.HSV,
        new Scalar(75, 100, 100),
        new Scalar(145, 255, 255)
    );

    public static final ColorRange YELLOW_HSV = new ColorRange(
        ColorSpace.HSV,
        new Scalar(20, 100, 100),
        new Scalar(35, 255, 255)
    );

    public static final ColorRange GREEN_HSV = new ColorRange(
        ColorSpace.HSV,
        new Scalar(40, 100, 100),
        new Scalar(80, 255, 255)
    );

    // -----------------------------------------------------------------------------
    // ROLL YOUR OWN
    // -----------------------------------------------------------------------------

    public ColorRange(ColorSpace colorSpace, Scalar min, Scalar max)
    {
        this.colorSpace = colorSpace;
        this.min = min;
        this.max = max;
    }
}
