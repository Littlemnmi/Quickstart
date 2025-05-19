package opmodes;

import com.qualcomm.robotcore.eventloop.opmode.LinearOpMode;
import com.qualcomm.robotcore.eventloop.opmode.TeleOp;
import com.qualcomm.robotcore.hardware.I2cAddr;
import com.qualcomm.robotcore.hardware.I2cDeviceSynchSimple;
import com.qualcomm.robotcore.hardware.I2cDeviceSynch;
import com.qualcomm.robotcore.hardware.I2cDeviceSynchImplOnSimple;

import com.qualcomm.hardware.sparkfun.SparkFunOTOS;


@TeleOp(name = "OTOS I2C Test (Simple)", group = "Test")
public class OtosTestOpMode extends LinearOpMode {

    /*private I2cDeviceSynch otosSynch;

    @Override
    public void runOpMode() {
        // This is the 7-bit I2C address for SparkFun OTOS or VL53L1X (check your datasheet!)
        I2cAddr i2cAddr = I2cAddr.create7bit(0x30);  // 0x30 → 7-bit, corresponds to 0x60 8-bit address

        // Create the Simple I2C device from the SDK
        I2cDeviceSynchImplOnSimple otosSynch = hardwareMap.get(I2cDeviceSynchImplOnSimple.class, "sensor_otos");

        // Wrap the simple device with a Synch wrapper
        otosSynch.engage();

        telemetry.addLine("OTOS sensor ready");
        telemetry.update();
        waitForStart();

        while (opModeIsActive()) {
            // Read 2 bytes from register 0x01 (replace with correct register for your sensor)
            byte[] data = otosSynch.read(0x01, 2);

            int distance = ((data[0] & 0xFF) << 8) | (data[1] & 0xFF);

            telemetry.addData("Distance", distance + " mm");
            telemetry.update();
        }
    }*/
    private SparkFunOTOS otos;

    @Override
    public void runOpMode() throws InterruptedException {
        otos = hardwareMap.get(SparkFunOTOS.class, "sensor_otos");
        telemetry.addData("otos connected", otos.isConnected());

        waitForStart();

        while (opModeIsActive()) {
            telemetry.addData("x", otos.getPosition().x);
            telemetry.addData("y", otos.getPosition().y);
            telemetry.addData("h", otos.getPosition().h);
            telemetry.update();
        }
    }
}


