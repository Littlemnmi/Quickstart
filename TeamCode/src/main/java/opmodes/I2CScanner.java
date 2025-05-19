package opmodes;

import com.qualcomm.robotcore.eventloop.opmode.LinearOpMode;
import com.qualcomm.robotcore.eventloop.opmode.TeleOp;
import com.qualcomm.robotcore.hardware.I2cAddr;
import com.qualcomm.robotcore.hardware.I2cDeviceSynchSimple;
import com.qualcomm.robotcore.hardware.I2cDeviceSynch;
import com.qualcomm.robotcore.hardware.I2cDeviceSynchImplOnSimple;

@TeleOp(name = "I2C Scanner", group = "Debug")
public class I2CScanner extends LinearOpMode {
    @Override
    public void runOpMode() {
        telemetry.addLine("Starting I2C scan...");
        telemetry.update();

        for (int i = 0; i <= 0x7F; i++) {
            I2cAddr addr = I2cAddr.create7bit(i);

            try {
                I2cDeviceSynchImplOnSimple device = hardwareMap.get(I2cDeviceSynchImplOnSimple.class, "sensor_otos");
                device.engage();

                // Try reading 1 byte from register 0
                device.read(0x00, 1);
                telemetry.addData("I2C Device Found at", String.format("0x%02X", i));
            } catch (Exception e) {
                telemetry.addData("I2C Device Found at with exception: ", e.getMessage());
                // Ignore errors; just means nothing responded at this address
            }
        }

        telemetry.update();
        waitForStart();

        while (opModeIsActive()) {
            idle();
        }
    }
}

