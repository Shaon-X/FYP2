#include <Wire.h>
#include <stdint.h>


#include "functions.c"
#include "MPU6050\mpu6050.h"

void i2c_test(void);
void send_float(float num);
void send_header();
void send_footer();

//change FS_SEL and AFS_SEL if data not correct
//I2C 400kHz
//address: 0110100X
//sampling rate = 500Hz	

//reg104 & 106, reset signal path

//reg36 master control
