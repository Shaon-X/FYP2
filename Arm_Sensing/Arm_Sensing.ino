// #define Matlab

#include "Functions\functions.h"

mpu_t mpu, mpu2;
uint32_t tt, ttbuff;


void setup() {

  pinMode(2, OUTPUT);
  digitalWrite(2, 1);
  Serial.begin(115200);
  Wire.begin();

  delay(2000);
  mpu6050_init(&mpu, 1, 0.6, 0.9, 0.5, 0.001, 0.0596, 0.0957, -0.0022, 0.0217, -0.0079, 0.0201);
  mpu6050_init(&mpu2, 0, 0.6, 0.9, 0.5, 0.001, 0.001, 0.0539910, -0.00019, -0.0213948, -0.000104, -0.0650226);
  delay(100);
  digitalWrite(2, 0);
  tt = 0;

#ifdef Matlab
  while(Serial.available())
    Serial.read();
  while(!Serial.available());
#endif
  
}

void loop() {
  
  ttbuff = millis() - tt;
  if(ttbuff >= 10){

    tt = millis();
    mpu6050_read(&mpu, ttbuff);
    mpu6050_read(&mpu2, ttbuff);

#ifndef Matlab

    send_header();
    send_float(-mpu.R[0][1]);
    send_float(-mpu.R[1][1]);
    send_float(-mpu.R[2][1]);
    send_float(mpu.R[0][2]);
    send_float(mpu.R[1][2]);
    send_float(mpu.R[2][2]);
    send_float(mpu2.R[0][1]);
    send_float(mpu2.R[1][1]);
    send_float(mpu2.R[2][1]);
    send_float(mpu2.R[0][2]);
    send_float(mpu2.R[1][2]);
    send_float(mpu2.R[2][2]);
    send_float((((float)analogRead(34)) - 230.0) / 2370.0);
    send_footer();
#else

    Serial.write(mpu2.x_acc_byte[0]);
    Serial.write(mpu2.x_acc_byte[1]);
    Serial.write(mpu2.y_acc_byte[0]);
    Serial.write(mpu2.y_acc_byte[1]);
    Serial.write(mpu2.z_acc_byte[0]);
    Serial.write(mpu2.z_acc_byte[1]);

    Serial.write(mpu2.x_gyro_byte[0]);
    Serial.write(mpu2.x_gyro_byte[1]);
    Serial.write(mpu2.y_gyro_byte[0]);
    Serial.write(mpu2.y_gyro_byte[1]);
    Serial.write(mpu2.z_gyro_byte[0]);
    Serial.write(mpu2.z_gyro_byte[1]);

    Serial.write(mpu2.x_mag_byte[0]);
    Serial.write(mpu2.x_mag_byte[1]);
    Serial.write(mpu2.y_mag_byte[0]);
    Serial.write(mpu2.y_mag_byte[1]);
    Serial.write(mpu2.z_mag_byte[0]);
    Serial.write(mpu2.z_mag_byte[1]);

#endif

  }

}
