
#include "mpu6050.c"

void mpu6050_init(mpu_t* p_mpu, uint8_t AD0, float ratio, float statratio, float maglim, float vellim, float xvel, float xmag, float yvel, float ymag, float zvel, float zmag);
uint8_t mpu6050_read(mpu_t* p_mpu, uint32_t ttpass);

