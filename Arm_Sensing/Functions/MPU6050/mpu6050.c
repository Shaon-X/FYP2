

//change FS_SEL and AFS_SEL if data not correct
//I2C 400kHz
//address: 0110100X
//sampling rate = 500Hz	

//reg104 & 106, reset signal path

//reg36 master control

typedef struct{
	
	union{
	
	int16_t x_mag_raw;
	uint8_t x_mag_byte[2];
	
	};
	
	float x_mag;
	
	union{
	
	int16_t y_mag_raw;
	uint8_t y_mag_byte[2];
	
	};
	
	float y_mag;
	
	union{
	
	int16_t z_mag_raw;
	uint8_t z_mag_byte[2];
	
	};
	
	float z_mag;
	
	union{
	
	int16_t x_gyro_raw;
	uint8_t x_gyro_byte[2];
	
	};
	
	float x_gyro;
	
	union{
	
	int16_t y_gyro_raw;
	uint8_t y_gyro_byte[2];
	
	};
	
	float y_gyro;
	
	union{
	
	int16_t z_gyro_raw;
	uint8_t z_gyro_byte[2];
	
	};
	
	float z_gyro;
	
	union{
	
	int16_t x_acc_raw;
	uint8_t x_acc_byte[2];
	
	};
	
	float x_acc;
	
	union{
	
	int16_t y_acc_raw;
	uint8_t y_acc_byte[2];
	
	};
	
	float y_acc;
	
	union{
	
	int16_t z_acc_raw;
	uint8_t z_acc_byte[2];
	
	};
	
	float z_acc;
	
	uint8_t addr;
	
	float R[3][3];
	//	ratio, nratio, staticratio, nstaticratio, maglim, sin(maglim), vellim, xvel, xmag, yvel, ymag, zvel, zmag
	float para[13];
	
}mpu_t;

void mpu6050_init(mpu_t* p_mpu, uint8_t AD0, float ratio, float statratio, float maglim, float vellim, float xvel, float xmag, float yvel, float ymag, float zvel, float zmag){
	
	p_mpu->addr =  0b01101000 | AD0;
	p_mpu->para[0] = ratio;
	p_mpu->para[1] = 1.0 - ratio;
	p_mpu->para[2] = statratio;
	p_mpu->para[3] = 1.0 - statratio;
	p_mpu->para[4] = maglim;
	p_mpu->para[5] = sin(maglim);
	p_mpu->para[6] = vellim;
	p_mpu->para[7] = xvel;
	p_mpu->para[8] = xmag;
	p_mpu->para[9] = yvel;
	p_mpu->para[10] = ymag;
	p_mpu->para[11] = zvel;
	p_mpu->para[12] = zmag;
	p_mpu->R[0][0] = 1;
	p_mpu->R[1][0] = 0;
	p_mpu->R[2][0] = 0;
	p_mpu->R[0][1] = 0;
	p_mpu->R[1][1] = 1;
	p_mpu->R[2][1] = 0;
	p_mpu->R[0][2] = 0;
	p_mpu->R[1][2] = 0;
	p_mpu->R[2][2] = 1;
	
	Wire.setClock(400000);
	delay(35);
	
	Wire.beginTransmission(p_mpu->addr);//start bit & addr & write bit
	Wire.write(55);						//INT_PIN_CFG
	Wire.write(0b00000010);				//I2C bypass to slave
	Wire.endTransmission(1);			//send data with stop bit
	
	Wire.beginTransmission(p_mpu->addr);//start bit & addr & write bit
	Wire.write(107);					//PWR_MGMT_1
	Wire.write(0b00000001);				//wake from sleep, PLL with X axis gyro
	Wire.endTransmission(1);			//send data with stop bit
	
	Wire.beginTransmission(0x0D);		//start bit & addr & write bit
	Wire.write(11);						//addr
	Wire.write(1);						//set/reset period = 1
	Wire.endTransmission(1);			//send data with stop bit
	
	Wire.beginTransmission(0x0D);		//start bit & addr & write bit
	Wire.write(9);						//addr
	Wire.write(0b00001101);				//+-2G, 200Hz, continuous mode
	Wire.endTransmission(1);			//send data with stop bit	
	
	Wire.beginTransmission(0x0D);		//start bit & addr & write bit
	Wire.write(10);						//addr
	Wire.write(0b01000001);				//disable INT pin, turn on roll function
	Wire.endTransmission(1);			//send data with stop bit
	
	Wire.beginTransmission(p_mpu->addr);//start bit & addr & write bit
	Wire.write(55);						//INT_PIN_CFG
	Wire.write(0b00000000);				//turn off I2C bypass to slave
	Wire.endTransmission(1);			//send data with stop bit
	
	Wire.beginTransmission(p_mpu->addr);//start bit & addr & write bit
	Wire.write(37);						//I2C_SLV0_ADDR
	Wire.write(0b10001101);				//read operation, slave address
	Wire.endTransmission(1);			//send data with stop bit
	
	Wire.beginTransmission(p_mpu->addr);//start bit & addr & write bit
	Wire.write(25);						//SMPLRT_DIV
	Wire.write(15);						//set sampling rate of 500Hz
	Wire.endTransmission(1);			//send data with stop bit	
	
	Wire.beginTransmission(p_mpu->addr);//start bit & addr & write bit
	Wire.write(27);						//GYRO_CONFIG
	Wire.write(0b1000);					//set gyro range 500deg/s
	Wire.write(0b1000);					//set accel range +-4g
	Wire.endTransmission(1);			//send data with stop bit
	
	Wire.beginTransmission(p_mpu->addr);//start bit & addr & write bit
	Wire.write(39);						//I2C_SLV0_CTRL
	Wire.write(0b10000110);				//number of bytes from slave, enable slave
	Wire.endTransmission(1);			//send data with stop bit
	
	Wire.beginTransmission(p_mpu->addr);//start bit & addr & write bit
	Wire.write(106);					//USER_CTRL
	Wire.write(0b00100000);				//Enable I2C Master
	Wire.endTransmission(1);			//send data with stop bit
	
}

uint8_t mpu6050_read(mpu_t* p_mpu, uint32_t ttpass){
	
	Wire.beginTransmission(p_mpu->addr);
	Wire.write(59);
	Wire.endTransmission(0);
	if(Wire.requestFrom((int)p_mpu->addr, 20, 1)!=20){
		while(Wire.available()){
			uint8_t buff = Wire.read();
		}
		return 1;
	}
	
	float Rn[3][3];
	float matbuff[3][3];
	float matbuff2[3][3];

	p_mpu->x_acc_byte[1] = Wire.read();
	p_mpu->x_acc_byte[0] = Wire.read();
	p_mpu->y_acc_byte[1] = Wire.read();
	p_mpu->y_acc_byte[0] = Wire.read();
	p_mpu->z_acc_byte[1] = Wire.read();
	p_mpu->z_acc_byte[0] = Wire.read();
	
	p_mpu->x_gyro_byte[1] = Wire.read();	//buffers for temperature data
	p_mpu->x_gyro_byte[1] = Wire.read();
	
	p_mpu->x_gyro_byte[1] = Wire.read();
	p_mpu->x_gyro_byte[0] = Wire.read();
	p_mpu->y_gyro_byte[1] = Wire.read();
	p_mpu->y_gyro_byte[0] = Wire.read();
	p_mpu->z_gyro_byte[1] = Wire.read();
	p_mpu->z_gyro_byte[0] = Wire.read();
	
	p_mpu->x_mag_byte[0] = Wire.read();
	p_mpu->x_mag_byte[1] = Wire.read();
	p_mpu->y_mag_byte[0] = Wire.read();
	p_mpu->y_mag_byte[1] = Wire.read();
	p_mpu->z_mag_byte[0] = Wire.read();
	p_mpu->z_mag_byte[1] = Wire.read();
	
	p_mpu->x_acc = ((float)p_mpu->x_acc_raw) / 8191.875;
	p_mpu->y_acc = ((float)p_mpu->y_acc_raw) / 8191.875;
	p_mpu->z_acc = ((float)p_mpu->z_acc_raw) / 8191.875;
	
	Rn[0][0] = ((float)ttpass)/1000.0;
	
	p_mpu->x_gyro = ((((float)p_mpu->x_gyro_raw) / 3754.5924) + p_mpu->para[7])*Rn[0][0];
	p_mpu->y_gyro = ((((float)p_mpu->y_gyro_raw) / 3754.5924) + p_mpu->para[9])*Rn[0][0];
	p_mpu->z_gyro = ((((float)p_mpu->z_gyro_raw) / 3754.5924) + p_mpu->para[11])*Rn[0][0];
	
	p_mpu->x_mag = (((float)p_mpu->y_mag_raw) / 16383.75) + p_mpu->para[8];
	p_mpu->y_mag = (-((float)p_mpu->x_mag_raw) / 16383.75) + p_mpu->para[10];
	p_mpu->z_mag = (((float)p_mpu->z_mag_raw) / 16383.75) + p_mpu->para[12];
  	
  	Rn[0][0] = sqrt(p_mpu->x_acc*p_mpu->x_acc + p_mpu->y_acc*p_mpu->y_acc + p_mpu->z_acc*p_mpu->z_acc);
  	p_mpu->x_acc = p_mpu->x_acc / Rn[0][0];
  	p_mpu->y_acc = p_mpu->y_acc / Rn[0][0];
  	p_mpu->z_acc = p_mpu->z_acc / Rn[0][0];
  	
  	Rn[0][0] = sqrt(p_mpu->x_mag*p_mpu->x_mag + p_mpu->y_mag*p_mpu->y_mag + p_mpu->z_mag*p_mpu->z_mag);
  	p_mpu->x_mag = p_mpu->x_mag / Rn[0][0];
  	p_mpu->y_mag = p_mpu->y_mag / Rn[0][0];
  	p_mpu->z_mag = p_mpu->z_mag / Rn[0][0];
  	
  	Rn[0][0] = acos(p_mpu->x_acc*p_mpu->x_mag+p_mpu->y_acc*p_mpu->y_mag+p_mpu->z_acc*p_mpu->z_mag);
    Rn[1][1] = sin((1.5708-Rn[0][0])/2);
    Rn[2][1] = cos((1.5708-Rn[0][0])/2);
    Rn[0][1] = Rn[1][1] * Rn[1][1] - Rn[2][1] * Rn[2][1];
    
    Rn[0][2] = (Rn[1][1]*p_mpu->x_mag-Rn[2][1]*p_mpu->x_acc) / Rn[0][1];
    Rn[1][2] = (Rn[1][1]*p_mpu->y_mag-Rn[2][1]*p_mpu->y_acc) / Rn[0][1];
    Rn[2][2] = (Rn[1][1]*p_mpu->z_mag-Rn[2][1]*p_mpu->z_acc) / Rn[0][1];
    
    Rn[0][0] = (p_mpu->x_acc - Rn[2][1]*Rn[0][2]) / Rn[1][1];
    Rn[1][0] = (p_mpu->y_acc - Rn[2][1]*Rn[1][2]) / Rn[1][1];
    Rn[2][0] = (p_mpu->z_acc - Rn[2][1]*Rn[2][2]) / Rn[1][1];
    
    Rn[0][1] = Rn[1][2]*Rn[2][0] - Rn[2][2]*Rn[1][0];
    Rn[1][1] = Rn[2][2]*Rn[0][0] - Rn[0][2]*Rn[2][0];
    Rn[2][1] = Rn[0][2]*Rn[1][0] - Rn[1][2]*Rn[0][0];
    
    matbuff[0][0] = Rn[1][1]*Rn[2][2] - Rn[1][2]*Rn[2][1];
    matbuff[1][0] = Rn[1][2]*Rn[2][0] - Rn[1][0]*Rn[2][2];
    matbuff[2][0] = Rn[1][0]*Rn[2][1] - Rn[1][1]*Rn[2][0];
    matbuff[2][2] = Rn[0][0]*matbuff[0][0] + Rn[0][1]*matbuff[1][0] + Rn[0][2]*matbuff[2][0];
    matbuff[0][1] = (Rn[0][2]*Rn[2][1] - Rn[0][1]*Rn[2][2]) / matbuff[2][2];
    matbuff[1][1] = (Rn[0][0]*Rn[2][2] - Rn[0][2]*Rn[2][0]) / matbuff[2][2];
    matbuff[2][1] = (Rn[0][1]*Rn[2][0] - Rn[0][0]*Rn[2][1]) / matbuff[2][2];
    matbuff[0][2] = (Rn[0][1]*Rn[1][2] - Rn[1][1]*Rn[0][2]) / matbuff[2][2];
    matbuff[1][2] = (Rn[0][2]*Rn[1][0] - Rn[0][0]*Rn[1][2]) / matbuff[2][2];
    matbuff[2][2] = (Rn[0][0]*Rn[1][1] - Rn[0][1]*Rn[1][0]) / matbuff[2][2];
    
    matbuff2[0][0] = p_mpu->R[1][1]*p_mpu->R[2][2] - p_mpu->R[1][2]*p_mpu->R[2][1];
    matbuff2[1][0] = p_mpu->R[1][2]*p_mpu->R[2][0] - p_mpu->R[1][0]*p_mpu->R[2][2];
    matbuff2[1][2] = p_mpu->R[0][0]*matbuff2[0][0] + p_mpu->R[0][1]*matbuff2[1][0] + p_mpu->R[0][2]*(p_mpu->R[1][0]*p_mpu->R[2][1] - p_mpu->R[1][1]*p_mpu->R[2][0]);
    matbuff2[0][0] = matbuff2[0][0] / matbuff2[1][2];
    matbuff2[1][0] = matbuff2[1][0] / matbuff2[1][2];
    matbuff2[0][1] = (p_mpu->R[0][2]*p_mpu->R[2][1] - p_mpu->R[0][1]*p_mpu->R[2][2]) / matbuff2[1][2];
    matbuff2[1][1] = (p_mpu->R[0][0]*p_mpu->R[2][2] - p_mpu->R[0][2]*p_mpu->R[2][0]) / matbuff2[1][2];
    matbuff2[0][2] = (p_mpu->R[0][1]*p_mpu->R[1][2] - p_mpu->R[1][1]*p_mpu->R[0][2]) / matbuff2[1][2];
    matbuff2[1][2] = (p_mpu->R[0][2]*p_mpu->R[1][0] - p_mpu->R[0][0]*p_mpu->R[1][2]) / matbuff2[1][2];
    
  	Rn[0][1] = matbuff2[0][0]*matbuff[0][1] + matbuff2[0][1]*matbuff[1][1] + matbuff2[0][2]*matbuff[2][1];
  	Rn[0][2] = matbuff2[0][0]*matbuff[0][2] + matbuff2[0][1]*matbuff[1][2] + matbuff2[0][2]*matbuff[2][2];
    Rn[1][2] = matbuff2[1][0]*matbuff[0][2] + matbuff2[1][1]*matbuff[1][2] + matbuff2[1][2]*matbuff[2][2];
  	  	
	
	if (Rn[0][2] > p_mpu->para[4])
		matbuff2[1][2] = p_mpu->para[5];
	else if (Rn[0][2] < -p_mpu->para[4])
		matbuff2[1][2] = -p_mpu->para[5];
	else
		matbuff2[1][2] = asin(Rn[0][2]);
		
	if (Rn[1][2]/cos(matbuff2[1][2]) > p_mpu->para[4])
        matbuff2[0][2] = -p_mpu->para[5];
    else if (Rn[1][2]/cos(matbuff2[1][2]) < -p_mpu->para[4])
        matbuff2[0][2] = p_mpu->para[5];
    else
        matbuff2[0][2] = -asin(Rn[1][2]/cos(matbuff2[1][2]));
        
	if (Rn[0][1]/cos(matbuff2[1][2]) > p_mpu->para[4])
        matbuff2[2][2] = -p_mpu->para[5];
    else if (Rn[0][1]/cos(matbuff2[1][2]) < -p_mpu->para[4])
        matbuff2[2][2] = p_mpu->para[5];
    else
        matbuff2[2][2] = -asin(Rn[0][1]/cos(matbuff2[1][2]));

	if (p_mpu->x_gyro < p_mpu->para[6] && p_mpu->x_gyro > -p_mpu->para[6])
        p_mpu->x_gyro = p_mpu->para[2]*p_mpu->x_gyro + p_mpu->para[3]*matbuff2[0][2];
    else
        p_mpu->x_gyro = p_mpu->para[0]*p_mpu->x_gyro + p_mpu->para[1]*matbuff2[0][2];

	if (p_mpu->y_gyro < p_mpu->para[6] && p_mpu->y_gyro > -p_mpu->para[6])
        p_mpu->y_gyro = p_mpu->para[2]*p_mpu->y_gyro + p_mpu->para[3]*matbuff2[1][2];
    else
        p_mpu->y_gyro = p_mpu->para[0]*p_mpu->y_gyro + p_mpu->para[1]*matbuff2[1][2];

    if (p_mpu->z_gyro < p_mpu->para[6] && p_mpu->z_gyro > -p_mpu->para[6])
        p_mpu->z_gyro = p_mpu->para[2]*p_mpu->z_gyro + p_mpu->para[3]*matbuff2[2][2];
    else
        p_mpu->z_gyro = p_mpu->para[0]*p_mpu->z_gyro + p_mpu->para[1]*matbuff2[2][2];
        
	Rn[0][0] = cos(p_mpu->y_gyro)*cos(p_mpu->z_gyro);
	Rn[1][0] = sin(p_mpu->x_gyro)*sin(p_mpu->y_gyro)*cos(p_mpu->z_gyro)+cos(p_mpu->x_gyro)*sin(p_mpu->z_gyro);
	Rn[2][0] = sin(p_mpu->x_gyro)*sin(p_mpu->z_gyro)-cos(p_mpu->x_gyro)*sin(p_mpu->y_gyro)*cos(p_mpu->z_gyro);
	Rn[0][2] = sin(p_mpu->y_gyro);
	Rn[1][2] = -sin(p_mpu->x_gyro)*cos(p_mpu->y_gyro);
	Rn[2][2] = cos(p_mpu->x_gyro)*cos(p_mpu->y_gyro);
        
    Rn[0][1] = Rn[1][2]*Rn[2][0] - Rn[2][2]*Rn[1][0];
    Rn[1][1] = Rn[2][2]*Rn[0][0] - Rn[0][2]*Rn[2][0];
    Rn[2][1] = Rn[0][2]*Rn[1][0] - Rn[1][2]*Rn[0][0];
    
    matbuff[0][0] = p_mpu->R[0][0];
    matbuff[1][0] = p_mpu->R[1][0];
    matbuff[2][0] = p_mpu->R[2][0];
    matbuff[0][1] = p_mpu->R[0][1];
    matbuff[1][1] = p_mpu->R[1][1];
    matbuff[2][1] = p_mpu->R[2][1];
    matbuff[0][2] = p_mpu->R[0][2];
    matbuff[1][2] = p_mpu->R[1][2];
    matbuff[2][2] = p_mpu->R[2][2];
	
	    
    p_mpu->R[0][0] = matbuff[0][0]*Rn[0][0] + matbuff[0][1]*Rn[1][0] + matbuff[0][2]*Rn[2][0];
    p_mpu->R[1][0] = matbuff[1][0]*Rn[0][0] + matbuff[1][1]*Rn[1][0] + matbuff[1][2]*Rn[2][0];
    p_mpu->R[2][0] = matbuff[2][0]*Rn[0][0] + matbuff[2][1]*Rn[1][0] + matbuff[2][2]*Rn[2][0];
    p_mpu->R[0][1] = matbuff[0][0]*Rn[0][1] + matbuff[0][1]*Rn[1][1] + matbuff[0][2]*Rn[2][1];
    p_mpu->R[1][1] = matbuff[1][0]*Rn[0][1] + matbuff[1][1]*Rn[1][1] + matbuff[1][2]*Rn[2][1];
    p_mpu->R[2][1] = matbuff[2][0]*Rn[0][1] + matbuff[2][1]*Rn[1][1] + matbuff[2][2]*Rn[2][1];
    p_mpu->R[0][2] = matbuff[0][0]*Rn[0][2] + matbuff[0][1]*Rn[1][2] + matbuff[0][2]*Rn[2][2];
    p_mpu->R[1][2] = matbuff[1][0]*Rn[0][2] + matbuff[1][1]*Rn[1][2] + matbuff[1][2]*Rn[2][2];
    p_mpu->R[2][2] = matbuff[2][0]*Rn[0][2] + matbuff[2][1]*Rn[1][2] + matbuff[2][2]*Rn[2][2];

	
	return 0;
	
}
