
union{

  float numf;
  uint8_t numb[4];

}convert;

void i2c_test(void){
	
	uint8_t i =0, j;
	for(i = 0; i!=255; i++){
	    Serial.print(i);
	    Serial.print(", ");
	    Wire.beginTransmission(i);
	    Wire.write(1);
	    j = Wire.endTransmission();
	    Serial.println(j);
	    if(j == 0)
	      Serial.println("-------------------------------------------");
	
	}
	
	    Serial.print(255);
	    Serial.print(", ");
	    Wire.beginTransmission(255);
	    Wire.write(1);
	    Serial.println(Wire.endTransmission());
	
}

void send_float(float num){
	
	convert.numf = num;
    Serial.write(convert.numb[0]);
    Serial.write(convert.numb[1]);
    Serial.write(convert.numb[2]);
    Serial.write(convert.numb[3]);
	
}

void send_header(){
	
	Serial.write(0x56);
    Serial.write(0x45);
    Serial.write(0x0A);
	
}

void send_footer(){
	
	Serial.write(0x69);
	
}
