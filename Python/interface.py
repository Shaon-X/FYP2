# leave trail?
# use simple_sphere instead of sphere if too laggy

from vpython import *
import serial
import struct
import math
import serial.tools.list_ports
import csv
import time
import numpy as np
import winsound
import random

from prediction import preprocess_input_live


class SerialClass:
    def __init__(self, port, hz, width, window):
        flag = 0
        try:
            self.serial = serial.Serial(port, 115200, timeout=0)
        except:
            flag = 1

        if flag:
            print("Failed to connect to port, checking available ports...")

            ports = list(serial.tools.list_ports.comports())
            print(ports)
            for p in ports:
                print(p)
            print("Done")
            while True:
                pass
        else:
            self.cnt = 0
            self.width = width
            self.window = window
            self.rate = 1/hz
            self.angle = 0
            self.x = 0
            self.y = 0
            self.ai = []
            self.ait = 0
            self.prediction = []
            self.state = 0
            self.datbuff = b'\x00\x00\x00\x00'
            self.fdat: list[float] = [1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2]
            self.scene = canvas(width=745, height=710, background=color.black)
            # self.scene.userzoom = False
            # self.scene.userspin = False
            # self.scene.userpan = False
            self.scene.autoscale = False
            self.scene.camera.pos = vector(2.5, 40, 50)
            self.scene.forward = vector(0, 0, -1)
            self.scene.up = vector(0, 1, 0)
            sphere_rad = 4
            sphere_color = color.cyan
            self.box_length = [26, 29]
            box_thick = 5
            self.palm_width = 8
            self.palm_length = 10
            self.OriginSphere = sphere(pos=vec(0, 0, 0), radius=sphere_rad, color=sphere_color)
            self.arm = box(pos=vector(0, 0, -self.box_length[0]/2), axis=vector(0, 0, -1), color=color.white, size=vector(self.box_length[0], box_thick, box_thick))
            self.joint = sphere(pos=vec(0, 0, -self.box_length[0]), radius=sphere_rad, color=sphere_color)
            self.hand = box(pos=vector(self.box_length[1]/2, 0, -self.box_length[0]), axis=vector(1, 0, 0), color=color.white, size=vector(self.box_length[1], box_thick, box_thick))
            self.palm1 = box(pos=vector(self.box_length[1]+(self.palm_length/2), 0, -self.box_length[0]+self.palm_width/4), axis=vector(1, 0, 0), color=color.white, size=vector(self.palm_length, box_thick, self.palm_width/2))
            self.palm2 = box(pos=vector(self.box_length[1]+(self.palm_length/2), 0, -self.box_length[0]-self.palm_width/4), axis=vector(1, 0, 0), color=color.white, size=vector(self.palm_length, box_thick, self.palm_width/2))
            self.label = label(text = ' ', pos=vector(0, 50, 20))

            # -20, 25, gap 15
            thickness = 0.5
            self.line1 = box(pos=vector(-20, 40, -self.box_length[0]-5), axis=vector(0, 1, 0), length = 30, heigth = 1, width = thickness, up = vector(0, 0, 1))
            self.line2 = box(pos=vector(-5, 40, -self.box_length[0]-5), axis=vector(0, 1, 0), length=30, heigth=1, width=thickness, up=vector(0, 0, 1))
            self.line3 = box(pos=vector(10, 40, -self.box_length[0]-5), axis=vector(0, 1, 0), length=30, heigth=1, width=thickness, up=vector(0, 0, 1))
            self.line4 = box(pos=vector(25, 40, -self.box_length[0]-5), axis=vector(0, 1, 0), length=30, heigth=1, width=thickness, up=vector(0, 0, 1))
            self.line5 = box(pos=vector(2.5, 55, -self.box_length[0]-5), axis=vector(1, 0, 0), length=45 + thickness, heigth=1, width=thickness, up=vector(0, 0, 1))
            self.line6 = box(pos=vector(2.5, 40, -self.box_length[0]-5), axis=vector(1, 0, 0), length=45 + thickness, heigth=1, width=thickness, up=vector(0, 0, 1))
            self.line7 = box(pos=vector(2.5, 25, -self.box_length[0]-5), axis=vector(1, 0, 0), length=45 + thickness, heigth=1, width=thickness, up=vector(0, 0, 1))
            self.indicator = box(pos=vector(0, 0, 100), length=15-thickness, width=15-thickness, height=1, up=vector(0, 0, 1))
            self.target = sphere(pos= vector(-12.5, 47.5, -self.box_length[0] - 5), radius = 3, color = color.red)
            self.target_t = 0
            self.target_state = 1
            self.arm_state = 0

            self.serial.reset_input_buffer()
            self.time = time.time()


def stop():
    while True:
        pass

def angle_bias(s):
    dat = [0]*2
    angle = [math.sin(s.angle), math.cos(s.angle)]
    dat[0] = s.fdat[0] * angle[1] + s.fdat[1] * angle[0]
    dat[1] = s.fdat[1] * angle[1] - s.fdat[0] * angle[0]
    s.fdat[0] = dat[0]
    s.fdat[1] = dat[1]
    dat[0] = s.fdat[3] * angle[1] + s.fdat[4] * angle[0]
    dat[1] = s.fdat[4] * angle[1] - s.fdat[3] * angle[0]
    s.fdat[3] = dat[0]
    s.fdat[4] = dat[1]
    dat[0] = s.fdat[6] * angle[1] + s.fdat[7] * angle[0]
    dat[1] = s.fdat[7] * angle[1] - s.fdat[6] * angle[0]
    s.fdat[6] = dat[0]
    s.fdat[7] = dat[1]
    dat[0] = s.fdat[9] * angle[1] + s.fdat[10] * angle[0]
    dat[1] = s.fdat[10] * angle[1] - s.fdat[9] * angle[0]
    s.fdat[9] = dat[0]
    s.fdat[10] = dat[1]


def update_serial(s, param):

    if s.serial.inWaiting() > 120:
        s.serial.reset_input_buffer()
    elif s.serial.inWaiting():
        if s.state == 0:
            if s.serial.read() == b'\x56':
                s.state = 1
        elif s.state == 1:
            if s.serial.read() == b'\x45':
                s.state = 2
            else:
                s.state = 0
        elif s.state == 2:
            if s.serial.read() == b'\x0A':
                s.state = 3
            else:
                s.state = 0
        elif s.state == 3 and s.serial.inWaiting() >= 53:
            dat = s.serial.read(52)
            if s.serial.read() == b'\x69':
                for x in [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]:
                    s.fdat[int(x/4)], *_ = struct.unpack('f', dat[x:x+4])
                angle_bias(s)
                s.x = s.fdat[0] * s.box_length[0] + s.fdat[6] * s.box_length[1]
                s.y = s.fdat[1] * s.box_length[0] + s.fdat[7] * s.box_length[1]
                s.cnt = s.cnt + 1
                if s.cnt == s.width:
                    s.ai.append([s.x, s.y])
                    if 25 <= s.y < 40:
                        if -20 <= s.x < -5:
                            s.ai[len(s.ai) - 1].append(4)
                        elif -5 <= s.x < 10:
                            s.ai[len(s.ai) - 1].append(5)
                        elif 10 <= s.x < 25:
                            s.ai[len(s.ai) - 1].append(6)
                        else:
                            s.ai[len(s.ai) - 1].append(0)
                    elif 40 <= s.y < 55:
                        if -20 <= s.x < -5:
                            s.ai[len(s.ai) - 1].append(1)
                        elif -5 <= s.x < 10:
                            s.ai[len(s.ai) - 1].append(2)
                        elif 10 <= s.x < 25:
                            s.ai[len(s.ai) - 1].append(3)
                        else:
                            s.ai[len(s.ai) - 1].append(0)
                    else:
                        s.ai[len(s.ai) - 1].append(0)
                    preprocess_input_live(s.ai, param)
                    if len(s.ai) > s.window:
                        s.ai.pop(0)
                    s.cnt = 0
            s.state = 0

    if time.time() - s.time >= s.rate:
        s.time = time.time()
        s.arm.axis = vector(s.fdat[0], s.fdat[1], s.fdat[2]) * s.box_length[0]
        s.arm.up = vector(s.fdat[3], s.fdat[4], s.fdat[5])
        s.arm.pos = s.arm.axis / 2
        s.joint.pos = s.arm.axis
        s.hand.axis = vector(s.fdat[6], s.fdat[7], s.fdat[8]) * s.box_length[1]
        s.hand.up = vector(s.fdat[9], s.fdat[10], s.fdat[11])
        s.hand.pos = s.joint.pos + s.hand.axis / 2
        s.fdat[12] = s.fdat[12] * 0.5236
        s.palm1.axis = vector(s.fdat[6] * math.cos(s.fdat[12]) + s.fdat[9] * math.sin(s.fdat[12]),
                              s.fdat[7] * math.cos(s.fdat[12]) + s.fdat[10] * math.sin(s.fdat[12]),
                              s.fdat[8] * math.cos(s.fdat[12]) + s.fdat[11] * math.sin(s.fdat[12])) * s.palm_length
        s.palm1.up = vector(s.fdat[9] * math.cos(s.fdat[12]) - s.fdat[6] * math.sin(s.fdat[12]),
                            s.fdat[10] * math.cos(s.fdat[12]) - s.fdat[7] * math.sin(s.fdat[12]),
                            s.fdat[11] * math.cos(s.fdat[12]) - s.fdat[8] * math.sin(s.fdat[12]))
        s.palm1.pos = s.joint.pos + s.hand.axis + s.palm1.up * s.palm_width / 4 + s.palm1.axis / 2
        s.palm2.axis = vector(s.fdat[6] * math.cos(-s.fdat[12]) + s.fdat[9] * math.sin(-s.fdat[12]),
                              s.fdat[7] * math.cos(-s.fdat[12]) + s.fdat[10] * math.sin(-s.fdat[12]),
                              s.fdat[8] * math.cos(-s.fdat[12]) + s.fdat[11] * math.sin(-s.fdat[12])) * s.palm_length
        s.palm2.up = vector(s.fdat[9] * math.cos(-s.fdat[12]) - s.fdat[6] * math.sin(-s.fdat[12]),
                            s.fdat[10] * math.cos(-s.fdat[12]) - s.fdat[7] * math.sin(-s.fdat[12]),
                            s.fdat[11] * math.cos(-s.fdat[12]) - s.fdat[8] * math.sin(-s.fdat[12]))
        s.palm2.pos = s.joint.pos + s.hand.axis - s.palm2.up * s.palm_width / 4 + s.palm2.axis / 2

        s.arm_state = 0
        if 25 <= s.y < 40:
            if -20 <= s.x < -5:
                s.indicator.pos = vector(-12.5, 32.5, -s.box_length[0]-5)
                s.arm_state = 4
            elif -5 <= s.x < 10:
                s.indicator.pos = vector(2.5, 32.5, -s.box_length[0]-5)
                s.arm_state = 5
            elif 10 <= s.x < 25:
                s.indicator.pos = vector(17.5, 32.5, -s.box_length[0]-5)
                s.arm_state = 6
        elif 40 <= s.y < 55:
            if -20 <= s.x < -5:
                s.indicator.pos = vector(-12.5, 47.5, -s.box_length[0]-5)
                s.arm_state = 1
            elif -5 <= s.x < 10:
                s.indicator.pos = vector(2.5, 47.5, -s.box_length[0]-5)
                s.arm_state = 2
            elif 10 <= s.x < 25:
                s.indicator.pos = vector(17.5, 47.5, -s.box_length[0]-5)
                s.arm_state = 3
        if s.target_state == s.arm_state:
            if s.target_t == 0:
                s.target_t = time.time()
            elif time.time() - s.target_t >= 1:
                while s.target_state == s.arm_state:
                    s.target_state = random.randint(1, 6)
                if s.target_state == 1:
                    s.target.pos = vector(-12.5, 47.5, -s.box_length[0] - 5)
                elif s.target_state == 2:
                    s.target.pos = vector(2.5, 47.5, -s.box_length[0] - 5)
                elif s.target_state == 3:
                    s.target.pos = vector(17.5, 47.5, -s.box_length[0] - 5)
                elif s.target_state == 4:
                    s.target.pos = vector(-12.5, 32.5, -s.box_length[0] - 5)
                elif s.target_state == 5:
                    s.target.pos = vector(2.5, 32.5, -s.box_length[0] - 5)
                elif s.target_state == 6:
                    s.target.pos = vector(17.5, 32.5, -s.box_length[0] - 5)
        else:
            s.target_t = 0





def calibrate(s, duration):
    print("Starting Calibration.....")
    start = time.time()
    data = []
    while time.time() - start < duration:
        update_serial(s, [1]*4*2)
        if s.fdat[7] < 0:
            raw = math.pi - math.atan(s.fdat[6] / s.fdat[7])
        elif s.fdat[6] > 0:
            raw = 2 * math.pi - math.atan(s.fdat[6] / s.fdat[7])
        else:
            raw = -math.atan(s.fdat[6] / s.fdat[7])
        data.append(raw)
    print("Done calibrating")
    s.angle = np.mean(data)


def write_csv(x, name):

    with open(name, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter='\t')
        for y in x:
            csv_writer.writerow(y)


def read_csv(x, name):

    with open(name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
        for line in csv_reader:
            x.append(line)


def collect_data(s, name, duration):

    val = []
    start = time.time()
    while time.time() - start < 3:
        pass
    print('Starting........')
    s.serial.reset_input_buffer()
    start = time.time()
    total = time.time()
    while time.time() - start < duration:
        if time.time() - total >= 10:
            s.label.text = str((time.time() - start) * 100 / duration)
            total = time.time()
        if s.state == 3:
           flag = 1
        else:
            flag = 0
        update_serial(s)
        if s.state == 0 and flag == 1:
            val.append([s.x, s.y, s.arm_state])

    print('Done, Writing CSV...')
    write_csv(val, name)
    print('Done, Written CSV with ', len(val), ' timestamps')


def beep():
    winsound.Beep(500, 1500)