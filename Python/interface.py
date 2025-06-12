
from vpython import *
from prediction import load_model
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
    def __init__(self, port, hz, width, window, rvel, delay):
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
            self.active_keys = []
            self.indicator_vis = 1
            self.arm_vis = 1
            self.robot_vis = 1
            self.robot_now_vis = 1
            self.robot_delay_vis = 1
            self.arm_delay_vis = 1
            self.collect_data = 0
            self.delay = delay
            self.delay_data = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
            self.cnt = 0
            self.width = width
            self.window = window
            self.rate = 1/hz
            self.angle = 0
            self.x = 0
            self.y = 0
            self.ai = [[0, 0, 0]] * window
            self.ait = 0
            self.prediction = [[0, 100]]
            self.state = 0
            self.datbuff = b'\x00\x00\x00\x00'
            self.fdat: list[float] = [1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2]
            self.fdat_now: list[float] = [1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2]
            self.scene = canvas(width=745, height=710, background=color.black)
            # self.scene.userzoom = False
            # self.scene.userspin = False
            # self.scene.userpan = False
            self.scene.autoscale = False
            self.scene.camera.pos = vector(2.5, 50, 50)
            self.scene.forward = vector(0, 0, -1)
            self.scene.up = vector(0, 1, 0)
            sphere_rad = 4
            self.box_length = [26, 29]
            self.box_thick = 5
            self.palm_width = 8
            self.palm_length = 10
            sphere(pos=vec(0, 0, 0), radius=sphere_rad, color=vector(0, 0.65, 1))
            self.arm = box(pos=vector(0, 0, -self.box_length[0]/2), axis=vector(0, 0, -1), color=vector(0, 0.65, 1), size=vector(self.box_length[0], self.box_thick, self.box_thick))
            self.joint = sphere(pos=vec(0, 0, -self.box_length[0]), radius=sphere_rad, color=vector(0, 0.65, 1))
            self.hand = box(pos=vector(self.box_length[1]/2, 0, -self.box_length[0]), axis=vector(1, 0, 0), color=vector(0, 0.65, 1), size=vector(self.box_length[1], self.box_thick, self.box_thick))
            # self.palm1 = box(pos=vector(self.box_length[1]+(self.palm_length/2), 0, -self.box_length[0]+self.palm_width/4), axis=vector(1, 0, 0), color=color.white, size=vector(self.palm_length, self.box_thick, self.palm_width/2))
            # self.palm2 = box(pos=vector(self.box_length[1]+(self.palm_length/2), 0, -self.box_length[0]-self.palm_width/4), axis=vector(1, 0, 0), color=color.white, size=vector(self.palm_length, self.box_thick, self.palm_width/2))
            self.palm = sphere(pos=vec(self.box_length[1], 0, -self.box_length[0]), radius=sphere_rad, color=vector(0, 0.65, 1))

            self.label = label(text = ' ', pos=vector(-23.275000000002855, 75.1470000000007, 20), align='left')

            robot_color = vector(0.4, 0.6, 0.4)
            self.r2 = 30
            self.r3 = 20
            sphere(pos=vec(2.5, -20, -self.box_length[0]+self.box_thick), radius=sphere_rad, color=robot_color)
            self.rarm1 = box(pos=vector(0, 0, 100), color=robot_color,
                             size=vector(self.r2, self.box_thick, self.box_thick))
            self.rjoint1 = sphere(pos=vec(0, 0, 100), radius=sphere_rad, color=robot_color)
            self.rarm2 = box(pos=vector(0, 0, 100), color=robot_color,
                             size=vector(self.r2, self.box_thick, self.box_thick))
            self.rjoint2 = sphere(pos=vec(0, 0, 100), radius=sphere_rad, color=robot_color)
            self.rarm3 = box(pos=vector(0, 0, 100), color=robot_color,
                             size=vector(self.r3, self.box_thick, self.box_thick))
            self.rjoint3 = sphere(pos=vec(2.5, 17.5, -self.box_length[0]+self.box_thick), radius=sphere_rad, color=robot_color)
            self.rpalm_vel = [0, 0]
            self.rvel = rvel
            self.rtime = 0

            robot_color_delay = vector(1, 0.75, 0)
            self.rarm1_delay = box(pos=vector(0, 0, 100), color=robot_color_delay,
                                   size=vector(self.r2, self.box_thick, self.box_thick))
            self.rjoint1_delay = sphere(pos=vec(0, 0, 100), radius=sphere_rad, color=robot_color_delay)
            self.rarm2_delay = box(pos=vector(0, 0, 100), color=robot_color_delay,
                                   size=vector(self.r2, self.box_thick, self.box_thick))
            self.rjoint2_delay = sphere(pos=vec(0, 0, 100), radius=sphere_rad, color=robot_color_delay)
            self.rarm3_delay = box(pos=vector(0, 0, 100), color=robot_color_delay,
                                   size=vector(self.r3, self.box_thick, self.box_thick))
            self.rjoint3_delay = sphere(pos=vec(2.5, 17.5, -self.box_length[0] + self.box_thick), radius=sphere_rad,
                                        color=robot_color_delay)
            self.rpalm_vel_delay = [0, 0]
            self.rtime_delay = 0

            robot_color_now = vector(1, 0, 1)
            self.rarm1_now = box(pos=vector(0, 0, 100), color=robot_color_now,
                                 size=vector(self.r2, self.box_thick, self.box_thick))
            self.rjoint1_now = sphere(pos=vec(0, 0, 100), radius=sphere_rad, color=robot_color_now)
            self.rarm2_now = box(pos=vector(0, 0, 100), color=robot_color_now,
                                 size=vector(self.r2, self.box_thick, self.box_thick))
            self.rjoint2_now = sphere(pos=vec(0, 0, 100), radius=sphere_rad, color=robot_color_now)
            self.rarm3_now = box(pos=vector(0, 0, 100), color=robot_color_now,
                                 size=vector(self.r3, self.box_thick, self.box_thick))
            self.rjoint3_now = sphere(pos=vec(2.5, 17.5, -self.box_length[0] + self.box_thick), radius=sphere_rad,
                                      color=robot_color_now)
            self.rpalm_vel_now = [0, 0]
            self.rtime_now = 0

            # -20, 25, gap 15
            thickness = 0.5
            tolerance = 2
            box(pos=vector(-20, 40, -self.box_length[0]-5), axis=vector(0, 1, 0), length = 30, heigth = 1, width = thickness, up = vector(0, 0, 1))
            box(pos=vector(-5, 40, -self.box_length[0]-5), axis=vector(0, 1, 0), length=30, heigth=1, width=thickness, up=vector(0, 0, 1))
            box(pos=vector(10, 40, -self.box_length[0]-5), axis=vector(0, 1, 0), length=30, heigth=1, width=thickness, up=vector(0, 0, 1))
            box(pos=vector(25, 40, -self.box_length[0]-5), axis=vector(0, 1, 0), length=30, heigth=1, width=thickness, up=vector(0, 0, 1))
            box(pos=vector(2.5, 55, -self.box_length[0]-5), axis=vector(1, 0, 0), length=45 + thickness, heigth=1, width=thickness, up=vector(0, 0, 1))
            box(pos=vector(2.5, 40, -self.box_length[0]-5), axis=vector(1, 0, 0), length=45 + thickness, heigth=1, width=thickness, up=vector(0, 0, 1))
            box(pos=vector(2.5, 25, -self.box_length[0]-5), axis=vector(1, 0, 0), length=45 + thickness, heigth=1, width=thickness, up=vector(0, 0, 1))
            pixel_num = 60
            pixel_len = 14.5 / pixel_num
            self.pattern = []
            for x in range(6):
                self.pattern.append([])
                for y in range(pixel_num*pixel_num):
                    self.pattern[x].append(y)
                random_num = random.randint(0, 300)
                while len(self.pattern[x]) > random_num:
                    num_remove = random.randint(0, len(self.pattern[x])-1)
                    self.pattern[x].pop(num_remove)
            base = [-19.75, 54.75]
            base[0] = base[0] + pixel_len / 2
            base[1] = base[1] - pixel_len / 2
            due_y = 0
            buff = []
            for x in range(len(self.pattern)):
                buff.append([])
                for y in range(len(self.pattern[x])):
                    buff[x].append(self.pattern[x][y])
            for x in range(6):
                for y in range(len(buff[x])):
                    while buff[x][y] > 59:
                        buff[x][y] = buff[x][y] - 60
                        due_y = due_y + 1
                    box(pos=vector(base[0] + buff[x][y] * pixel_len, base[1] - due_y * pixel_len, -self.box_length[0] - 5), axis=vector(1, 0, 0), length=pixel_len, heigth=1,
                        width=pixel_len, up=vector(0, 0, 1))
                    due_y = 0
                if x == 0 or x == 1:
                    base[0] = base[0] + 15
                elif x == 2:
                    base[1] = base[1] - 15
                    base[0] = base[0] - 30
                else:
                    base[0] = base[0] + 15
            vid_factor = 40 / 14.5
            vid_fram_thickness = 3
            self.vid_objects = []
            base[0] = 2.5 - 20 + vid_factor * pixel_len / 2
            base[1] = 85.4 - vid_fram_thickness + 20 - vid_factor * pixel_len / 2
            for y in range(pixel_num):
                for x in range(pixel_num):
                    self.vid_objects.append(box(pos=vector(base[0] + x * vid_factor * pixel_len, base[1] - y * vid_factor * pixel_len, -self.box_length[0]-5), axis=vector(1, 0, 0), length=vid_factor * pixel_len, heigth=1, visible = False,
                            width=vid_factor * pixel_len, up=vector(0, 0, 1)))

            self.prev_r_state = 0

            self.indicator = box(pos=vector(0, 0, 100), length=15-thickness-tolerance, width=15-thickness-tolerance, height=1, up=vector(0, 0, 1), color=vector(0, 1, 1))
            self.target = vector(-12.5, 47.5, -self.box_length[0] - 5)
            self.targetl = box(pos=vector(-20, 47.5, -self.box_length[0]-4), axis=vector(0, 1, 0), length = 15,
                               heigth = 1, width = thickness, up = vector(0, 0, 1), color = color.red)
            self.targetr = box(pos=vector(-5, 47.5, -self.box_length[0] - 4), axis=vector(0, 1, 0), length=15,
                               heigth=1, width=thickness, up=vector(0, 0, 1), color = color.red)
            self.targetu = box(pos=vector(-12.5, 55, -self.box_length[0] - 4), axis=vector(1, 0, 0), length=15 + thickness,
                               heigth=1, width=thickness, up=vector(0, 0, 1), color = color.red)
            self.targetd = box(pos=vector(-12.5, 40, -self.box_length[0] - 4), axis=vector(1, 0, 0), length=15 + thickness,
                               heigth=1, width=thickness, up=vector(0, 0, 1), color = color.red)
            self.predictor = box(pos=vector(0, 0, 100), length=15-thickness, width=15-thickness, height=1, up=vector(0, 0, 1), color=color.white)
            self.indicator_now = box(pos=vector(0, 0, 100), length=15-thickness-tolerance*2, width=15-thickness-tolerance*2, height=1, up=vector(0, 0, 1), color=color.yellow)
            self.target_t = 0
            self.target_state = 1
            self.arm_state = 0

            self.indicator_robot = box(pos=vector(0, 0, 100), length=15 - thickness - tolerance * 3,
                                       width=15 - thickness - tolerance * 3, height=1, up=vector(0, 0, 1),
                                       color=vector(0.2, 0.9, 0.2))
            self.indicator_robot_delay = box(pos=vector(0, 0, 100), length=15 - thickness,
                                       width=15 - thickness - tolerance * 5, height=1, up=vector(0, 0, 1),
                                       color=vector(1, 0.75, 0))
            self.indicator_robot_now = box(pos=vector(0, 0, 100), length=15 - thickness - tolerance * 5,
                                       width=15 - thickness, height=1, up=vector(0, 0, 1),
                                       color=vector(1, 0, 1))

            box(pos=vector(2.5 - 20 - vid_fram_thickness / 2, 85.4 - vid_fram_thickness, -self.box_length[0]-5), axis=vector(0, 1, 0), length = 40,
                                  heigth = 5, width = vid_fram_thickness, up = vector(0, 0, 1), color = color.blue)
            box(pos=vector(2.5 + 20 + vid_fram_thickness / 2, 85.4 - vid_fram_thickness, -self.box_length[0] - 5), axis=vector(0, 1, 0), length=40,
                                  heigth=5, width=vid_fram_thickness, up=vector(0, 0, 1), color = color.blue)
            box(pos=vector(2.5, 85.4 - vid_fram_thickness - 20 - vid_fram_thickness / 2, -self.box_length[0] - 5), axis=vector(1, 0, 0), length=40 + vid_fram_thickness * 2,
                                  heigth=5, width=vid_fram_thickness, up=vector(0, 0, 1), color = color.blue)
            box(pos=vector(2.5, 85.4 - vid_fram_thickness + 20 + vid_fram_thickness / 2, -self.box_length[0] - 5), axis=vector(1, 0, 0), length=40 + vid_fram_thickness * 2,
                                  heigth=5, width=vid_fram_thickness, up=vector(0, 0, 1), color=color.blue)

            self.arm_now = box(pos=vector(0, 0, -self.box_length[0] / 2), axis=vector(0, 0, -1), color=vector(0.75, 0.75, 0),
                           size=vector(self.box_length[0], self.box_thick, self.box_thick))
            self.joint_now = sphere(pos=vec(0, 0, -self.box_length[0]), radius=sphere_rad, color=vector(0.75, 0.75, 0))
            self.hand_now = box(pos=vector(self.box_length[1] / 2, 0, -self.box_length[0]), axis=vector(1, 0, 0),
                            color=vector(0.75, 0.75, 0), size=vector(self.box_length[1], self.box_thick, self.box_thick))
            # self.palm1 = box(pos=vector(self.box_length[1]+(self.palm_length/2), 0, -self.box_length[0]+self.palm_width/4), axis=vector(1, 0, 0), color=color.white, size=vector(self.palm_length, self.box_thick, self.palm_width/2))
            # self.palm2 = box(pos=vector(self.box_length[1]+(self.palm_length/2), 0, -self.box_length[0]-self.palm_width/4), axis=vector(1, 0, 0), color=color.white, size=vector(self.palm_length, self.box_thick, self.palm_width/2))
            self.palm_now = sphere(pos=vec(self.box_length[1], 0, -self.box_length[0]), radius=sphere_rad,
                               color=vector(0.75, 0.75, 0))
            self.state_now = 0

            base = [-36.2885, 83.8045]
            box(pos=vector(base[0], base[1], 0), axis=vector(1, 0, 0),
                color=vector(0.75, 0.75, 0), size=vector(1.95, 1.95, 0.1))
            base[1] = base[1] - 1.95
            box(pos=vector(base[0], base[1], 0), axis=vector(1, 0, 0),
                color=vector(0, 0.65, 1), size=vector(1.95, 1.95, 0.1))
            base[1] = base[1] - 1.95
            box(pos=vector(base[0], base[1], 0), axis=vector(1, 0, 0),
                color=robot_color_now, size=vector(1.95, 1.95, 0.1))
            base[1] = base[1] - 1.95
            box(pos=vector(base[0], base[1], 0), axis=vector(1, 0, 0),
                color=robot_color_delay, size=vector(1.95, 1.95, 0.1))
            base[1] = base[1] - 1.95
            box(pos=vector(base[0], base[1], 0), axis=vector(1, 0, 0),
                color=robot_color, size=vector(1.95, 1.95, 0.1))

            self.serial.reset_input_buffer()
            self.time = time.time()


def stop():
    while True:
        pass

def angle_bias(s, data):
    dat = [0]*2
    angle = [math.sin(s.angle), math.cos(s.angle)]
    dat[0] = data[0] * angle[1] + data[1] * angle[0]
    dat[1] = data[1] * angle[1] - data[0] * angle[0]
    data[0] = dat[0]
    data[1] = dat[1]
    dat[0] = data[3] * angle[1] + data[4] * angle[0]
    dat[1] = data[4] * angle[1] - data[3] * angle[0]
    data[3] = dat[0]
    data[4] = dat[1]
    dat[0] = data[6] * angle[1] + data[7] * angle[0]
    dat[1] = data[7] * angle[1] - data[6] * angle[0]
    data[6] = dat[0]
    data[7] = dat[1]
    dat[0] = data[9] * angle[1] + data[10] * angle[0]
    dat[1] = data[10] * angle[1] - data[9] * angle[0]
    data[9] = dat[0]
    data[10] = dat[1]


def update_serial(s, param):

    if s.serial.inWaiting() > 1000:
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
                fbuff = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 0]
                for x in [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48]:
                    fbuff[int(x/4)], *_ = struct.unpack('f', dat[x:x+4])
                angle_bias(s, fbuff)
                s.fdat_now = []
                for x in range(len(fbuff)):
                    s.fdat_now.append(fbuff[x])
                fbuff[len(fbuff)-1] = time.time()
                s.delay_data.append(fbuff[:])

            s.state = 0

    while len(s.delay_data) > 0:
        if time.time() - s.delay_data[0][len(s.delay_data[0])-1] >= s.delay:
            for x in range(len(s.fdat)):
                s.fdat[x] = s.delay_data[0][x]
            s.x = s.fdat[0] * s.box_length[0] + s.fdat[6] * s.box_length[1]
            s.y = s.fdat[1] * s.box_length[0] + s.fdat[7] * s.box_length[1]
            s.ai.append([s.x, s.y, s.delay_data[0][len(s.delay_data[0])-1] + s.delay])
            s.delay_data.pop(0)
        else:
            break

    if time.time() - s.time >= s.rate:
        s.time = time.time()
        s.arm.axis = vector(s.fdat[0], s.fdat[1], s.fdat[2]) * s.box_length[0]
        s.arm.up = vector(s.fdat[3], s.fdat[4], s.fdat[5])
        s.arm.pos = s.arm.axis / 2
        s.joint.pos = s.arm.axis
        s.hand.axis = vector(s.fdat[6], s.fdat[7], s.fdat[8]) * s.box_length[1]
        s.hand.up = vector(s.fdat[9], s.fdat[10], s.fdat[11])
        s.hand.pos = s.joint.pos + s.hand.axis / 2
        s.palm.pos = s.joint.pos + s.hand.axis

        s.arm_now.axis = vector(s.fdat_now[0], s.fdat_now[1], s.fdat_now[2]) * s.box_length[0]
        s.arm_now.up = vector(s.fdat_now[3], s.fdat_now[4], s.fdat_now[5])
        s.arm_now.pos = s.arm_now.axis / 2
        s.joint_now.pos = s.arm_now.axis
        s.hand_now.axis = vector(s.fdat_now[6], s.fdat_now[7], s.fdat_now[8]) * s.box_length[1]
        s.hand_now.up = vector(s.fdat_now[9], s.fdat_now[10], s.fdat_now[11])
        s.hand_now.pos = s.joint_now.pos + s.hand_now.axis / 2
        s.palm_now.pos = s.joint_now.pos + s.hand_now.axis

        if 25 <= s.palm_now.pos.y < 40:
            if -20 <= s.palm_now.pos.x < -5:
                s.indicator_now.pos = vector(-12.5, 32.5, -s.box_length[0]-4.8)
                s.state_now = 4
            elif -5 <= s.palm_now.pos.x < 10:
                s.indicator_now.pos = vector(2.5, 32.5, -s.box_length[0]-4.8)
                s.state_now = 5
            elif 10 <= s.palm_now.pos.x < 25:
                s.indicator_now.pos = vector(17.5, 32.5, -s.box_length[0]-4.8)
                s.state_now = 6
        elif 40 <= s.palm_now.pos.y < 55:
            if -20 <= s.palm_now.pos.x < -5:
                s.indicator_now.pos = vector(-12.5, 47.5, -s.box_length[0]-4.8)
                s.state_now = 1
            elif -5 <= s.palm_now.pos.x < 10:
                s.indicator_now.pos = vector(2.5, 47.5, -s.box_length[0]-4.8)
                s.state_now = 2
            elif 10 <= s.palm_now.pos.x < 25:
                s.indicator_now.pos = vector(17.5, 47.5, -s.box_length[0]-4.8)
                s.state_now = 3


        s.arm_state = 0
        if 25 <= s.y < 40:
            if -20 <= s.x < -5:
                s.indicator.pos = vector(-12.5, 32.5, -s.box_length[0]-4.9)
                s.arm_state = 4
            elif -5 <= s.x < 10:
                s.indicator.pos = vector(2.5, 32.5, -s.box_length[0]-4.9)
                s.arm_state = 5
            elif 10 <= s.x < 25:
                s.indicator.pos = vector(17.5, 32.5, -s.box_length[0]-4.9)
                s.arm_state = 6
        elif 40 <= s.y < 55:
            if -20 <= s.x < -5:
                s.indicator.pos = vector(-12.5, 47.5, -s.box_length[0]-4.9)
                s.arm_state = 1
            elif -5 <= s.x < 10:
                s.indicator.pos = vector(2.5, 47.5, -s.box_length[0]-4.9)
                s.arm_state = 2
            elif 10 <= s.x < 25:
                s.indicator.pos = vector(17.5, 47.5, -s.box_length[0]-4.9)
                s.arm_state = 3

        num = round(s.prediction[0][0])
        if num == 4:
            x_coor = -12.5
            y_coor = 32.5
        elif num == 5:
            x_coor = 2.5
            y_coor = 32.5
        elif num == 6:
            x_coor = 17.5
            y_coor = 32.5
        elif num == 1:
            x_coor = -12.5
            y_coor = 47.5
        elif num == 2:
            x_coor = 2.5
            y_coor = 47.5
        elif num == 3:
            x_coor = 17.5
            y_coor = 47.5
        else:
            x_coor = s.rjoint3.pos.x
            y_coor = s.rjoint3.pos.y

        if 1 <= num <= 6:
            s.predictor.pos = vector(x_coor, y_coor, -s.box_length[0] - 5)
        else:
            s.predictor.pos = vector(0, 0, 100)

        prev_rpalm_vel = [s.rpalm_vel[0], s.rpalm_vel[1]]

        # if time.time() < s.prediction[0][1]:
        #     s.rpalm_vel[0] = (x_coor - s.rjoint3.pos.x) / (s.prediction[0][1] - time.time())
        #     s.rpalm_vel[1] = (y_coor - s.rjoint3.pos.y) / (s.prediction[0][1] - time.time())
        #
        #     factor = sqrt(s.rpalm_vel[0]*s.rpalm_vel[0] + s.rpalm_vel[1]*s.rpalm_vel[1])
        #     if  factor > s.rvel:
        #         factor = s.rvel / factor
        #         s.rpalm_vel[0] = s.rpalm_vel[0] * factor
        #         s.rpalm_vel[1] = s.rpalm_vel[1] * factor
        # else:
        #     factor = sqrt((x_coor - s.rjoint3.pos.x) * (x_coor - s.rjoint3.pos.x) + (y_coor - s.rjoint3.pos.y) * (
        #                 y_coor - s.rjoint3.pos.y))
        #     if factor != 0:
        #         s.rpalm_vel[0] = (x_coor - s.rjoint3.pos.x) * s.rvel / factor
        #         s.rpalm_vel[1] = (y_coor - s.rjoint3.pos.y) * s.rvel / factor
        #     else:
        #         s.rpalm_vel[0] = 0
        #         s.rpalm_vel[1] = 0

        factor = sqrt((x_coor - s.rjoint3.pos.x) * (x_coor - s.rjoint3.pos.x) + (y_coor - s.rjoint3.pos.y) * (
                    y_coor - s.rjoint3.pos.y))
        if factor != 0:
            s.rpalm_vel[0] = (x_coor - s.rjoint3.pos.x) * s.rvel / factor
            s.rpalm_vel[1] = (y_coor - s.rjoint3.pos.y) * s.rvel / factor
        else:
            s.rpalm_vel[0] = 0
            s.rpalm_vel[1] = 0

        dt = time.time() - s.rtime
        s.rtime = time.time()
        x_new = s.rpalm_vel[0] * dt
        y_new = s.rpalm_vel[1] * dt
        if s.rpalm_vel[0] < 0:
            if x_new > x_coor - s.rjoint3.pos.x:
                s.rjoint3.pos.x = s.rjoint3.pos.x + x_new
            else:
                s.rjoint3.pos.x = x_coor
        elif s.rpalm_vel[0] > 0:
            if x_new < x_coor - s.rjoint3.pos.x:
                s.rjoint3.pos.x = s.rjoint3.pos.x + x_new
            else:
                s.rjoint3.pos.x = x_coor
        if s.rpalm_vel[1] < 0:
            if y_new > y_coor - s.rjoint3.pos.y:
                s.rjoint3.pos.y = s.rjoint3.pos.y + y_new
            else:
                s.rjoint3.pos.y = y_coor
        elif s.rpalm_vel[1] > 0:
            if y_new < y_coor - s.rjoint3.pos.y:
                s.rjoint3.pos.y = s.rjoint3.pos.y + y_new
            else:
                s.rjoint3.pos.y = y_coor

        if s.rjoint3.pos.x > 25 or s.rjoint3.pos.x < -20:
            s.rjoint3.pos.x = 2.5
        if s.rjoint3.pos.y > 55 or s.rjoint3.pos.y < 17:
            s.rjoint3.pos.y = 17.5
        xdir = s.rjoint3.pos.x - 2.5
        ydir = s.rjoint3.pos.y + 20
        factor = sqrt(xdir*xdir + ydir*ydir)
        s.rarm3.axis = vector(s.r3*xdir/factor, s.r3*ydir/factor, 0)
        s.rarm3.pos = s.rjoint3.pos - (s.rarm3.axis / 2)
        s.rjoint2.pos = s.rarm3.pos - (s.rarm3.axis / 2)
        theta2 = asin((factor - s.r3) / (2*s.r2))
        rlen = sqrt(s.r2*s.r2 - s.r2*cos(theta2)*s.r2*cos(theta2))
        s.rarm2.axis = vector(rlen * xdir / factor, rlen * ydir / factor, -s.r2*cos(theta2))
        s.rarm2.pos = s.rjoint2.pos - (s.rarm2.axis / 2)
        s.rjoint1.pos = s.rarm2.pos - (s.rarm2.axis / 2)
        s.rarm1.axis = s.rjoint1.pos - vec(2.5, -20, -s.box_length[0]+s.box_thick)
        s.rarm1.pos = s.rjoint1.pos - (s.rarm1.axis / 2)

        num = s.state_now
        if num == 4:
            x_coor = -12.5
            y_coor = 32.5
        elif num == 5:
            x_coor = 2.5
            y_coor = 32.5
        elif num == 6:
            x_coor = 17.5
            y_coor = 32.5
        elif num == 1:
            x_coor = -12.5
            y_coor = 47.5
        elif num == 2:
            x_coor = 2.5
            y_coor = 47.5
        elif num == 3:
            x_coor = 17.5
            y_coor = 47.5
        else:
            x_coor = s.rjoint3_now.pos.x
            y_coor = s.rjoint3_now.pos.y

        # if 1 <= num <= 6:
        #     s.predictor.pos = vector(x_coor, y_coor, -s.box_length[0] - 5)
        # else:
        #     s.predictor.pos = vector(0, 0, 100)

        prev_rpalm_vel = [s.rpalm_vel_now[0], s.rpalm_vel_now[1]]

        factor = sqrt(
            (x_coor - s.rjoint3_now.pos.x) * (x_coor - s.rjoint3_now.pos.x) + (y_coor - s.rjoint3_now.pos.y) * (
                    y_coor - s.rjoint3_now.pos.y))
        if factor != 0:
            s.rpalm_vel_now[0] = (x_coor - s.rjoint3_now.pos.x) * s.rvel / factor
            s.rpalm_vel_now[1] = (y_coor - s.rjoint3_now.pos.y) * s.rvel / factor
        else:
            s.rpalm_vel_now[0] = 0
            s.rpalm_vel_now[1] = 0

        dt = time.time() - s.rtime_now
        s.rtime_now = time.time()
        x_new = s.rpalm_vel_now[0] * dt
        y_new = s.rpalm_vel_now[1] * dt
        if s.rpalm_vel_now[0] < 0:
            if x_new > x_coor - s.rjoint3_now.pos.x:
                s.rjoint3_now.pos.x = s.rjoint3_now.pos.x + x_new
            else:
                s.rjoint3_now.pos.x = x_coor
        elif s.rpalm_vel_now[0] > 0:
            if x_new < x_coor - s.rjoint3_now.pos.x:
                s.rjoint3_now.pos.x = s.rjoint3_now.pos.x + x_new
            else:
                s.rjoint3_now.pos.x = x_coor
        if s.rpalm_vel_now[1] < 0:
            if y_new > y_coor - s.rjoint3_now.pos.y:
                s.rjoint3_now.pos.y = s.rjoint3_now.pos.y + y_new
            else:
                s.rjoint3_now.pos.y = y_coor
        elif s.rpalm_vel_now[1] > 0:
            if y_new < y_coor - s.rjoint3_now.pos.y:
                s.rjoint3_now.pos.y = s.rjoint3_now.pos.y + y_new
            else:
                s.rjoint3_now.pos.y = y_coor

        if s.rjoint3_now.pos.x > 25 or s.rjoint3_now.pos.x < -20:
            s.rjoint3_now.pos.x = 2.5
        if s.rjoint3_now.pos.y > 55 or s.rjoint3_now.pos.y < 17:
            s.rjoint3_now.pos.y = 17.5
        xdir = s.rjoint3_now.pos.x - 2.5
        ydir = s.rjoint3_now.pos.y + 20
        factor = sqrt(xdir * xdir + ydir * ydir)
        s.rarm3_now.axis = vector(s.r3 * xdir / factor, s.r3 * ydir / factor, 0)
        s.rarm3_now.pos = s.rjoint3_now.pos - (s.rarm3_now.axis / 2)
        s.rjoint2_now.pos = s.rarm3_now.pos - (s.rarm3_now.axis / 2)
        theta2 = asin((factor - s.r3) / (2 * s.r2))
        rlen = sqrt(s.r2 * s.r2 - s.r2 * cos(theta2) * s.r2 * cos(theta2))
        s.rarm2_now.axis = vector(rlen * xdir / factor, rlen * ydir / factor, -s.r2 * cos(theta2))
        s.rarm2_now.pos = s.rjoint2_now.pos - (s.rarm2_now.axis / 2)
        s.rjoint1_now.pos = s.rarm2_now.pos - (s.rarm2_now.axis / 2)
        s.rarm1_now.axis = s.rjoint1_now.pos - vec(2.5, -20, -s.box_length[0] + s.box_thick)
        s.rarm1_now.pos = s.rjoint1_now.pos - (s.rarm1_now.axis / 2)

        num = s.arm_state
        if num == 4:
            x_coor = -12.5
            y_coor = 32.5
        elif num == 5:
            x_coor = 2.5
            y_coor = 32.5
        elif num == 6:
            x_coor = 17.5
            y_coor = 32.5
        elif num == 1:
            x_coor = -12.5
            y_coor = 47.5
        elif num == 2:
            x_coor = 2.5
            y_coor = 47.5
        elif num == 3:
            x_coor = 17.5
            y_coor = 47.5
        else:
            x_coor = s.rjoint3_delay.pos.x
            y_coor = s.rjoint3_delay.pos.y

        # if 1 <= num <= 6:
        #     s.predictor.pos = vector(x_coor, y_coor, -s.box_length[0] - 5)
        # else:
        #     s.predictor.pos = vector(0, 0, 100)

        prev_rpalm_vel = [s.rpalm_vel_delay[0], s.rpalm_vel_delay[1]]

        factor = sqrt(
            (x_coor - s.rjoint3_delay.pos.x) * (x_coor - s.rjoint3_delay.pos.x) + (y_coor - s.rjoint3_delay.pos.y) * (
                    y_coor - s.rjoint3_delay.pos.y))
        if factor != 0:
            s.rpalm_vel_delay[0] = (x_coor - s.rjoint3_delay.pos.x) * s.rvel / factor
            s.rpalm_vel_delay[1] = (y_coor - s.rjoint3_delay.pos.y) * s.rvel / factor
        else:
            s.rpalm_vel_delay[0] = 0
            s.rpalm_vel_delay[1] = 0

        dt = time.time() - s.rtime_delay
        s.rtime_delay = time.time()
        x_new = s.rpalm_vel_delay[0] * dt
        y_new = s.rpalm_vel_delay[1] * dt
        if s.rpalm_vel_delay[0] < 0:
            if x_new > x_coor - s.rjoint3_delay.pos.x:
                s.rjoint3_delay.pos.x = s.rjoint3_delay.pos.x + x_new
            else:
                s.rjoint3_delay.pos.x = x_coor
        elif s.rpalm_vel_delay[0] > 0:
            if x_new < x_coor - s.rjoint3_delay.pos.x:
                s.rjoint3_delay.pos.x = s.rjoint3_delay.pos.x + x_new
            else:
                s.rjoint3_delay.pos.x = x_coor
        if s.rpalm_vel_delay[1] < 0:
            if y_new > y_coor - s.rjoint3_delay.pos.y:
                s.rjoint3_delay.pos.y = s.rjoint3_delay.pos.y + y_new
            else:
                s.rjoint3_delay.pos.y = y_coor
        elif s.rpalm_vel_delay[1] > 0:
            if y_new < y_coor - s.rjoint3_delay.pos.y:
                s.rjoint3_delay.pos.y = s.rjoint3_delay.pos.y + y_new
            else:
                s.rjoint3_delay.pos.y = y_coor

        if s.rjoint3_delay.pos.x > 25 or s.rjoint3_delay.pos.x < -20:
            s.rjoint3_delay.pos.x = 2.5
        if s.rjoint3_delay.pos.y > 55 or s.rjoint3_delay.pos.y < 17:
            s.rjoint3_delay.pos.y = 17.5
        xdir = s.rjoint3_delay.pos.x - 2.5
        ydir = s.rjoint3_delay.pos.y + 20
        factor = sqrt(xdir * xdir + ydir * ydir)
        s.rarm3_delay.axis = vector(s.r3 * xdir / factor, s.r3 * ydir / factor, 0)
        s.rarm3_delay.pos = s.rjoint3_delay.pos - (s.rarm3_delay.axis / 2)
        s.rjoint2_delay.pos = s.rarm3_delay.pos - (s.rarm3_delay.axis / 2)
        theta2 = asin((factor - s.r3) / (2 * s.r2))
        rlen = sqrt(s.r2 * s.r2 - s.r2 * cos(theta2) * s.r2 * cos(theta2))
        s.rarm2_delay.axis = vector(rlen * xdir / factor, rlen * ydir / factor, -s.r2 * cos(theta2))
        s.rarm2_delay.pos = s.rjoint2_delay.pos - (s.rarm2_delay.axis / 2)
        s.rjoint1_delay.pos = s.rarm2_delay.pos - (s.rarm2_delay.axis / 2)
        s.rarm1_delay.axis = s.rjoint1_delay.pos - vec(2.5, -20, -s.box_length[0] + s.box_thick)
        s.rarm1_delay.pos = s.rjoint1_delay.pos - (s.rarm1_delay.axis / 2)

        s.r_state = 0
        if 25 <= s.rjoint3.pos.y < 40:
            if -20 <= s.rjoint3.pos.x < -5:
                s.r_state = 4
            elif -5 <= s.rjoint3.pos.x < 10:
                s.r_state = 5
            elif 10 <= s.rjoint3.pos.x < 25:
                s.r_state = 6
        elif 40 <= s.rjoint3.pos.y < 55:
            if -20 <= s.rjoint3.pos.x < -5:
                s.r_state = 1
            elif -5 <= s.rjoint3.pos.x < 10:
                s.r_state = 2
            elif 10 <= s.rjoint3.pos.x < 25:
                s.r_state = 3

        if s.r_state != s.prev_r_state:
            s.vid_objects
            if s.prev_r_state != 0:
                s.prev_r_state = s.prev_r_state - 1
                for x in range(len(s.pattern[s.prev_r_state])):
                    s.vid_objects[s.pattern[s.prev_r_state][x]].visible = False
            s.prev_r_state = s.r_state
            if s.r_state != 0:
                s.r_state = s.r_state - 1
                for x in range(len(s.pattern[s.r_state])):
                    s.vid_objects[s.pattern[s.r_state][x]].visible = True
                s.r_state = s.r_state + 1


        if s.collect_data:
            state_cehck = s.state_now
        else:
            state_cehck = s.r_state
        if s.target_state == state_cehck:
            if s.target_t == 0:
                s.target_t = time.time()
            elif time.time() - s.target_t >= 1:
                while s.target_state == state_cehck:
                    s.target_state = random.randint(1, 6)
                if s.target_state == 1:
                    s.target = vector(-12.5, 47.5, -s.box_length[0] - 4)
                elif s.target_state == 2:
                    s.target = vector(2.5, 47.5, -s.box_length[0] - 4)
                elif s.target_state == 3:
                    s.target = vector(17.5, 47.5, -s.box_length[0] - 4)
                elif s.target_state == 4:
                    s.target = vector(-12.5, 32.5, -s.box_length[0] - 4)
                elif s.target_state == 5:
                    s.target = vector(2.5, 32.5, -s.box_length[0] - 4)
                elif s.target_state == 6:
                    s.target = vector(17.5, 32.5, -s.box_length[0] - 4)

                s.targetu.pos = s.target + vector(0, 7.5, 0)
                s.targetd.pos = s.target + vector(0, -7.5, 0)
                s.targetl.pos = s.target + vector(-7.5, 0, 0)
                s.targetr.pos = s.target + vector(7.5, 0, 0)

        if 25 <= s.rjoint3.pos.y < 40:
            if -20 <= s.rjoint3.pos.x < -5:
                s.indicator_robot.pos = vector(-12.5, 32.5, -s.box_length[0]-4.7)
            elif -5 <= s.rjoint3.pos.x < 10:
                s.indicator_robot.pos = vector(2.5, 32.5, -s.box_length[0]-4.7)
            elif 10 <= s.rjoint3.pos.x < 25:
                s.indicator_robot.pos = vector(17.5, 32.5, -s.box_length[0]-4.7)
        elif 40 <= s.rjoint3.pos.y < 55:
            if -20 <= s.rjoint3.pos.x < -5:
                s.indicator_robot.pos = vector(-12.5, 47.5, -s.box_length[0]-4.7)
            elif -5 <= s.rjoint3.pos.x < 10:
                s.indicator_robot.pos = vector(2.5, 47.5, -s.box_length[0]-4.7)
            elif 10 <= s.rjoint3.pos.x < 25:
                s.indicator_robot.pos = vector(17.5, 47.5, -s.box_length[0]-4.7)

        if 25 <= s.rjoint3_delay.pos.y < 40:
            if -20 <= s.rjoint3_delay.pos.x < -5:
                s.indicator_robot_delay.pos = vector(-12.5, 32.5, -s.box_length[0]-4.6)
            elif -5 <= s.rjoint3_delay.pos.x < 10:
                s.indicator_robot_delay.pos = vector(2.5, 32.5, -s.box_length[0]-4.6)
            elif 10 <= s.rjoint3_delay.pos.x < 25:
                s.indicator_robot_delay.pos = vector(17.5, 32.5, -s.box_length[0]-4.6)
        elif 40 <= s.rjoint3_delay.pos.y < 55:
            if -20 <= s.rjoint3_delay.pos.x < -5:
                s.indicator_robot_delay.pos = vector(-12.5, 47.5, -s.box_length[0]-4.6)
            elif -5 <= s.rjoint3_delay.pos.x < 10:
                s.indicator_robot_delay.pos = vector(2.5, 47.5, -s.box_length[0]-4.6)
            elif 10 <= s.rjoint3_delay.pos.x < 25:
                s.indicator_robot_delay.pos = vector(17.5, 47.5, -s.box_length[0]-4.6)

        if 25 <= s.rjoint3_now.pos.y < 40:
            if -20 <= s.rjoint3_now.pos.x < -5:
                s.indicator_robot_now.pos = vector(-12.5, 32.5, -s.box_length[0]-4.5)
            elif -5 <= s.rjoint3_now.pos.x < 10:
                s.indicator_robot_now.pos = vector(2.5, 32.5, -s.box_length[0]-4.5)
            elif 10 <= s.rjoint3_now.pos.x < 25:
                s.indicator_robot_now.pos = vector(17.5, 32.5, -s.box_length[0]-4.5)
        elif 40 <= s.rjoint3_now.pos.y < 55:
            if -20 <= s.rjoint3_now.pos.x < -5:
                s.indicator_robot_now.pos = vector(-12.5, 47.5, -s.box_length[0]-4.5)
            elif -5 <= s.rjoint3_now.pos.x < 10:
                s.indicator_robot_now.pos = vector(2.5, 47.5, -s.box_length[0]-4.5)
            elif 10 <= s.rjoint3_now.pos.x < 25:
                s.indicator_robot_now.pos = vector(17.5, 47.5, -s.box_length[0]-4.5)

        else:
            s.target_t = 0


def calibrate(s, duration):
    print("Starting Calibration.....")
    temp_time = s.delay
    temp_rvel = s.rvel
    s.delay = 0
    s.rvel = 0
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
    s.delay = temp_time
    s.rvel = temp_rvel
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
    s.collect_data = 1
    s.delay = 0
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
        update_serial(s, [1] * 4 * 2)
        if s.state == 0 and flag == 1:
            val.append([s.x, s.y, s.arm_state])

    print('Done, Writing CSV...')
    write_csv(val, name)
    print('Done, Written CSV with ', len(val), ' timestamps')

def visibility_indicator(s, vis):
    s.indicator_vis = vis
    s.predictor.visible = vis
    if s.arm_delay_vis == 0:
        s.indicator.visible = 0
    else:
        s.indicator.visible = vis
    if s.arm_vis == 0:
        s.indicator_now.visible = 0
    else:
        s.indicator_now.visible = vis
    if s.robot_vis == 0:
        s.indicator_robot.visible = 0
    else:
        s.indicator_robot.visible = vis
    if s.robot_now_vis == 0:
        s.indicator_robot_now.visible = 0
    else:
        s.indicator_robot_now.visible = vis
    if s.robot_delay_vis == 0:
        s.indicator_robot_delay.visible = 0
    else:
        s.indicator_robot_delay.visible = vis


def visibility_arm(s, vis):
    s.arm_vis = vis
    s.arm_now.visible = vis
    s.joint_now.visible = vis
    s.hand_now.visible = vis
    s.palm_now.visible = vis
    if s.indicator_vis == 0:
        s.indicator_now.visible = 0
    else:
        s.indicator_now.visible = vis

def visibility_arm_delay(s, vis):
    s.arm_delay_vis = vis
    s.arm.visible = vis
    s.joint.visible = vis
    s.hand.visible = vis
    s.palm.visible = vis
    if s.indicator_vis == 0:
        s.indicator.visible = 0
    else:
        s.indicator.visible = vis


def visibility_robot(s, vis):
    s.robot_vis = vis
    s.rjoint3.visible = vis
    s.rarm3.visible = vis
    s.rjoint2.visible = vis
    s.rarm2.visible = vis
    s.rjoint1.visible = vis
    s.rarm1.visible = vis
    if s.indicator_vis == 0:
        s.indicator_robot.visible = 0
    else:
        s.indicator_robot.visible = vis

def visibility_robot_delay(s, vis):
    s.robot_delay_vis = vis
    s.rjoint3_delay.visible = vis
    s.rarm3_delay.visible = vis
    s.rjoint2_delay.visible = vis
    s.rarm2_delay.visible = vis
    s.rjoint1_delay.visible = vis
    s.rarm1_delay.visible = vis
    if s.indicator_vis == 0:
        s.indicator_robot_delay.visible = 0
    else:
        s.indicator_robot_delay.visible = vis

def visibility_robot_now(s, vis):
    s.robot_now_vis = vis
    s.rjoint3_now.visible = vis
    s.rarm3_now.visible = vis
    s.rjoint2_now.visible = vis
    s.rarm2_now.visible = vis
    s.rjoint1_now.visible = vis
    s.rarm1_now.visible = vis
    if s.indicator_vis == 0:
        s.indicator_robot_now.visible = 0
    else:
        s.indicator_robot_now.visible = vis

def update_text(s):

    s.label.text = str(int(s.delay*10)) + '00ms Delay\nHand Position Now (A)\nHand Position Delayed (S)\nRobot Now (D)\nRobot Delayed (F)\nRobot Predicted (G)\nIndicators (H)'

def beep():
    winsound.Beep(500, 1500)

def change_model(s, delay):
    s.delay = delay
    s.model = []
    s.model = load_model('Model - Data 5 - ' + str(int(delay*10)) + '00ms - empty window 3 - LSTM 128~8')
    buffer_param = []
    s.output_param = []
    read_csv(buffer_param, 'Param - Data 5 - ' + str(int(delay*10)) + '00ms - empty window 3 - LSTM 128~8')
    for y in range(len(buffer_param[0])):
        s.output_param.append(buffer_param[0][y])
    update_text(s)

def keyboard_input(s):
    keys = keysdown()
    # for x in range(len(keys)):
    #     if keys[x] == 'j':
    #         s.label.pos.x = s.label.pos.x - 0.0005
    #         print([s.label.pos.x, s.label.pos.y])
    #     elif keys[x] == 'l':
    #         s.label.pos.x = s.label.pos.x + 0.0005
    #         print([s.label.pos.x, s.label.pos.y])
    #     elif keys[x] == 'i':
    #         s.label.pos.y = s.label.pos.y + 0.0005
    #         print([s.label.pos.x, s.label.pos.y])
    #     elif keys[x] == 'k':
    #         s.label.pos.y = s.label.pos.y - 0.0005
    #         print([s.label.pos.x, s.label.pos.y])
    for x in range(len(s.active_keys)-1, -1, -1):
        same = 0
        for y in range(len(keys)):
            if keys[y] == s.active_keys[x]:
                same = 1
                break
        if same == 0:
            s.active_keys.pop(x)

    for x in range(len(keys)-1, -1, -1):
        for y in range(len(s.active_keys)):
            if keys[x] == s.active_keys[y]:
                keys.pop(x)
                break

    for x in range(len(keys)):
        s.active_keys.append(keys[x])

    for x in range(len(keys)):
        if keys[x] == 'a':
            if s.arm_vis:
                visibility_arm(s, 0)
            else:
                visibility_arm(s, 1)
        elif keys[x] == 's':
            if s.arm_delay_vis:
                visibility_arm_delay(s, 0)
            else:
                visibility_arm_delay(s, 1)
        elif keys[x] == 'd':
            if s.robot_now_vis:
                visibility_robot_now(s, 0)
            else:
                visibility_robot_now(s, 1)
        elif keys[x] == 'f':
            if s.robot_delay_vis:
                visibility_robot_delay(s, 0)
            else:
                visibility_robot_delay(s, 1)
        elif keys[x] == 'g':
            if s.robot_vis:
                visibility_robot(s, 0)
            else:
                visibility_robot(s, 1)
        elif keys[x] == 'h':
            if s.indicator_vis:
                visibility_indicator(s, 0)
            else:
                visibility_indicator(s, 1)
        elif keys[x] == '`':
            change_model(s, 0.3)
        elif keys[x] == '1':
            change_model(s, 0.5)
        elif keys[x] == '2':
            change_model(s, 0.8)
        elif keys[x] == '3':
            change_model(s, 1)
        elif keys[x] == '4':
            change_model(s, 1.3)
        elif keys[x] == '5':
            change_model(s, 1.5)
        elif keys[x] == '6':
            change_model(s, 1.8)
        elif keys[x] == '7':
            change_model(s, 2)
        elif keys[x] == '8':
            change_model(s, 2.3)
        elif keys[x] == '9':
            change_model(s, 2.5)
        elif keys[x] == '0':
            change_model(s, 2.8)
        elif keys[x] == '-':
            change_model(s, 3)
