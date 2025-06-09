
from interface import *

class SerialClass_rfd:
    def __init__(self, hz, width, window, rvel, delay, file_name, split):

        raw_data = []
        self.raw_data = []
        read_csv(raw_data, file_name)

        new_len = round(len(raw_data) * (1 - split))
        while len(raw_data) > new_len:
            raw_data.pop(0)

        for x in range(len(raw_data)):
            raw_data[x].pop(2)
            self.raw_data.append(raw_data[x])
        self.indicator_vis = 1
        self.arm_vis = 1
        self.robot_vis = 1
        self.arm_delay_vis = 1
        self.collect_data = 0
        self.delay = delay
        self.delay_data = []
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
        self.label = label(text = ' ', pos=vector(-20, 75, 20))

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

        robot_color_delay = vector(0, 1, 1)
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

        robot_color_now = vector(1, 1, 0)
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
        tolerance = 3
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
            random_num = 20
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
                    buff[x][y] = buff[x][y] - 59
                    due_y = due_y + 1
                box(pos=vector(base[0] + buff[x][y] * pixel_len, base[1] - due_y * pixel_len, -self.box_length[0] - 5), axis=vector(1, 0, 0), length=pixel_len, heigth=1,
                    width=pixel_len, up=vector(0, 0, 1))
                due_y = 0
            if x == 0 or x == 1:
                base[0] = base[0] + 15
            elif x == 2:
                base[1] = base[1] - 15
            else:
                base[0] = base[0] - 15
        vid_factor = 40 / 14.5
        vid_fram_thickness = 3
        self.vid_objects = []
        base[0] = 2.5 - 20 + vid_factor * pixel_len / 2
        base[1] = 85.4 - vid_fram_thickness + 20 - vid_factor * pixel_len / 2
        for x in range(pixel_num):
            for y in range(pixel_num):
                self.vid_objects.append(box(pos=vector(base[0] + x * vid_factor * pixel_len, base[1] - y * vid_factor * pixel_len, -self.box_length[0]-5), axis=vector(1, 0, 0), length=vid_factor * pixel_len, heigth=1, visible = False,
                        width=vid_factor * pixel_len, up=vector(0, 0, 1)))

        self.prev_r_state = 0

        self.indicator = box(pos=vector(0, 0, 100), length=15-thickness-tolerance, width=15-thickness-tolerance, height=1, up=vector(0, 0, 1), color=vector(0, 0.2, 1))
        self.target = vector(-12.5, 47.5, -self.box_length[0] - 5)
        self.targetl = box(pos=vector(-20, 47.5, -self.box_length[0]-4), axis=vector(0, 1, 0), length = 15,
                           heigth = 1, width = thickness, up = vector(0, 0, 1), color = color.red)
        self.targetr = box(pos=vector(-5, 47.5, -self.box_length[0] - 4), axis=vector(0, 1, 0), length=15,
                           heigth=1, width=thickness, up=vector(0, 0, 1), color = color.red)
        self.targetu = box(pos=vector(-12.5, 55, -self.box_length[0] - 4), axis=vector(1, 0, 0), length=15 + thickness,
                           heigth=1, width=thickness, up=vector(0, 0, 1), color = color.red)
        self.targetd = box(pos=vector(-12.5, 40, -self.box_length[0] - 4), axis=vector(1, 0, 0), length=15 + thickness,
                           heigth=1, width=thickness, up=vector(0, 0, 1), color = color.red)
        self.predictor = box(pos=vector(0, 0, 100), length=15-thickness, width=15-thickness, height=1, up=vector(0, 0, 1), color=vector(0.2, 0.9, 0.2))
        self.indicator_now = box(pos=vector(0, 0, 100), length=15-thickness-tolerance*2, width=15-thickness-tolerance*2, height=1, up=vector(0, 0, 1), color=color.yellow)
        self.target_t = 0
        self.target_state = 1
        self.arm_state = 0


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
        self.palm_now = sphere(pos=vec(self.box_length[1], 0, -self.box_length[0]+self.box_thick), radius=sphere_rad,
                           color=vector(0.75, 0.75, 0))
        self.state_now = 0

        self.time = time.time()
        self.data_time = time.time()
        self.hand.visible = False
        self.joint.visible = False
        self.arm.visible = False
        self.hand_now.visible = False
        self.joint_now.visible = False
        self.arm_now.visible = False
        self.rmse = []
        self.x_now = 0
        self.y_now = 0
        self.r_state_now = 0
        self.r_state_delay = 0




def update_serial_rfd(s, total_len):

    while time.time() - s.data_time >= 0.01 and len(s.raw_data) > 0:
        s.x_now = s.raw_data[0][0]
        s.y_now = s.raw_data[0][1]
        s.data_time = s.data_time + 0.01
        s.delay_data.append([s.x_now, s.y_now, s.data_time + s.delay])
        s.raw_data.pop(0)

    if len(s.delay_data) > 0:
        while time.time() >= s.delay_data[0][2]:
            s.x = s.delay_data[0][0]
            s.y = s.delay_data[0][1]
            s.ai.append([s.x, s.y, s.delay_data[0][2]])
            s.delay_data.pop(0)

    if time.time() - s.time >= s.rate:
        s.time = time.time()

        s.label.text = str(round(len(s.raw_data) * 100 / total_len, 2)) +'%'

        s.palm_now.pos.x = s.x_now
        s.palm_now.pos.y = s.y_now

        s.palm.pos.x = s.x
        s.palm.pos.y = s.y

        if 25 <= s.palm_now.pos.y < 40:
            if -20 <= s.palm_now.pos.x < -5:
                s.indicator_now.pos = vector(-12.5, 32.5, -s.box_length[0]-3)
                s.state_now = 4
            elif -5 <= s.palm_now.pos.x < 10:
                s.indicator_now.pos = vector(2.5, 32.5, -s.box_length[0]-3)
                s.state_now = 5
            elif 10 <= s.palm_now.pos.x < 25:
                s.indicator_now.pos = vector(17.5, 32.5, -s.box_length[0]-3)
                s.state_now = 6
        elif 40 <= s.palm_now.pos.y < 55:
            if -20 <= s.palm_now.pos.x < -5:
                s.indicator_now.pos = vector(-12.5, 47.5, -s.box_length[0]-3)
                s.state_now = 1
            elif -5 <= s.palm_now.pos.x < 10:
                s.indicator_now.pos = vector(2.5, 47.5, -s.box_length[0]-3)
                s.state_now = 2
            elif 10 <= s.palm_now.pos.x < 25:
                s.indicator_now.pos = vector(17.5, 47.5, -s.box_length[0]-3)
                s.state_now = 3


        s.arm_state = 0
        if 25 <= s.y < 40:
            if -20 <= s.x < -5:
                s.indicator.pos = vector(-12.5, 32.5, -s.box_length[0]-4)
                s.arm_state = 4
            elif -5 <= s.x < 10:
                s.indicator.pos = vector(2.5, 32.5, -s.box_length[0]-4)
                s.arm_state = 5
            elif 10 <= s.x < 25:
                s.indicator.pos = vector(17.5, 32.5, -s.box_length[0]-4)
                s.arm_state = 6
        elif 40 <= s.y < 55:
            if -20 <= s.x < -5:
                s.indicator.pos = vector(-12.5, 47.5, -s.box_length[0]-4)
                s.arm_state = 1
            elif -5 <= s.x < 10:
                s.indicator.pos = vector(2.5, 47.5, -s.box_length[0]-4)
                s.arm_state = 2
            elif 10 <= s.x < 25:
                s.indicator.pos = vector(17.5, 47.5, -s.box_length[0]-4)
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
        x_new = s.rpalm_vel[0]*dt
        y_new = s.rpalm_vel[1]*dt
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

        # if prev_rpalm_vel[0] != 0 and s.rpalm_vel[0] != 0 and prev_rpalm_vel[0] * s.rpalm_vel[0] < 0:
        #     s.rjoint3.pos.x = x_coor
        #     s.rpalm_vel[0] = 0
        #
        # if prev_rpalm_vel[1] != 0 and s.rpalm_vel[1] != 0 and prev_rpalm_vel[1] * s.rpalm_vel[1] < 0:
        #     s.rjoint3.pos.y = y_coor
        #     s.rpalm_vel[1] = 0


        # s.rjoint3.pos = s.rjoint3.pos + vector(s.rpalm_vel[0]*dt, s.rpalm_vel[1]*dt, 0)
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

        factor = sqrt((x_coor - s.rjoint3_now.pos.x) * (x_coor - s.rjoint3_now.pos.x) + (y_coor - s.rjoint3_now.pos.y) * (
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


        factor = sqrt((x_coor - s.rjoint3_delay.pos.x) * (x_coor - s.rjoint3_delay.pos.x) + (y_coor - s.rjoint3_delay.pos.y) * (
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

            if s.prev_r_state != 0:
                s.prev_r_state = s.prev_r_state - 1
                for x in range(len(s.pattern[s.prev_r_state])):
                    s.vid_objects[s.pattern[s.prev_r_state][x]].visible = False
            s.prev_r_state = s.r_state
            if s.r_state != 0:
                s.r_state = s.r_state - 1
                for x in range(len(s.pattern[s.r_state])):
                    s.vid_objects[s.pattern[s.r_state][x]].visible = True


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
        else:
            s.target_t = 0

        s.r_state_now = 0
        if 25 <= s.rjoint3_now.pos.y < 40:
            if -20 <= s.rjoint3_now.pos.x < -5:
                s.r_state_now = 4
            elif -5 <= s.rjoint3_now.pos.x < 10:
                s.r_state_now = 5
            elif 10 <= s.rjoint3_now.pos.x < 25:
                s.r_state_now = 6
        elif 40 <= s.rjoint3_now.pos.y < 55:
            if -20 <= s.rjoint3_now.pos.x < -5:
                s.r_state_now = 1
            elif -5 <= s.rjoint3_now.pos.x < 10:
                s.r_state_now = 2
            elif 10 <= s.rjoint3_now.pos.x < 25:
                s.r_state_now = 3

        s.r_state_delay = 0
        if 25 <= s.rjoint3_delay.pos.y < 40:
            if -20 <= s.rjoint3_delay.pos.x < -5:
                s.r_state_delay = 4
            elif -5 <= s.rjoint3_delay.pos.x < 10:
                s.r_state_delay = 5
            elif 10 <= s.rjoint3_delay.pos.x < 25:
                s.r_state_delay = 6
        elif 40 <= s.rjoint3_delay.pos.y < 55:
            if -20 <= s.rjoint3_delay.pos.x < -5:
                s.r_state_delay = 1
            elif -5 <= s.rjoint3_delay.pos.x < 10:
                s.r_state_delay = 2
            elif 10 <= s.rjoint3_delay.pos.x < 25:
                s.r_state_delay = 3

        #based on robot_now
        # dx = s.rjoint3_now.pos.x - s.rjoint3.pos.x
        # dy = s.rjoint3_now.pos.y - s.rjoint3.pos.y
        # dx_delay = s.rjoint3_now.pos.x - s.rjoint3_delay.pos.x
        # dy_delay = s.rjoint3_now.pos.y - s.rjoint3_delay.pos.y
        # if s.r_state_now == s.r_state:
        #     num1 = 1
        # else:
        #     num1 = 0
        # if s.r_state_delay == s.r_state_now:
        #     num2 = 1
        # else:
        #     num2 = 0
        # s.rmse.append([dx*dx + dy*dy, dx_delay*dx_delay + dy_delay*dy_delay, num1, num2])

        #based on arm_now
        dx = s.palm_now.pos.x - s.rjoint3.pos.x
        dy = s.palm_now.pos.y - s.rjoint3.pos.y
        dx_delay = s.palm_now.pos.x - s.rjoint3_delay.pos.x
        dy_delay = s.palm_now.pos.y - s.rjoint3_delay.pos.y
        if s.state_now == s.r_state:
            num1 = 1
        else:
            num1 = 0
        if s.r_state_delay == s.state_now:
            num2 = 1
        else:
            num2 = 0
        s.rmse.append([dx * dx + dy * dy, dx_delay * dx_delay + dy_delay * dy_delay, num1, num2])


