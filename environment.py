from entities import Sensor, UAV
import random
import os
import simpy
import numpy as np

class World(object):

    def __init__(self, length=100, width=100, sensor_num=5, uav_num=1):
        self.length = length
        self.width = width
        self.sensor_num = sensor_num
        self.uav_num = uav_num
        self.sensors = []
        self.uavs = []
        self.max_x = 0
        self.min_x = length
        self.max_y = 0
        self.min_y = width
        self.env = None
        self.cnt_step = 0


    def reset(self, validate=False):
        self.env = simpy.Environment()
        self.set_sensors()
        if validate:
            self.set_uavs()
        else:
            self.random_set_uavs()
        for u in self.uavs:
            u.set_env(self.env)
        state = []
        for u in self.uavs:
            state.append(u.x / self.length)
            state.append(u.y / self.width)
        sum_receive = 0
        for s in self.sensors:
            sum_receive += s.receive
        for s in self.sensors:
            state.append(s.receive/sum_receive)
        return state

    def step(self, actions):
        state_ = []
        reward = 0
        done = False
        if len(actions) == self.uav_num*2:
            energy_consumption = 0
            for i, u in enumerate(self.uavs):
                energy_consumption += u.move(actions[i*2],actions[i*2+1])
                state_.append(u.x / self.length)
                state_.append(u.y / self.width)
                self.env.process(u.run(actions[i*2+1]))
            self.cnt_step += 1
            self.env.run(until=self.uavs[0].update_time * self.cnt_step)
            done = not self.uav_pos_legel()
            if not done:
                reward += self.get_reward(actions)/energy_consumption
            else:
                reward -= 1
            sum_receive = 0
            for s in self.sensors:
                sum_receive += s.receive
            for s in self.sensors:
                state_.append(s.receive / sum_receive)
            return state_, reward, done
        else:
            print('number of actions is wrong!')

    def step_inside(self, actions):
        state_ = []
        reward = 0
        done = False
        if len(actions) == self.uav_num*2:
            energy_consumption = 0
            fa = 0
            for i, u in enumerate(self.uavs):
                fa += u.move_inside(actions[i*2], actions[i*2+1], self.length, self.width)
                state_.append(u.x/self.length)
                state_.append(u.y/self.width)
                self.env.process(u.run(actions[i*2+1]))
            self.cnt_step += 1
            self.env.run(until=self.uavs[0].update_time * self.cnt_step)
            reward += self.get_reward(actions) + fa
            sum_receive = 0
            for s in self.sensors:
                sum_receive += s.receive
            for s in self.sensors:
                state_.append(s.receive / sum_receive)
            return state_, reward, done
        else:
            print('number of actions is wrong!')

    def get_reward(self, actions):
        reward = 0
        for i, u in enumerate(self.uavs):
            hover_time = u.update_time * (1 - actions[2*i+1])
            for s in self.sensors:
                u_s_reward = hover_time * 0.01 / (u.altitude*u.altitude + (u.x-s.x)*(u.x-s.x) + (u.y-s.y)*(u.y-s.y))
                s.receive += u_s_reward * 100
                reward += u_s_reward * 100
        f_n, f_d = 0, 0
        for s in self.sensors:
            f_n += s.receive
            f_d += s.receive*s.receive
        f = f_n*f_n/(len(self.sensors)*f_d)
        # print('r: ', reward, ' f: ', f)
        return reward * f

    def uav_pos_legel(self):
        legel = True
        for u in self.uavs:
            if u.x<self.min_x or u.x>self.max_x or u.y<self.min_y or u.y>self.max_y:
                legel = False
                break
        return legel

    def set_sensors(self):
        self.sensors = []
        if os.path.exists('sensors.txt'):
            f = open('sensors.txt', 'r')
            if f:
                sensor_loc = f.readline()
                sensor_loc = sensor_loc.split(' ')
                r = np.random.rand()*0.01
                self.sensors.append(Sensor(int(sensor_loc[0]), int(sensor_loc[1]), r))
                self.max_x = max(self.max_x, int(sensor_loc[0]))
                self.min_x = min(self.min_x, int(sensor_loc[0]))
                self.max_y = max(self.max_y, int(sensor_loc[1]))
                self.min_y = min(self.min_y, int(sensor_loc[1]))
                while sensor_loc:
                    sensor_loc = f.readline()
                    if sensor_loc:
                        sensor_loc = sensor_loc.split(' ')
                        r = np.random.rand()*0.01
                        self.sensors.append(Sensor(int(sensor_loc[0]), int(sensor_loc[1]), r))
                        self.max_x = max(self.max_x, int(sensor_loc[0]))
                        self.min_x = min(self.min_x, int(sensor_loc[0]))
                        self.max_y = max(self.max_y, int(sensor_loc[1]))
                        self.min_y = min(self.min_y, int(sensor_loc[1]))
                f.close()
        else:
            f = open('sensors.txt', 'w')
            for i in range(self.sensor_num):
                x = random.randint(0, self.length)
                y = random.randint(0, self.width)
                r = np.random.rand() * 0.01
                self.max_x = max(self.max_x, x)
                self.min_x = min(self.min_x, x)
                self.max_y = max(self.max_y, y)
                self.min_y = min(self.min_y, y)
                self.sensors.append(Sensor(x, y, r))
                f.writelines([str(x), ' ', str(y), '\n'])
            f.close()


    def random_set_uavs(self):
        self.uavs = []
        for i in range(self.uav_num):
            x = random.randint(0, self.length)
            y = random.randint(0, self.width)
            self.uavs.append(UAV(x, y, 5, 5, 20))

    def print_sensors(self):
        for sensor in self.sensors:
            print(sensor.x, ' ', sensor.y)

    def set_uavs(self):
        self.uavs = []
        if os.path.exists('uavs.txt'):
            f = open('uavs.txt', 'r')
            if f:
                uav_loc = f.readline()
                uav_loc = uav_loc.split(' ')
                self.uavs.append(UAV(int(uav_loc[0]), int(uav_loc[1]), 5, 5, 20))
                while uav_loc:
                    uav_loc = f.readline()
                    if uav_loc:
                        uav_loc = uav_loc.split(' ')
                        self.uavs.append(UAV(int(uav_loc[0]), int(uav_loc[1]), 5, 5, 20))
                f.close()
        else:
            f = open('uavs.txt', 'w')
            for i in range(self.uav_num):
                x = random.randint(0, self.length)
                y = random.randint(0, self.width)
                self.uavs.append(UAV(x, y, 20, 5, 20))
                f.writelines([str(x), ' ', str(y), '\n'])
            f.close()

    def print_uavs(self):
        for uav in self.uavs:
            print(uav.x, ' ', uav.y)



