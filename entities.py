import simpy
import random
import math
import numpy as np
import time


class Sensor(object):
    def __init__(self, x, y, remain):
        self.x = x
        self.y = y
        self.receive = remain


class UAV(object):

    def __init__(self, x, y, altitude, velocity, update_time):
        self.env = None
        self.x = x
        self.y = y
        self.altitude = altitude
        self.velocity = velocity
        self.update_time = update_time
        self.reward = 0
        self.step = 0
        #self.action = self.env.process(self.run())

    def set_env(self, env):
        self.env = env

    def run(self, normal_move_time):
        self.reward = 0
        while True:

            move_time = normal_move_time * self.update_time
            hover_time = self.update_time - move_time

            reward = 0
            yield self.env.timeout(move_time)
            # print(self.x, ' ', self.y, ' ', self.env.now)
            yield self.env.timeout(hover_time)

    # return energy consumption
    def move(self, angle, normal_move_time):
        dist = self.update_time * self.velocity * normal_move_time
        self.x += math.cos(2*math.pi*angle) * dist
        self.y += math.sin(2*math.pi*angle) * dist
        return dist/10 + 10

    def move_inside(self, angle, normal_move_time, length, width):
        dist = self.update_time * self.velocity * normal_move_time
        self.x += math.cos(2*math.pi*angle) * dist
        fa = 0
        if self.x < 0:
            fa += self.x / dist
            self.x = 0
        elif self.x>length:
            fa += (length - self.x) / dist
            self.x = length
        self.y += math.sin(2*math.pi*angle) * dist
        if self.y < 0:
            fa += self.y / dist
            self.y = 0
        elif self.y > width:
            fa += (width - self.y) / dist
            self.y = width
        return fa


