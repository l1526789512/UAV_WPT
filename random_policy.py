from environment import World
import numpy as np

WORLD_LENGTH = 100
WORLD_WIDTH = 100
EPISUDE_NUM = 100
SENSOR_NUM = 5
T = 1800

world = World(length=WORLD_LENGTH, width=WORLD_WIDTH)
valid_reward = 0
average_fair = 0
average_receive = 0
for i in range(EPISUDE_NUM):
    s = world.reset(validate=True)
    valid_step_cnt = 0
    for t in range(int(T / world.uavs[0].update_time)):
        a = np.random.rand(len(world.uavs)*2)
        actions = list(a)
        s_, r, done = world.step_inside(actions)
        valid_step_cnt += 1
        valid_reward += r
    f_n = 0
    f_d = 0
    for sensor in world.sensors:
        f_n += sensor.receive
        f_d += sensor.receive * sensor.receive
    f = f_n * f_n / SENSOR_NUM / f_d
    average_fair += f
    average_receive += f_n
    print('fair: ', f)
print('average-valid-reward: ', valid_reward/EPISUDE_NUM, ' fair: ', average_fair/EPISUDE_NUM, 'average-receive: ', average_receive/EPISUDE_NUM)
