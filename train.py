from environment import World
from ddpg import DDPG
import numpy as np
import os

################## hyper parameter#####################
WORLD_LENGTH = 100
WORLD_WIDTH = 100
UAV_NUM = 1
SENSOR_NUM = 5
VALID_NUM = 20
EPISUDE_NUM = 10001
# mission time (unit: s)
T = 1800


world = World(length=WORLD_LENGTH, width=WORLD_WIDTH)
agent = DDPG(a_dim=UAV_NUM*2, s_dim=(UAV_NUM*2+SENSOR_NUM))

explore_rate = 0.5


for i in range(EPISUDE_NUM):
    s = world.reset()
    # print(s)
    episude_reward = 0
    step_cnt = 0
    for t in range(int(T/world.uavs[0].update_time)):
        a = agent.choose_action(s)
        a = np.clip(np.random.normal(a, explore_rate), 0, 1)
        actions = list(a)
        s_, r, done = world.step_inside(actions)
        step_cnt += 1
        episude_reward += r
        agent.store_transition(s, a, r, s_, done)
        if agent.pointer > agent.memory_size:
            explore_rate *= .9998  # decay the action randomness
            agent.learn()
        if done:
            # print('step: ', step)
            break
        s = s_
    if i % 10 == 0:
        # print(s)
        print('episude:', i, '  episude_reward: ', episude_reward, '  step: ', step_cnt, '  explore_rate: ', explore_rate)

    if i % 100 == 0:
        f1 = open('valid.txt', 'w')
        f2 = open('final_energy.txt', 'w')
        valid_reward = 0
        average_receive = 0
        init_energy = 0
        f = 0
        for v in range(VALID_NUM):
            s = world.reset(validate=True)
            for sensor in world.sensors:
                init_energy += sensor.receive
            for t in range(int(T / world.uavs[0].update_time)):
                a = agent.choose_action(s)
                actions = list(a)
                s_, r, done = world.step_inside(actions)
                if v == 0:
                    trace = []
                    for u in world.uavs:
                        trace.append(' ')
                        trace.append(str(u.x))
                        trace.append(' ')
                        trace.append(str(u.y))
                        trace.append(' ')
                        trace.append(str(actions[1]))
                    trace.append('\n')
                    f1.writelines(trace)
                valid_reward += r
                if done:
                    break
            f_n = 0
            f_d = 0
     
            for sensor in world.sensors:
                if v == 0:
                    f2.writelines([str(sensor.receive)])
                average_receive += sensor.receive
                f_n += sensor.receive
                f_d += sensor.receive * sensor.receive
            if v == 0:
                f2.writelines([str(f_n * f_n / SENSOR_NUM / f_d)])
            f += f_n * f_n / SENSOR_NUM / f_d
        f1.writelines([str(valid_reward/VALID_NUM), '\n'])
        print('valid-reward: ', valid_reward/VALID_NUM, ' fair-index: ', f/VALID_NUM, 'init-energy: ', init_energy/VALID_NUM, ' average-receive: ', average_receive/VALID_NUM)
        f1.close()
        f2.close()
