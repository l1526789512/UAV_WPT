import tensorflow as tf
import numpy as np
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
gpuConfig = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
# gpuConfig.gpu_options.allow_growth = True
#####################  hyper parameters  ####################

LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 200


###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound=None):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 2), dtype=np.float32)
        self.pointer = 0
        self.memory_size = 20000
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
    
        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            self.q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            self.q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        # q_target = self.R + GAMMA * q_
        self.q_target = tf.placeholder("float", [None, 1])
        # in the feed_dic for the td_error, the self.a should change to actions in memory

        td_error = tf.losses.mean_squared_error(labels=self.q_target, predictions=self.q)
        with tf.name_scope('critic_adam_optimizer'):
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(self.q)  # maximize the q
        with tf.name_scope('actor_adam_optimizer'):
            self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        saver = tf.train.Saver()
        checkpoint_dir = "./model/"
        # 返回checkpoint文件中checkpoint的状态
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        # print(ckpt)
        if ckpt and ckpt.model_checkpoint_path:  # 如果存在以前保存的模型
            print('Restore the model from checkpoint %s' % ckpt.model_checkpoint_path)
            # Restores from checkpoint
            saver.restore(self.sess, ckpt.model_checkpoint_path)  # 加载模型
            # start_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        else:  # 如果不存在之前保存的模型
            self.sess.run(tf.global_variables_initializer())  # 变量初始化
            print('start training from new state')

        self.writer = tf.summary.FileWriter("logs/", self.sess.graph)

    def choose_action(self, s):
        s = np.array(s)
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        self.sess.run(self.atrain, {self.S: bs})
        self.train_critic(bt)

    def train_critic(self, batch):
        q_target = []
        bs = batch[:, :self.s_dim]
        ba = batch[:, self.s_dim: self.s_dim + self.a_dim]
        br = batch[:, -self.s_dim - 2: -self.s_dim - 1]
        bs_ = batch[:, -self.s_dim - 1:-1]
        done = batch[:, -1:]
        q_value_batch = self.q_.eval({self.S_: bs_})
        for i in range(BATCH_SIZE):
            if done[i]:
                q_target.append(br[i])
            else:
                q_target.append(br[i] + GAMMA * q_value_batch[i])
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_, self.q_target: q_target})

    def store_transition(self, s, a, r, s_, done):
        transition = np.hstack((s, a, [r], s_, done))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 200, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.sigmoid, name='a', trainable=trainable)
            # return tf.multiply(a, self.a_bound, name='scaled_a')
            return a

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 200
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            q = tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)
            return q

    def save_model(self):
        # define the saver
        saver = tf.train.Saver()
        saver.save(self.sess, "model/my-model")

    def load_model(self):
        saver = tf.train.import_meta_graph('model/my-model.meta')
        saver.restore(self.sess, tf.train.latest_checkpoint("model/"))
