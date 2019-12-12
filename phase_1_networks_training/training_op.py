from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

import os

import numpy as np
import datetime
import tensorflow as tf

import keras.backend as K
import keras.models as models

import utils
from networks_builder import NetworksBuilder

from .Interceptor_V2_training import Game, \
    ACTION_LEFT, ACTION_NONE, ACTION_RIGHT, ACTION_FIRE, \
    COLLISION_TYPE_GROUND, COLLISION_TYPE_CITY, COLLISION_TYPE_ROCKET

from threading import Lock


class Memory(object):

    def __init__(self):
        self.n_actions = 4
        self.action_space = [i for i in range(self.n_actions)]

        self.hit_counter = 0  # miss_counter = batch - hit_counter

        if train_type == TRAIN_FIRE_ANGLE_ASSESSMENT:
            self.batch_s_fire_angle_hit = []  # r_ang
            self.labels_reg_fire_angle_hit = []

        elif train_type == TRAIN_COLLISION_ASSESSMENTS:
            self.batch_s_collision = []  # r_c2
            self.labels_clss_collision = []
            self.labels_reg_time_steps_collision = []

            self.city_hit_counter = 0
            self.ground_hit_counter = 0

        elif train_type == TRAIN_INTERCEPTION_ASSESSMENTS or train_type == TRAIN_PRE_FIRE_INTERCEPTION_ASSESSMENTS:
            self.batch_s_interception_hit = []  # r_i
            self.labels_clss_interception_hit = []
            self.labels_reg_time_steps_interception = []

            self.batch_s_interception_miss = []
            self.labels_clss_interception_miss = []

    def end_of_episode_process_get_hit_miss(self, env):
        # in the end of each episode, update rewards according to hit, and add final data to memory.
        rocket = env.total_rockets[0]

        is_hit = rocket.collision_classification == COLLISION_TYPE_ROCKET
        if is_hit:  # single hit
            self.hit_counter += 1

        if train_type == TRAIN_FIRE_ANGLE_ASSESSMENT:
            if is_hit:  # single hit
                self.save_rocket_and_ang_data(rocket, env)
        elif train_type == TRAIN_INTERCEPTION_ASSESSMENTS:
            self.store_interception_data_episode_end(rocket, env)
        elif train_type == TRAIN_PRE_FIRE_INTERCEPTION_ASSESSMENTS:
            self.store_pre_fire_interception_data_episode_end(rocket, env)

        air_time = len(env.total_interceptors[rocket.colliding_rocket_index].data) if is_hit else 0
        return is_hit, len(env.total_interceptors), air_time

    def save_rocket_and_ang_data(self, rocket, env):
        # collision_index = rocket.collision_time_step - rocket.appearance_time_step            # impact time

        interceptor = env.total_interceptors[rocket.colliding_rocket_index]  # the intercepting interceptor
        # fire_action_index = interceptor.appearance_time_step - 1 - rocket.appearance_time_step  # firing time - one step before the interceptor appeared

        # add friendly turret angle and rocket data when intercepting interceptor was fired:
        self.batch_s_fire_angle_hit.append(interceptor.rocket_data_when_fired)
        self.labels_reg_fire_angle_hit.append(interceptor.fire_ang)

    def store_interception_data_episode_end(self, rocket, env):
        if rocket.collision_classification == COLLISION_TYPE_ROCKET:  # single hit - save only intercepting interceptor data
            interceptor = env.total_interceptors[rocket.colliding_rocket_index]  # the intercepting interceptor
            if interceptor.data:
                steps_from_interceptor_appearance = max(0, rocket.appearance_time_step - interceptor.appearance_time_step - 1)
                for i_data in interceptor.data[steps_from_interceptor_appearance:]:
                    r_data = rocket.data[interceptor.appearance_time_step + 1 - rocket.appearance_time_step + steps_from_interceptor_appearance]
                    self.batch_s_interception_hit.append(np.array([*r_data[:4], *i_data]))
                    a_indices_one_hot = np.zeros(2, dtype=np.ubyte)
                    a_indices_one_hot[1] = 1  # hit [0,1]
                    self.labels_clss_interception_hit.append(a_indices_one_hot)
                    self.labels_reg_time_steps_interception.append(interceptor.collision_time_step - (interceptor.appearance_time_step + 1) - steps_from_interceptor_appearance)
                    steps_from_interceptor_appearance += 1

        else:  # all misses - save all missed interceptors data
            interceptors = env.total_interceptors
            for interceptor in interceptors:
                steps_from_interceptor_appearance = max(0, rocket.appearance_time_step - interceptor.appearance_time_step - 1)
                for i_data in interceptor.data[steps_from_interceptor_appearance:]:
                    r_data = rocket.data[interceptor.appearance_time_step + 1 - rocket.appearance_time_step + steps_from_interceptor_appearance]
                    self.batch_s_interception_miss.append(np.array([*r_data[:4], *i_data]))
                    a_indices_one_hot = np.zeros(2, dtype=np.ubyte)
                    a_indices_one_hot[0] = 1  # miss [1,0]
                    self.labels_clss_interception_miss.append(a_indices_one_hot)
                    steps_from_interceptor_appearance += 1

    def store_pre_fire_interception_data_episode_end(self, rocket, env):
        if rocket.collision_classification == COLLISION_TYPE_ROCKET:  # single hit - save only intercepting interceptor data
            interceptor = env.total_interceptors[rocket.colliding_rocket_index]  # the intercepting interceptor
            r_data = rocket.data[interceptor.appearance_time_step - rocket.appearance_time_step]
            self.batch_s_interception_hit.append(np.array([*r_data, interceptor.fire_ang]))
            a_indices_one_hot = np.zeros(2, dtype=np.ubyte)
            a_indices_one_hot[1] = 1  # hit [0,1]
            self.labels_clss_interception_hit.append(a_indices_one_hot)
            self.labels_reg_time_steps_interception.append(interceptor.collision_time_step - interceptor.appearance_time_step)

        else:  # all misses - save all missed interceptors data
            interceptors = env.total_interceptors
            for interceptor in interceptors:
                r_data = rocket.data[interceptor.appearance_time_step - rocket.appearance_time_step]
                self.batch_s_interception_miss.append(np.array([*r_data, interceptor.fire_ang]))
                a_indices_one_hot = np.zeros(2, dtype=np.ubyte)
                a_indices_one_hot[0] = 1  # miss [1,0]
                self.labels_clss_interception_miss.append(a_indices_one_hot)

    def store_transitions_episode_end_collision_assessment(self, env):
        for rocket in env.total_rockets:
            if rocket.collision_classification == -1:
                break

            if rocket.collision_classification == COLLISION_TYPE_CITY:
                self.city_hit_counter += 1
            elif rocket.collision_classification == COLLISION_TYPE_GROUND:
                self.ground_hit_counter += 1

            counter = 1
            for r_data in rocket.data:
                self.batch_s_collision.append(np.array([*r_data[:4], *env.get_cities_data()]))
                collision_clss_one_hot = np.zeros(2, dtype=np.ubyte)
                collision_clss_one_hot[rocket.collision_classification] = 1
                self.labels_clss_collision.append(collision_clss_one_hot)
                self.labels_reg_time_steps_collision.append(len(rocket.data) - counter)
                counter += 1

    def reset_memory(self):
        self.hit_counter = 0

        if train_type == TRAIN_FIRE_ANGLE_ASSESSMENT:
            self.batch_s_fire_angle_hit = []
            self.labels_reg_fire_angle_hit = []

        elif train_type == TRAIN_COLLISION_ASSESSMENTS:
            self.batch_s_collision = []
            self.labels_clss_collision = []
            self.labels_reg_time_steps_collision = []

            self.city_hit_counter = 0
            self.ground_hit_counter = 0

        elif train_type == TRAIN_INTERCEPTION_ASSESSMENTS or train_type == TRAIN_PRE_FIRE_INTERCEPTION_ASSESSMENTS:
            self.batch_s_interception_hit = []
            self.labels_clss_interception_hit = []
            self.labels_reg_time_steps_interception = []

            self.batch_s_interception_miss = []
            self.labels_clss_interception_miss = []


class Agent:

    def __init__(self):
        self.n_actions = 4
        self.action_space = [i for i in range(self.n_actions)]

        # sub_dir = utils.General.get_file_name(None, self, self.BETA) + '/'
        self.chkpt_dir_base = base_dir + sub_dir
        self.counter = 1
        self.update_dirs()

        # initialize_trainability_variables:
        self.optimizer = utils.Optimizers.keras_get_optimizer(optimizer_type, ALPHA, epsilon=1e-3)  # RMSprop(lr, epsilon=0.1, rho=0.99)
        self.memory = Memory()

        if train_type == TRAIN_FIRE_ANGLE_ASSESSMENT:
            self.fire_ang_assessment = NetworksBuilder.build_fire_ang_assessment_network(optimizer_type, ALPHA)
        elif train_type == TRAIN_COLLISION_ASSESSMENTS:
            self.collision_assessment, self.collision_time_steps_assessment = NetworksBuilder.build_collision_assessment_networks(optimizer_type, ALPHA)
        elif train_type == TRAIN_INTERCEPTION_ASSESSMENTS:
            self.interception_assessment, self.interception_time_steps_assessment = NetworksBuilder.build_interception_assessment_networks(optimizer_type, ALPHA)
        elif train_type == TRAIN_PRE_FIRE_INTERCEPTION_ASSESSMENTS:
            self.pre_fire_interception_assessment, self.pre_fire_interception_time_steps_assessment = NetworksBuilder.build_pre_fire_interception_assessment_networks(optimizer_type, ALPHA)

        if load_checkpoint:
            try:
                self.load_models() if load_models else self.load_weights()
            except (ValueError, tf.OpError, OSError):
                print('...No models to load...')

        if multi_threading:
            # needed for the multi-threading
            global tf_session, tf_graph
            tf_session = K.get_session()
            tf_graph = tf.get_default_graph()

    def update_dirs(self):
        self.sub_dir = str(self.counter) + '/'
        self.chkpt_dir = self.chkpt_dir_base + self.sub_dir

        utils.General.make_sure_dir_exists(self.chkpt_dir)

        if train_type == TRAIN_FIRE_ANGLE_ASSESSMENT:
            self.h5_file_fire_ang = os.path.join(self.chkpt_dir, 'ac_keras_fire_angle.h5')
            self.h5_file_fire_ang_weights = os.path.join(self.chkpt_dir, 'ac_keras_fire_angle_weights.h5')
        elif train_type == TRAIN_COLLISION_ASSESSMENTS:
            self.h5_file_collision = os.path.join(self.chkpt_dir, 'ac_keras_collision.h5')
            self.h5_file_collision_weights = os.path.join(self.chkpt_dir, 'ac_keras_collision_weights.h5')
            self.h5_file_collision_time_steps = os.path.join(self.chkpt_dir, 'ac_keras_collision_time_steps.h5')
            self.h5_file_collision_time_steps_weights = os.path.join(self.chkpt_dir, 'ac_keras_collision_time_steps_weights.h5')
        elif train_type == TRAIN_INTERCEPTION_ASSESSMENTS:
            self.h5_file_interception = os.path.join(self.chkpt_dir, 'ac_keras_interception.h5')
            self.h5_file_interception_weights = os.path.join(self.chkpt_dir, 'ac_keras_interception_weights.h5')
            self.h5_file_interception_time_steps = os.path.join(self.chkpt_dir, 'ac_keras_interception_time_steps.h5')
            self.h5_file_interception_time_steps_weights = os.path.join(self.chkpt_dir, 'ac_keras_interception_time_steps_weights.h5')
        elif train_type == TRAIN_PRE_FIRE_INTERCEPTION_ASSESSMENTS:
            self.h5_file_pre_fire_interception = os.path.join(self.chkpt_dir, 'ac_keras_pre_fire_interception.h5')
            self.h5_file_pre_fire_interception_weights = os.path.join(self.chkpt_dir, 'ac_keras_pre_fire_interception_weights.h5')
            self.h5_file_pre_fire_interception_time_steps = os.path.join(self.chkpt_dir, 'ac_keras_pre_fire_interception_time_steps.h5')
            self.h5_file_pre_fire_interception_time_steps_weights = os.path.join(self.chkpt_dir, 'ac_keras_pre_fire_interception_time_steps_weights.h5')

    ###################################

    def learn(self):
        self.learn_assessment()
        self.memory.reset_memory()

    def learn_assessment(self):
        if train_type == TRAIN_FIRE_ANGLE_ASSESSMENT:
            if self.memory.batch_s_fire_angle_hit:
                batch_s_fire_angle_hit = np.array(self.memory.batch_s_fire_angle_hit, dtype=np.float32)
                labels_reg_fire_angle_hit = np.array(self.memory.labels_reg_fire_angle_hit, dtype=np.float32)
                batch_s_fire_angle_hit[:, 0] = batch_s_fire_angle_hit[:, 0] + 2000  # x axis origin adjustment
                batch_s_fire_angle_hit[:, :2] = batch_s_fire_angle_hit[:, :2] / 7000
                batch_s_fire_angle_hit[:, 2:4] = batch_s_fire_angle_hit[:, 2:4] / 180
                batch_s_fire_angle_hit[:, 4] = batch_s_fire_angle_hit[:, 4] / 7000
                batch_s_fire_angle_hit[:, 5] = batch_s_fire_angle_hit[:, 5] / 90
                self.fire_ang_assessment.fit(batch_s_fire_angle_hit, labels_reg_fire_angle_hit, verbose=0)  # verbose=2

        elif train_type == TRAIN_COLLISION_ASSESSMENTS:
            if self.memory.batch_s_collision:
                batch_s_collision = np.array(self.memory.batch_s_collision, dtype=np.float32)
                labels_clss_collision = np.array(self.memory.labels_clss_collision, dtype=np.ubyte)
                labels_reg_time_steps_collision = np.array(self.memory.labels_reg_time_steps_collision, dtype=np.uint16)
                batch_s_collision[:, 0] = batch_s_collision[:, 0] / 4950
                batch_s_collision[:, 1] = batch_s_collision[:, 1] / 5965
                batch_s_collision[:, 2:4] = batch_s_collision[:, 2:4] / 180
                batch_s_collision[:, 4:8] = batch_s_collision[:, 4:8] / 4950
                self.collision_assessment.fit(batch_s_collision, labels_clss_collision, verbose=0)  # verbose=2
                self.collision_time_steps_assessment.fit(batch_s_collision, labels_reg_time_steps_collision, verbose=0)  # verbose=2

        elif train_type == TRAIN_INTERCEPTION_ASSESSMENTS:
            if self.memory.batch_s_interception_hit or self.memory.batch_s_interception_miss:  # making sure interceptors were fired
                if self.memory.batch_s_interception_hit and self.memory.batch_s_interception_miss:    # hits & misses
                    batch_s_all = np.row_stack((self.memory.batch_s_interception_hit, self.memory.batch_s_interception_miss))
                    labels_clss_all = np.row_stack((self.memory.labels_clss_interception_hit, self.memory.labels_clss_interception_miss))
                elif self.memory.batch_s_interception_hit:                               # only hits
                    batch_s_all = np.array(self.memory.batch_s_interception_hit, dtype=np.float32)
                    labels_clss_all = np.array(self.memory.labels_clss_interception_hit, dtype=np.ubyte)
                else:                                                           # only misses
                    batch_s_all = np.array(self.memory.batch_s_interception_miss, dtype=np.float32)
                    labels_clss_all = np.array(self.memory.labels_clss_interception_miss, dtype=np.ubyte)
                # y axis origin adjustment formula: + 180 - (6500 + 180) / 2 = 180 - 3340 = - 3160
                batch_s_all[:, 1] = batch_s_all[:, 1] - 3160  # y axis origin adjustment
                batch_s_all[:, 5] = batch_s_all[:, 5] - 3160  # y axis origin adjustment
                batch_s_all[:, 0] = batch_s_all[:, 0] / 5180
                batch_s_all[:, 1] = batch_s_all[:, 1] / 3340
                batch_s_all[:, 2:4] = batch_s_all[:, 2:4] / 180
                batch_s_all[:, 4] = batch_s_all[:, 4] / 5180
                batch_s_all[:, 5] = batch_s_all[:, 5] / 3340
                batch_s_all[:, 6:8] = batch_s_all[:, 6:8] / 180
                self.interception_assessment.fit(batch_s_all, labels_clss_all, verbose=0)  # verbose=2

            if self.memory.batch_s_interception_hit:
                batch_s_interception_hit = np.array(self.memory.batch_s_interception_hit, dtype=np.float32)
                labels_reg_time_steps_interception = np.array(self.memory.labels_reg_time_steps_interception, dtype=np.uint16)
                # y axis origin adjustment formula: + 180 - (6500 + 180) / 2 = 180 - 3340 = - 3160
                batch_s_interception_hit[:, 1] = batch_s_interception_hit[:, 1] - 3160  # y axis origin adjustment
                batch_s_interception_hit[:, 5] = batch_s_interception_hit[:, 5] - 3160  # y axis origin adjustment
                batch_s_interception_hit[:, 0] = batch_s_interception_hit[:, 0] / 5180
                batch_s_interception_hit[:, 1] = batch_s_interception_hit[:, 1] / 3340
                batch_s_interception_hit[:, 2:4] = batch_s_interception_hit[:, 2:4] / 180
                batch_s_interception_hit[:, 4] = batch_s_interception_hit[:, 4] / 5180
                batch_s_interception_hit[:, 5] = batch_s_interception_hit[:, 5] / 3340
                batch_s_interception_hit[:, 6:8] = batch_s_interception_hit[:, 6:8] / 180
                self.interception_time_steps_assessment.fit(batch_s_interception_hit, labels_reg_time_steps_interception, verbose=0)  # verbose=2

        elif train_type == TRAIN_PRE_FIRE_INTERCEPTION_ASSESSMENTS:
            if self.memory.batch_s_interception_hit or self.memory.batch_s_interception_miss:  # making sure interceptors were fired
                if self.memory.batch_s_interception_hit and self.memory.batch_s_interception_miss:    # hits & misses
                    batch_s_all = np.row_stack((self.memory.batch_s_interception_hit, self.memory.batch_s_interception_miss))
                    labels_clss_all = np.row_stack((self.memory.labels_clss_interception_hit, self.memory.labels_clss_interception_miss))
                elif self.memory.batch_s_interception_hit:                               # only hits
                    batch_s_all = np.array(self.memory.batch_s_interception_hit, dtype=np.float32)
                    labels_clss_all = np.array(self.memory.labels_clss_interception_hit, dtype=np.ubyte)
                else:                                                           # only misses
                    batch_s_all = np.array(self.memory.batch_s_interception_miss, dtype=np.float32)
                    labels_clss_all = np.array(self.memory.labels_clss_interception_miss, dtype=np.ubyte)
                batch_s_all[:, 0] = batch_s_all[:, 0] + 2000  # x axis origin adjustment
                batch_s_all[:, :2] = batch_s_all[:, :2] / 7000
                batch_s_all[:, 2:4] = batch_s_all[:, 2:4] / 180
                batch_s_all[:, 4] = batch_s_all[:, 4] / 7000
                batch_s_all[:, 5] = batch_s_all[:, 5] / 90
                batch_s_all[:, 6] = batch_s_all[:, 6] / 84
                self.pre_fire_interception_assessment.fit(batch_s_all, labels_clss_all, verbose=0)  # verbose=2

            if self.memory.batch_s_interception_hit:
                batch_s_interception_hit = np.array(self.memory.batch_s_interception_hit, dtype=np.float32)
                labels_reg_time_steps_interception = np.array(self.memory.labels_reg_time_steps_interception, dtype=np.uint16)
                batch_s_interception_hit[:, 0] = batch_s_interception_hit[:, 0] + 2000  # x axis origin adjustment
                batch_s_interception_hit[:, :2] = batch_s_interception_hit[:, :2] / 7000
                batch_s_interception_hit[:, 2:4] = batch_s_interception_hit[:, 2:4] / 180
                batch_s_interception_hit[:, 4] = batch_s_interception_hit[:, 4] / 7000
                batch_s_interception_hit[:, 5] = batch_s_interception_hit[:, 5] / 90
                batch_s_interception_hit[:, 6] = batch_s_interception_hit[:, 6] / 84
                self.pre_fire_interception_time_steps_assessment.fit(batch_s_interception_hit, labels_reg_time_steps_interception, verbose=0)  # verbose=2

    ###################################

    def save_models(self, index):
        if train_type == TRAIN_FIRE_ANGLE_ASSESSMENT:
            self.fire_ang_assessment.save(self.h5_file_fire_ang)
        elif train_type == TRAIN_COLLISION_ASSESSMENTS:
            self.collision_assessment.save(self.h5_file_collision)
            self.collision_time_steps_assessment.save(self.h5_file_collision_time_steps)
        elif train_type == TRAIN_INTERCEPTION_ASSESSMENTS:
            self.interception_assessment.save(self.h5_file_interception)
            self.interception_time_steps_assessment.save(self.h5_file_interception_time_steps)
        elif train_type == TRAIN_PRE_FIRE_INTERCEPTION_ASSESSMENTS:
            self.pre_fire_interception_assessment.save(self.h5_file_pre_fire_interception)
            self.pre_fire_interception_time_steps_assessment.save(self.h5_file_pre_fire_interception_time_steps)

        self.save_weights()

        if (index + 1) % new_save_folder_mark == 0 and (index + 1) != n_episodes:
            self.counter += 1
            self.update_dirs()

    def save_weights(self):
        if train_type == TRAIN_FIRE_ANGLE_ASSESSMENT:
            self.fire_ang_assessment.save_weights(self.h5_file_fire_ang_weights)
        elif train_type == TRAIN_COLLISION_ASSESSMENTS:
            self.collision_assessment.save_weights(self.h5_file_collision_weights)
            self.collision_time_steps_assessment.save_weights(self.h5_file_collision_time_steps_weights)
        elif train_type == TRAIN_INTERCEPTION_ASSESSMENTS:
            self.interception_assessment.save_weights(self.h5_file_interception_weights)
            self.interception_time_steps_assessment.save_weights(self.h5_file_interception_time_steps_weights)
        elif train_type == TRAIN_PRE_FIRE_INTERCEPTION_ASSESSMENTS:
            self.pre_fire_interception_assessment.save_weights(self.h5_file_pre_fire_interception_weights)
            self.pre_fire_interception_time_steps_assessment.save_weights(self.h5_file_pre_fire_interception_time_steps_weights)

    def load_models(self):
        print('...Loading models...')
        if train_type == TRAIN_FIRE_ANGLE_ASSESSMENT:
            self.fire_ang_assessment = models.load_model(self.h5_file_fire_ang)
        elif train_type == TRAIN_COLLISION_ASSESSMENTS:
            self.collision_assessment = models.load_model(self.h5_file_collision)
            self.collision_time_steps_assessment = models.load_model(self.h5_file_collision_time_steps)
        elif train_type == TRAIN_INTERCEPTION_ASSESSMENTS:
            self.interception_assessment = models.load_model(self.h5_file_interception)
            self.interception_time_steps_assessment = models.load_model(self.h5_file_interception_time_steps)
        elif train_type == TRAIN_PRE_FIRE_INTERCEPTION_ASSESSMENTS:
            self.pre_fire_interception_assessment = models.load_model(self.h5_file_pre_fire_interception)
            self.pre_fire_interception_time_steps_assessment = models.load_model(self.h5_file_pre_fire_interception_time_steps)

    def load_weights(self):
        print('...Loading weights...')
        if train_type == TRAIN_FIRE_ANGLE_ASSESSMENT:
            self.fire_ang_assessment.load_weights(self.h5_file_fire_ang_weights)
        elif train_type == TRAIN_COLLISION_ASSESSMENTS:
            self.collision_assessment.load_weights(self.h5_file_collision_weights)
            self.collision_time_steps_assessment.load_weights(self.h5_file_collision_time_steps_weights)
        elif train_type == TRAIN_INTERCEPTION_ASSESSMENTS:
            self.interception_assessment.load_weights(self.h5_file_interception_weights)
            self.interception_time_steps_assessment.load_weights(self.h5_file_interception_time_steps_weights)
        elif train_type == TRAIN_PRE_FIRE_INTERCEPTION_ASSESSMENTS:
            self.pre_fire_interception_assessment.load_weights(self.h5_file_pre_fire_interception_weights)
            self.pre_fire_interception_time_steps_assessment.load_weights(self.h5_file_pre_fire_interception_time_steps_weights)

    ###################################

    def train(self):
        if train_type == TRAIN_COLLISION_ASSESSMENTS:  # enemy fires unlimited interceptors
            env = Game(visualize=visualize)
        else:
            env = Game(single_rocket_restriction=True, assessment=True, visualize=visualize)

        learn_episode_index = -1
        if load_checkpoint:
            try:
                print('...Loading learn_episode_index...')
                learn_episode_index = utils.SaverLoader.pickle_load('learn_episode_index', self.chkpt_dir)
            except FileNotFoundError:
                print('...No data to load...')

        print('\n', 'Training Started', '\n')
        train_start_time = datetime.datetime.now()

        starting_ep = learn_episode_index + 1
        for i in range(starting_ep, n_episodes):
            if i % ep_batch_num == 0:
                ep_start_time = datetime.datetime.now()

            if train_type == TRAIN_FIRE_ANGLE_ASSESSMENT or \
                    train_type == TRAIN_INTERCEPTION_ASSESSMENTS or \
                    train_type == TRAIN_PRE_FIRE_INTERCEPTION_ASSESSMENTS:
                self.train_interception_assessment_networks_episode(env)  # agent fires unlimited interceptors

            elif train_type == TRAIN_COLLISION_ASSESSMENTS:
                self.train_collision_assessment_network_episode(env)  # enemy fires unlimited interceptors

            if (i + 1) % ep_batch_num == 0:
                if train_type == TRAIN_COLLISION_ASSESSMENTS:
                    print('Episodes %d - %d ; Runtime: %s ; City hits: %d, Ground hits: %d' %
                          (i + 2 - ep_batch_num, i + 1,
                           str(datetime.datetime.now() - ep_start_time).split('.')[0],
                           self.memory.city_hit_counter, self.memory.ground_hit_counter))  # score: %.2f
                else:
                    print('Episodes %d - %d ; Runtime: %s ; Hits: %d, Misses: %d' %
                          (i + 2 - ep_batch_num, i + 1,
                           str(datetime.datetime.now() - ep_start_time).split('.')[0],
                           self.memory.hit_counter, ep_batch_num - self.memory.hit_counter))  # score: %.2f

                learn_episode_index = i
                learn_start_time = datetime.datetime.now()
                self.learn()
                print('Learn time: %s' % str(datetime.datetime.now() - learn_start_time).split('.')[0], '\n')

                if enable_models_saving:
                    utils.SaverLoader.pickle_save(learn_episode_index, 'learn_episode_index', self.chkpt_dir)
                    self.save_models(index=i)

            # if visualize and i == n_episodes - 1:
            #     env.close()

        print('\n', 'Training Ended ~~~ Episodes: %d ~~~ Runtime: %s' %
              (n_episodes - starting_ep, str(datetime.datetime.now() - train_start_time).split('.')[0]), '\n')

    def train_collision_assessment_network_episode(self, env):
        done = False
        env.reset()
        while not done:
            observation_, r, done = env.step(ACTION_NONE)
        self.memory.store_transitions_episode_end_collision_assessment(env)

    def train_interception_assessment_networks_episode(self, env):
        done = False

        s = env.get_state_single_rocket()

        while not done:
            current_angle = s[-1]
            possible_actions = [ACTION_NONE if len(s) == 1 else ACTION_FIRE]
            if -84 < current_angle:
                possible_actions.append(ACTION_LEFT)
            if current_angle < 84:
                possible_actions.append(ACTION_RIGHT)
            a = np.random.choice(possible_actions)

            observation_, r, done = env.step(a)
            s_ = env.get_state_single_rocket()
            s = s_

        # in the end of each episode, update rewards according to hit, and add final data to memory.
        hit_miss = self.memory.end_of_episode_process_get_hit_miss(env)

        return hit_miss


def main():  # train agent
    utils.DeviceSetUtils.set_device(devices_dict)
    agent = Agent()
    agent.train()


TRAIN_FIRE_ANGLE_ASSESSMENT = 1
TRAIN_COLLISION_ASSESSMENTS = 2
TRAIN_INTERCEPTION_ASSESSMENTS = 3
TRAIN_PRE_FIRE_INTERCEPTION_ASSESSMENTS = 4

train_type = TRAIN_FIRE_ANGLE_ASSESSMENT

multi_threading = False

visualize = False


base_dir = 'training/'

today = '19-11-14/'

load_models = False  # as opposed to weights

# training params
optimizer_type = utils.Optimizers.OPTIMIZER_Adam
ALPHA = 0.001

enable_models_saving = True

if multi_threading:
    episode_end_lock = Lock()


def set_sub_dir_according_to_train_type():
    global sub_dir

    if train_type == TRAIN_FIRE_ANGLE_ASSESSMENT:
        sub_dir = 'fire_angle_assessment/' + today
    elif train_type == TRAIN_COLLISION_ASSESSMENTS:
        sub_dir = 'collision_assessment/' + today
    elif train_type == TRAIN_INTERCEPTION_ASSESSMENTS:
        sub_dir = 'interception_assessment/timesteps/' + today
    elif train_type == TRAIN_PRE_FIRE_INTERCEPTION_ASSESSMENTS:
        sub_dir = 'pre_fire_interception_assessment/' + today


if __name__ == "__main__":
    load_checkpoint = False

    if train_type == TRAIN_COLLISION_ASSESSMENTS:
        n_episodes = 20000
        ep_batch_num = 1
        new_save_folder_mark = 500
    else:
        n_episodes = 2000000
        ep_batch_num = 100
        new_save_folder_mark = 50000

    devices_dict = None

    set_sub_dir_according_to_train_type()

    main()


def main_cloud(train_type_in, n_episodes_in, ep_batch_num_in, new_save_folder_mark_in, load_in):
    global train_type, load_checkpoint, n_episodes, ep_batch_num, new_save_folder_mark, devices_dict

    train_type = train_type_in

    load_checkpoint = load_in

    n_episodes = n_episodes_in
    ep_batch_num = ep_batch_num_in
    new_save_folder_mark = new_save_folder_mark_in

    devices_dict = {'XLA_GPU': 0}

    set_sub_dir_according_to_train_type()

    main()
