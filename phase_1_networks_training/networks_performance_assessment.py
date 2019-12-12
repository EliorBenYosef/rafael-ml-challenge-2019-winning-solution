from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

import numpy as np
import datetime

import utils
from networks_builder import NetworksBuilder

from .Interceptor_V2_training import Game,\
    ACTION_LEFT, ACTION_NONE, ACTION_RIGHT, ACTION_FIRE,\
    COLLISION_TYPE_GROUND, COLLISION_TYPE_CITY, COLLISION_TYPE_ROCKET

from threading import Lock


class Memory(object):

    def __init__(self):
        self.n_actions = 4
        self.action_space = [i for i in range(self.n_actions)]

        self.hit_counter = 0  # miss_counter = batch - hit_counter

        if test_type == TEST_COLLISION_ASSESSMENTS:
            self.batch_s_collision = []  # r_c2
            self.labels_clss_collision = []
            self.labels_reg_time_steps_collision = []

            self.city_hit_counter = 0
            self.ground_hit_counter = 0

        else:
            # if test_type == TEST_FIRE_ANGLE_ASSESSMENT:
            self.batch_s_fire_angle_hit = []  # r_ang
            self.labels_reg_fire_angle_hit = []

            # elif test_type == TEST_INTERCEPTION_ASSESSMENTS:
            self.batch_s_interception_hit = []  # r_i
            self.labels_clss_interception_hit = []
            self.labels_reg_time_steps_interception = []

            self.batch_s_interception_miss = []
            self.labels_clss_interception_miss = []

            # elif test_type == TEST_PRE_FIRE_INTERCEPTION_ASSESSMENTS:
            self.batch_s_pre_fire_interception_hit = []  # r_i
            self.labels_clss_pre_fire_interception_hit = []
            self.labels_reg_time_steps_pre_fire_interception = []

            self.batch_s_pre_fire_interception_miss = []
            self.labels_clss_pre_fire_interception_miss = []

    def end_of_episode_process_get_hit_miss(self, env):
        # in the end of each episode, update rewards according to hit, and add final data to memory.
        rocket = env.total_rockets[0]

        is_hit = rocket.collision_classification == COLLISION_TYPE_ROCKET
        if is_hit:  # single hit
            self.hit_counter += 1

        # if test_type == TEST_FIRE_ANGLE_ASSESSMENT:
        if is_hit:  # single hit
            self.save_rocket_and_ang_data(rocket, env)
        # elif test_type == TEST_INTERCEPTION_ASSESSMENTS:
        self.store_interception_data_episode_end(rocket, env)
        # elif test_type == TEST_PRE_FIRE_INTERCEPTION_ASSESSMENTS:
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
            self.batch_s_pre_fire_interception_hit.append(np.array([*r_data, interceptor.fire_ang]))
            a_indices_one_hot = np.zeros(2, dtype=np.ubyte)
            a_indices_one_hot[1] = 1  # hit [0,1]
            self.labels_clss_pre_fire_interception_hit.append(a_indices_one_hot)
            self.labels_reg_time_steps_pre_fire_interception.append(interceptor.collision_time_step - interceptor.appearance_time_step)

        else:  # all misses - save all missed interceptors data
            interceptors = env.total_interceptors
            for interceptor in interceptors:
                r_data = rocket.data[interceptor.appearance_time_step - rocket.appearance_time_step]
                self.batch_s_pre_fire_interception_miss.append(np.array([*r_data, interceptor.fire_ang]))
                a_indices_one_hot = np.zeros(2, dtype=np.ubyte)
                a_indices_one_hot[0] = 1  # miss [1,0]
                self.labels_clss_pre_fire_interception_miss.append(a_indices_one_hot)

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

        if test_type == TEST_COLLISION_ASSESSMENTS:
            # elif test_type == TEST_COLLISION_ASSESSMENTS:
            self.batch_s_collision = []
            self.labels_clss_collision = []
            self.labels_reg_time_steps_collision = []

            self.city_hit_counter = 0
            self.ground_hit_counter = 0

        else:
            # if test_type == TEST_FIRE_ANGLE_ASSESSMENT:
            self.batch_s_fire_angle_hit = []
            self.labels_reg_fire_angle_hit = []

            # elif test_type == TEST_INTERCEPTION_ASSESSMENTS:
            self.batch_s_interception_hit = []
            self.labels_clss_interception_hit = []
            self.labels_reg_time_steps_interception = []

            self.batch_s_interception_miss = []
            self.labels_clss_interception_miss = []

            # elif test_type == TEST_PRE_FIRE_INTERCEPTION_ASSESSMENTS:
            self.batch_s_pre_fire_interception_hit = []
            self.labels_clss_pre_fire_interception_hit = []
            self.labels_reg_time_steps_pre_fire_interception = []

            self.batch_s_pre_fire_interception_miss = []
            self.labels_clss_pre_fire_interception_miss = []


fc_layers_dims_fire_ang_assessment = [512, 512, 512]
fc_layers_dims_collision_assessment = [512, 512, 512]
fc_layers_dims_collision_time_steps_assessment = [512, 512, 512]
fc_layers_dims_interception_assessment = [512, 512, 512]
fc_layers_dims_interception_time_steps_assessment = [512, 512, 512]
fc_layers_dims_pre_fire_interception_assessment = [512, 512, 512]
fc_layers_dims_pre_fire_interception_time_steps_assessment = [512, 512, 512]

optimizer_type = utils.Optimizers.OPTIMIZER_Adam
ALPHA = 0.001  # lr_actor  # ep.20: alpha = 0.0003, ep.120: alpha = 0.0001


class Agent:

    def __init__(self):
        self.fire_ang_assessment = NetworksBuilder.build_fire_ang_assessment_network(optimizer_type, ALPHA)
        self.collision_assessment, self.collision_time_steps_assessment = NetworksBuilder.build_collision_assessment_networks(optimizer_type, ALPHA)
        self.interception_assessment, self.interception_time_steps_assessment = NetworksBuilder.build_interception_assessment_networks(optimizer_type, ALPHA)
        self.pre_fire_interception_assessment, self.pre_fire_interception_time_steps_assessment = NetworksBuilder.build_pre_fire_interception_assessment_networks(optimizer_type, ALPHA)

        self.fire_ang_assessment.load_weights('ac_keras_fire_angle_weights.h5')
        self.collision_assessment.load_weights('ac_keras_collision_weights.h5')
        self.collision_time_steps_assessment.load_weights('ac_keras_collision_time_steps_weights.h5')
        self.interception_assessment.load_weights('ac_keras_interception_weights.h5')
        self.interception_time_steps_assessment.load_weights('ac_keras_interception_time_steps_weights.h5')
        self.pre_fire_interception_assessment.load_weights('ac_keras_pre_fire_interception_weights.h5')
        self.pre_fire_interception_time_steps_assessment.load_weights('ac_keras_pre_fire_interception_time_steps_weights.h5')

        self.memory = Memory()

    ###################################

    def test(self):
        self.test_assessment()
        self.memory.reset_memory()

    def test_assessment(self):
        if test_type == TEST_COLLISION_ASSESSMENTS:
            if self.memory.batch_s_collision:
                batch_s_collision = np.array(self.memory.batch_s_collision, dtype=np.float32)
                labels_clss_collision = np.array(self.memory.labels_clss_collision, dtype=np.ubyte)
                labels_reg_time_steps_collision = np.array(self.memory.labels_reg_time_steps_collision, dtype=np.uint16)
                batch_s_collision[:, 0] = batch_s_collision[:, 0] / 4950
                batch_s_collision[:, 1] = batch_s_collision[:, 1] / 5965
                batch_s_collision[:, 2:4] = batch_s_collision[:, 2:4] / 180
                batch_s_collision[:, 4:8] = batch_s_collision[:, 4:8] / 4950
                collision_assessment_loss, collision_assessment_metrics = self.collision_assessment.evaluate(batch_s_collision, labels_clss_collision, verbose=0)  # verbose=2
                collision_time_steps_assessment_loss, collision_time_steps_assessment_metrics = self.collision_time_steps_assessment.evaluate(batch_s_collision, labels_reg_time_steps_collision, verbose=0)  # verbose=2
                print('collision_assessment - loss:', collision_assessment_loss, '; metrics:', collision_assessment_metrics)
                print('collision_time_steps_assessment - loss:', collision_time_steps_assessment_loss, '; metrics:', collision_time_steps_assessment_metrics)

        else:
            # if test_type == TEST_FIRE_ANGLE_ASSESSMENT:
            if self.memory.batch_s_fire_angle_hit:
                batch_s_fire_angle_hit = np.array(self.memory.batch_s_fire_angle_hit, dtype=np.float32)
                labels_reg_fire_angle_hit = np.array(self.memory.labels_reg_fire_angle_hit, dtype=np.float32)
                batch_s_fire_angle_hit[:, 0] = batch_s_fire_angle_hit[:, 0] + 2000  # x axis origin adjustment
                batch_s_fire_angle_hit[:, :2] = batch_s_fire_angle_hit[:, :2] / 7000
                batch_s_fire_angle_hit[:, 2:4] = batch_s_fire_angle_hit[:, 2:4] / 180
                batch_s_fire_angle_hit[:, 4] = batch_s_fire_angle_hit[:, 4] / 7000
                batch_s_fire_angle_hit[:, 5] = batch_s_fire_angle_hit[:, 5] / 90
                fire_ang_assessment_loss, fire_ang_assessment_metrics = self.fire_ang_assessment.evaluate(batch_s_fire_angle_hit, labels_reg_fire_angle_hit, verbose=0)  # verbose=2
                print('fire_ang_assessment - loss:', fire_ang_assessment_loss, '; metrics:', fire_ang_assessment_metrics)

            # elif test_type == TEST_INTERCEPTION_ASSESSMENTS:
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
                interception_assessment_loss, interception_assessment_metrics = self.interception_assessment.evaluate(batch_s_all, labels_clss_all, verbose=0)  # verbose=2
                print('interception_assessment - loss:', interception_assessment_loss, '; metrics:', interception_assessment_metrics)

            if self.memory.batch_s_interception_hit:
                batch_s_pre_fire_interception_hit = np.array(self.memory.batch_s_interception_hit, dtype=np.float32)
                labels_reg_time_steps_interception = np.array(self.memory.labels_reg_time_steps_interception, dtype=np.uint16)
                # y axis origin adjustment formula: + 180 - (6500 + 180) / 2 = 180 - 3340 = - 3160
                batch_s_pre_fire_interception_hit[:, 1] = batch_s_pre_fire_interception_hit[:, 1] - 3160  # y axis origin adjustment
                batch_s_pre_fire_interception_hit[:, 5] = batch_s_pre_fire_interception_hit[:, 5] - 3160  # y axis origin adjustment
                batch_s_pre_fire_interception_hit[:, 0] = batch_s_pre_fire_interception_hit[:, 0] / 5180
                batch_s_pre_fire_interception_hit[:, 1] = batch_s_pre_fire_interception_hit[:, 1] / 3340
                batch_s_pre_fire_interception_hit[:, 2:4] = batch_s_pre_fire_interception_hit[:, 2:4] / 180
                batch_s_pre_fire_interception_hit[:, 4] = batch_s_pre_fire_interception_hit[:, 4] / 5180
                batch_s_pre_fire_interception_hit[:, 5] = batch_s_pre_fire_interception_hit[:, 5] / 3340
                batch_s_pre_fire_interception_hit[:, 6:8] = batch_s_pre_fire_interception_hit[:, 6:8] / 180
                interception_time_steps_assessment_loss, interception_time_steps_assessment_metrics = self.interception_time_steps_assessment.evaluate(batch_s_pre_fire_interception_hit, labels_reg_time_steps_interception, verbose=0)  # verbose=2
                print('interception_time_steps_assessment - loss:', interception_time_steps_assessment_loss, '; metrics:', interception_time_steps_assessment_metrics)

            # elif test_type == TEST_PRE_FIRE_INTERCEPTION_ASSESSMENTS:
            if self.memory.batch_s_pre_fire_interception_hit or self.memory.batch_s_pre_fire_interception_miss:  # making sure interceptors were fired
                if self.memory.batch_s_pre_fire_interception_hit and self.memory.batch_s_pre_fire_interception_miss:    # hits & misses
                    batch_s_all = np.row_stack((self.memory.batch_s_pre_fire_interception_hit, self.memory.batch_s_pre_fire_interception_miss))
                    labels_clss_all = np.row_stack((self.memory.labels_clss_pre_fire_interception_hit, self.memory.labels_clss_pre_fire_interception_miss))
                elif self.memory.batch_s_pre_fire_interception_hit:                               # only hits
                    batch_s_all = np.array(self.memory.batch_s_pre_fire_interception_hit, dtype=np.float32)
                    labels_clss_all = np.array(self.memory.labels_clss_pre_fire_interception_hit, dtype=np.ubyte)
                else:                                                           # only misses
                    batch_s_all = np.array(self.memory.batch_s_pre_fire_interception_miss, dtype=np.float32)
                    labels_clss_all = np.array(self.memory.labels_clss_pre_fire_interception_miss, dtype=np.ubyte)
                batch_s_all[:, 0] = batch_s_all[:, 0] + 2000  # x axis origin adjustment
                batch_s_all[:, :2] = batch_s_all[:, :2] / 7000
                batch_s_all[:, 2:4] = batch_s_all[:, 2:4] / 180
                batch_s_all[:, 4] = batch_s_all[:, 4] / 7000
                batch_s_all[:, 5] = batch_s_all[:, 5] / 90
                batch_s_all[:, 6] = batch_s_all[:, 6] / 84
                pre_fire_interception_assessment_loss, pre_fire_interception_assessment_metrics = self.pre_fire_interception_assessment.evaluate(batch_s_all, labels_clss_all, verbose=0)  # verbose=2
                print('pre_fire_interception_assessment - loss:', pre_fire_interception_assessment_loss, '; metrics:', pre_fire_interception_assessment_metrics)

            if self.memory.batch_s_pre_fire_interception_hit:
                batch_s_pre_fire_interception_hit = np.array(self.memory.batch_s_pre_fire_interception_hit, dtype=np.float32)
                labels_reg_time_steps_pre_fire_interception = np.array(self.memory.labels_reg_time_steps_pre_fire_interception, dtype=np.uint16)
                batch_s_pre_fire_interception_hit[:, 0] = batch_s_pre_fire_interception_hit[:, 0] + 2000  # x axis origin adjustment
                batch_s_pre_fire_interception_hit[:, :2] = batch_s_pre_fire_interception_hit[:, :2] / 7000
                batch_s_pre_fire_interception_hit[:, 2:4] = batch_s_pre_fire_interception_hit[:, 2:4] / 180
                batch_s_pre_fire_interception_hit[:, 4] = batch_s_pre_fire_interception_hit[:, 4] / 7000
                batch_s_pre_fire_interception_hit[:, 5] = batch_s_pre_fire_interception_hit[:, 5] / 90
                batch_s_pre_fire_interception_hit[:, 6] = batch_s_pre_fire_interception_hit[:, 6] / 84
                pre_fire_interception_time_steps_assessment_loss, pre_fire_interception_time_steps_assessment_metrics = self.pre_fire_interception_time_steps_assessment.evaluate(batch_s_pre_fire_interception_hit, labels_reg_time_steps_pre_fire_interception, verbose=0)  # verbose=2
                print('pre_fire_interception_time_steps_assessment - loss:', pre_fire_interception_time_steps_assessment_loss, '; metrics:', pre_fire_interception_time_steps_assessment_metrics)

    ###################################

    def test_networks(self):
        if test_type == TEST_COLLISION_ASSESSMENTS:  # enemy fires unlimited interceptors
            env = Game()
        else:
            env = Game(single_rocket_restriction=True, assessment=True)

        print('\n', 'Testing Started', '\n')
        test_start_time = datetime.datetime.now()

        for i in range(0, n_episodes):
            if i % ep_batch_num == 0:
                ep_start_time = datetime.datetime.now()

            if test_type == TEST_COLLISION_ASSESSMENTS:
                self.test_collision_assessment_network_episode(env)  # enemy fires unlimited interceptors
            else:
                self.test_interception_assessment_networks_episode(env)  # agent fires unlimited interceptors

            if (i + 1) % ep_batch_num == 0:
                if test_type == TEST_COLLISION_ASSESSMENTS:
                    print('Episodes %d - %d ; Runtime: %s ; City hits: %d, Ground hits: %d' %
                          (i + 2 - ep_batch_num, i + 1,
                           str(datetime.datetime.now() - ep_start_time).split('.')[0],
                           self.memory.city_hit_counter, self.memory.ground_hit_counter))  # score: %.2f
                else:
                    print('Episodes %d - %d ; Runtime: %s ; Hits: %d, Misses: %d' %
                          (i + 2 - ep_batch_num, i + 1,
                           str(datetime.datetime.now() - ep_start_time).split('.')[0],
                           self.memory.hit_counter, ep_batch_num - self.memory.hit_counter))  # score: %.2f

                episodic_test_start_time = datetime.datetime.now()
                self.test()
                print('Test time: %s' % str(datetime.datetime.now() - episodic_test_start_time).split('.')[0], '\n')

        print('\n', 'Testing Ended ~~~ Episodes: %d ~~~ Runtime: %s' %
              (n_episodes, str(datetime.datetime.now() - test_start_time).split('.')[0]), '\n')

    def test_collision_assessment_network_episode(self, env):
        done = False
        env.reset()
        while not done:
            observation_, r, done = env.step(ACTION_NONE)
        self.memory.store_transitions_episode_end_collision_assessment(env)

    def test_interception_assessment_networks_episode(self, env):
        done = False

        observation = env.reset()
        # s = custom_env.get_state(observation, None)
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
            # s_ = custom_env.get_state(observation_, s)
            s_ = env.get_state_single_rocket()
            observation, s = observation_, s_

        # in the end of each episode, update rewards according to hit, and add final data to memory.
        hit_miss = self.memory.end_of_episode_process_get_hit_miss(env)

        return hit_miss


def test_agent_and_plot_scores():
    utils.DeviceSetUtils.set_device(devices_dict)
    agent = Agent()
    agent.test_networks()


TEST_COLLISION_ASSESSMENTS = 0
TEST_OTHER_ASSESSMENTS = 1
test_type = TEST_COLLISION_ASSESSMENTS

multi_threading = False


if multi_threading:
    episode_end_lock = Lock()


if __name__ == "__main__":
    test_type = TEST_COLLISION_ASSESSMENTS
    n_episodes = 20000
    ep_batch_num = 1
    new_save_folder_mark = 500

    # test_type = TEST_OTHER_ASSESSMENTS
    # n_episodes = 2000000
    # ep_batch_num = 100
    # new_save_folder_mark = 50000

    devices_dict = None

    test_agent_and_plot_scores()


def main_cloud(test_type_in, n_episodes_in, ep_batch_num_in, new_save_folder_mark_in, load_in):
    global test_type, n_episodes, ep_batch_num, new_save_folder_mark, devices_dict

    test_type = test_type_in

    n_episodes = n_episodes_in
    ep_batch_num = ep_batch_num_in
    new_save_folder_mark = new_save_folder_mark_in

    devices_dict = {'XLA_GPU': 0}

    test_agent_and_plot_scores()
