import math
import numpy as np
import utils
from networks_builder import NetworksBuilder


ACTION_LEFT = 0     # Change turret angle one step left (-6°)
ACTION_NONE = 1     # Do nothing
ACTION_RIGHT = 2    # Change turret angle one step right (+6°)
ACTION_FIRE = 3

COLLISION_TYPE_GROUND = 0
COLLISION_TYPE_CITY = 1

INTERCEPTION_TYPE_MISS = 0
INTERCEPTION_TYPE_HIT = 1


fc_layers_dims_fire_ang_assessment = [512, 512, 512]
fc_layers_dims_collision_assessment = [512, 512, 512]
fc_layers_dims_collision_time_steps_assessment = [512, 512, 512]
fc_layers_dims_interception_assessment = [512, 512, 512]
fc_layers_dims_interception_time_steps_assessment = [512, 512, 512]
fc_layers_dims_pre_fire_interception_assessment = [512, 512, 512]
fc_layers_dims_pre_fire_interception_time_steps_assessment = [512, 512, 512]


def get_state(observation, prev_s):
    r_data_list = update_observation(observation, prev_s, 0)
    i_data_list = update_observation(observation, prev_s, 1)
    cities_data = get_cities_data(observation[2])
    return r_data_list, i_data_list, cities_data, observation[3]


def update_observation(observation, prev_s, index):
    coordinates = observation[index]
    prev_data = prev_s[index] if prev_s is not None and prev_s[index] is not None else []

    if len(coordinates) != 0:

        deltas = np.zeros((coordinates.shape[0], 2))
        if index == 0:
            additional = np.zeros((coordinates.shape[0], 2))

        for i in range(len(coordinates)):
            match_found = False
            first_move = False
            while not match_found:

                # new interceptor (starts with coordinates (-2000, 0)):
                if index == 1 and coordinates[i][0] == -2000 and coordinates[i][1] == 0:
                    deltas[i, 0] = 0
                    deltas[i, 1] = 0
                    match_found = True
                    break

                # any rocket or existing interceptors:
                if index == 0 and (len(prev_data) == 0 or len(prev_data) < i + 1):  # new rocket (in an empty or non-empty list)
                    # meaning: no previous data exists (will never happen for an interceptor because we get the static starting point)
                    first_move = True
                    x_prev = 4800  # [m]  # -2000 if index == 1 else 4800
                    y_prev = 0
                    dx_prev = 0
                    dy_prev = 0
                else:
                    if index == 1 and prev_data[i][0] == -2000 and prev_data[i][1] == 0:  # first move of a new interceptor
                        first_move = True
                    x_prev = prev_data[i][0]
                    y_prev = prev_data[i][1]
                    dx_prev = prev_data[i][2]
                    dy_prev = prev_data[i][3]

                dx_curr = coordinates[i][0] - x_prev
                dy_curr = coordinates[i][1] - y_prev

                if len(prev_data) > i and (np.abs(dx_curr) > 180 or np.abs(dy_curr) > 180):
                    # print('R' if index == 0 else 'I', 'is gone:', prev_data[i][:4])
                    prev_data = np.delete(prev_data, i, 0)

                else:
                    # first condition - interceptor was fired at 0° ang
                    if (index == 1 and not first_move and coordinates[i][0] == x_prev == -2000) \
                            or (first_move or (0.9 < dx_curr / dx_prev < 1.15)):  # bounds: 0.896, 1.18 ; must include: 1.119
                        # in case of a previous 0° interceptor that disappeared - dx_curr / dx_prev == inf ; math.isinf(dx_curr / dx_prev) == True
                        deltas[i, 0] = dx_curr
                        deltas[i, 1] = dy_curr
                        match_found = True
                    elif index == 1 and dx_curr == 0 and dy_curr == 0:  # some interceptors don't move properly ('freeze' for a time-step)
                        deltas[i, 0] = dx_prev
                        deltas[i, 1] = dy_prev
                        match_found = True
                    else:  # previous rocket \ interceptor is gone
                        # print('R' if index == 0 else 'I', 'is gone:', prev_data[i][:4])
                        prev_data = np.delete(prev_data, i, 0)

                if index == 0 and match_found:
                    delta_x = coordinates[i][0] - (-2000)
                    delta_y = coordinates[i][1] - 0
                    dist = math.sqrt(math.pow(delta_x, 2) + math.pow(delta_y, 2))
                    # math.degrees() - from theta do degrees.
                    # theta is measured counter-clockwise from the +x axis. we need clockwise from the +y axis
                    # self.ang = math.degrees(math.atan2(-delta_y, delta_x)) + 90  # range: -90° to 270°
                    ang = math.degrees(math.atan2(delta_y, -delta_x)) - 90  # range: -270° to 90°
                    additional[i, 0] = dist
                    additional[i, 1] = ang

        # if len(prev_data) > len(coordinates):
        #     for prev in prev_data[len(coordinates):]:
        #         print('R' if index == 0 else 'I', 'is gone:', prev[:4])

        return np.hstack((coordinates, deltas, additional) if index == 0 else (coordinates, deltas))


def get_cities_data(cities):
    return np.array([cities[0, 0] - cities[0, 1] / 2, cities[0, 0] + cities[0, 1] / 2,
                     cities[1, 0] - cities[1, 1] / 2, cities[1, 0] + cities[1, 1] / 2])


class Agent:

    def __init__(self):
        self.fire_ang_assessment = NetworksBuilder.build_fire_ang_assessment_network()
        self.collision_assessment, self.collision_time_steps_assessment = NetworksBuilder.build_collision_assessment_networks()
        self.interception_assessment, self.interception_time_steps_assessment = NetworksBuilder.build_interception_assessment_networks()
        self.pre_fire_interception_assessment, self.pre_fire_interception_time_steps_assessment = NetworksBuilder.build_pre_fire_interception_assessment_networks()

        base_dir = 'weights/'
        self.fire_ang_assessment.load_weights(base_dir + 'ac_keras_fire_angle_weights.h5')
        self.collision_assessment.load_weights(base_dir + 'ac_keras_collision_weights.h5')
        self.collision_time_steps_assessment.load_weights(base_dir + 'ac_keras_collision_time_steps_weights.h5')
        self.interception_assessment.load_weights(base_dir + 'ac_keras_interception_weights.h5')
        self.interception_time_steps_assessment.load_weights(base_dir + 'ac_keras_interception_time_steps_weights.h5')
        self.pre_fire_interception_assessment.load_weights(base_dir + 'ac_keras_pre_fire_interception_weights.h5')
        self.pre_fire_interception_time_steps_assessment.load_weights(base_dir + 'ac_keras_pre_fire_interception_time_steps_weights.h5')

    def predict_fire_angle(self, s):
        s = np.array(s)
        s[:, 0] = s[:, 0] + 2000
        s[:, :2] = s[:, :2] / 7000
        s[:, 2:4] = s[:, 2:4] / 180
        s[:, 4] = s[:, 4] / 7000
        s[:, 5] = s[:, 5] / 90
        return self.fire_ang_assessment.predict(s[:, :6])

    def predict_collision(self, s):
        s = np.array(s)
        s[:, 0] = s[:, 0] / 4950
        s[:, 1] = s[:, 1] / 5965
        s[:, 2:4] = s[:, 2:4] / 180
        s[:, 4:8] = s[:, 4:8] / 4950
        probabilities = self.collision_assessment.predict(s)

        collision_classification = np.zeros(probabilities.shape[0], dtype=np.uint8)
        for i, p in enumerate(probabilities):
            try:
                clss = np.random.choice([i for i in range(2)], p=p)
            except ValueError:
                if np.sum(p) == 0:  # no prediction
                    p = [1, 0]  # predict ground (higher chance)
                else:
                    p /= np.sum(p)  # normalize
                clss = np.random.choice([i for i in range(2)], p=p)
            collision_classification[i] = clss

        return collision_classification

    def predict_time_steps_to_collision(self, s):
        s = np.array(s)
        s[:, 0] = s[:, 0] / 4950
        s[:, 1] = s[:, 1] / 5965
        s[:, 2:4] = s[:, 2:4] / 180
        s[:, 4:8] = s[:, 4:8] / 4950
        return self.collision_time_steps_assessment.predict(s)

    def predict_interception(self, s):
        s = np.array(s)
        s[:, 1] = s[:, 1] - 3160  # y axis origin adjustment
        s[:, 5] = s[:, 5] - 3160  # y axis origin adjustment
        s[:, 0] = s[:, 0] / 5180
        s[:, 1] = s[:, 1] / 3340
        s[:, 2:4] = s[:, 2:4] / 180
        s[:, 4] = s[:, 4] / 5180
        s[:, 5] = s[:, 5] / 3340
        s[:, 6:8] = s[:, 6:8] / 180
        probabilities = self.interception_assessment.predict(s)

        interception_classification = np.zeros(probabilities.shape[0], dtype=np.uint8)
        for i, p in enumerate(probabilities):
            try:
                clss = np.random.choice([i for i in range(2)], p=p)
            except ValueError:
                if np.sum(p) == 0:  # no prediction
                    p = [1, 0]  # predict miss (higher chance)
                else:
                    p /= np.sum(p)  # normalize
                clss = np.random.choice([i for i in range(2)], p=p)
            interception_classification[i] = clss

        return interception_classification

    def predict_time_steps_to_interception(self, s):
        s = np.array(s)
        s[:, 1] = s[:, 1] - 3160  # y axis origin adjustment
        s[:, 5] = s[:, 5] - 3160  # y axis origin adjustment
        s[:, 0] = s[:, 0] / 5180
        s[:, 1] = s[:, 1] / 3340
        s[:, 2:4] = s[:, 2:4] / 180
        s[:, 4] = s[:, 4] / 5180
        s[:, 5] = s[:, 5] / 3340
        s[:, 6:8] = s[:, 6:8] / 180
        return self.interception_time_steps_assessment.predict(s)

    def predict_pre_fire_interception(self, s):
        s = np.array(s)
        s[:, 0] = s[:, 0] + 2000  # x axis origin adjustment
        s[:, :2] = s[:, :2] / 7000
        s[:, 2:4] = s[:, 2:4] / 180
        s[:, 4] = s[:, 4] / 7000
        s[:, 5] = s[:, 5] / 90
        s[:, 6] = s[:, 6] / 84
        probabilities = self.pre_fire_interception_assessment.predict(s)

        pre_fire_interception_classification = np.zeros(probabilities.shape[0], dtype=np.uint8)
        for i, p in enumerate(probabilities):
            try:
                clss = np.random.choice([i for i in range(2)], p=p)
            except ValueError:
                if np.sum(p) == 0:  # no prediction
                    p = [1, 0]  # predict miss (higher chance)
                else:
                    p /= np.sum(p)  # normalize
                clss = np.random.choice([i for i in range(2)], p=p)
            pre_fire_interception_classification[i] = clss

        return pre_fire_interception_classification

    def predict_time_steps_to_pre_fire_interception(self, s):
        s = np.array(s)
        s[:, 0] = s[:, 0] + 2000  # x axis origin adjustment
        s[:, :2] = s[:, :2] / 7000
        s[:, 2:4] = s[:, 2:4] / 180
        s[:, 4] = s[:, 4] / 7000
        s[:, 5] = s[:, 5] / 90
        s[:, 6] = s[:, 6] / 84
        return self.pre_fire_interception_time_steps_assessment.predict(s)


class Rocket:

    def __init__(self, collision_classification, t_to_collision):
        self.collision_classification = collision_classification
        self.base_threat = 15 if collision_classification == COLLISION_TYPE_CITY else 1
        self.t_to_collision = t_to_collision

        self.is_interceptable = False

        # note that even though a rocket is not interceptable by a current FIRE from the immediate fire-ang angles,
        #   it's still can be on the way to be intercepted... (e.g. if an interceptor was previously fired)
        self.final_preference_score = 0


class InterceptibleRocket(Rocket):

    def __init__(self, collision_classification, t_to_collision,
                 t_to_pre_fire_interception_from_fire_ang, fire_angle, current_ang):

        super().__init__(collision_classification, t_to_collision)

        self.is_interceptable = True

        self.fire_angle = fire_angle
        self.t_to_fire_angle = int(np.abs(fire_angle - current_ang) // 6)
        self.t_to_pre_fire_interception_from_fire_ang = t_to_pre_fire_interception_from_fire_ang

        denominator = (t_to_collision - t_to_pre_fire_interception_from_fire_ang - self.t_to_fire_angle)
        # further penalize high t_to_fire_angle missiles:
        # note: the +1 is important when: t_to_fire_angle=0 and fire_action_range<0
        #   (without it, it would eventually turn the high-profile rocket's weighted_preference_score negative)
        denominator *= (self.t_to_fire_angle + 1) if self.t_to_fire_angle > fire_action_range else 1

        if denominator == 0:
            denominator = -1

        self.weighted_preference_score = self.base_threat / denominator

        self.final_preference_score = self.weighted_preference_score


optimal_initial_angle = 60
fire_threshold = 7 + 1  # reload_time // dt = 1.5 // 0.2 = 7 (this is the cooldown period)
steps_since_fire = fire_threshold                       # range: 1+ (min 1)
fire_action_range = fire_threshold - steps_since_fire   # range: 7- (max 7)

config = 0
use_gpu = False


class Solution:

    def __init__(self, config_in):
        if use_gpu:
            devices_dict = {'XLA_GPU': 0}
            utils.DeviceSetUtils.set_device(devices_dict)

        self.agent = Agent()
        self.s = None

        global config
        config = config_in

    def reset(self):
        self.s = None

        global steps_since_fire, fire_action_range
        steps_since_fire = fire_threshold
        fire_action_range = fire_threshold - steps_since_fire

    def get_action_according_to_ang(self, fire_angle, enable_fire):
        current_ang = self.s[3]
        if current_ang < fire_angle - 3:
            a = ACTION_RIGHT
        elif current_ang > fire_angle + 3:
            a = ACTION_LEFT
        else:
            a = ACTION_FIRE if enable_fire else ACTION_NONE
        return a

    def get_action(self, observation):
        """
        Action Decision algorithm
        :param observation:
        :return:
        """

        self.s = get_state(observation, self.s)
        r_data_list, i_data_list, cities_data, current_ang = self.s

        #############################################################

        # no rockets --> move towards the optimal initial angle
        if r_data_list is None:
            return self.get_action_according_to_ang(optimal_initial_angle, enable_fire=False)

        #############################################################

        # 1. gather data for every rocket - do all non-interceptor related rockets assessments
        fire_angle_list = self.agent.predict_fire_angle(r_data_list)
        fire_angle_list_step_round_down = (fire_angle_list // 6) * 6
        fire_angle_list_step_round_up = (fire_angle_list // 6 + 1) * 6

        rocket_cities_data = np.zeros((len(r_data_list), 8), dtype=np.float32)
        rocket_fire_ang_data_step_round_down = np.zeros((len(r_data_list), 7), dtype=np.float32)
        rocket_fire_ang_data_step_round_up = np.zeros((len(r_data_list), 7), dtype=np.float32)
        for r_index, r_data in enumerate(r_data_list):
            rocket_cities_data[r_index] = np.array([*r_data[:4], *cities_data], dtype=np.float32)
            rocket_fire_ang_data_step_round_down[r_index] = np.array([*r_data, fire_angle_list_step_round_down[r_index]], dtype=np.float32)
            rocket_fire_ang_data_step_round_up[r_index] = np.array([*r_data, fire_angle_list_step_round_up[r_index]], dtype=np.float32)

        collision_classification_list = self.agent.predict_collision(rocket_cities_data)
        time_steps_to_collision_list = self.agent.predict_time_steps_to_collision(rocket_cities_data)

        fire_ang_step_round_down_pre_fire_interception_list = self.agent.predict_pre_fire_interception(rocket_fire_ang_data_step_round_down)
        fire_ang_step_round_up_pre_fire_interception_list = self.agent.predict_pre_fire_interception(rocket_fire_ang_data_step_round_up)

        #############################################################

        # 2. set data for each rocket ('t' = 'time_steps')
        r_objects_list = []
        for r_index in range(len(r_data_list)):

            collision_classification = collision_classification_list[r_index]

            t_to_collision = time_steps_to_collision_list[r_index][0]
            t_to_collision = 1 if t_to_collision < 1 else int(round(t_to_collision))

            is_interceptable = fire_ang_step_round_down_pre_fire_interception_list[r_index] == INTERCEPTION_TYPE_HIT or \
                               fire_ang_step_round_up_pre_fire_interception_list[r_index] == INTERCEPTION_TYPE_HIT

            if not is_interceptable:
                r_objects_list.append(Rocket(collision_classification, t_to_collision))

            else:
                if fire_ang_step_round_down_pre_fire_interception_list[r_index] == INTERCEPTION_TYPE_HIT:
                    fire_angle = fire_angle_list_step_round_down[r_index][0]
                else:
                    fire_angle = fire_angle_list_step_round_up[r_index][0]

                t_to_pre_fire_interception_from_fire_ang = self.agent.predict_time_steps_to_pre_fire_interception(
                    np.array([[*r_data_list[r_index], fire_angle]], dtype=np.float32))[0, 0]
                t_to_pre_fire_interception_from_fire_ang = int(round(t_to_pre_fire_interception_from_fire_ang))

                r_objects_list.append(InterceptibleRocket(
                    collision_classification, t_to_collision,
                    t_to_pre_fire_interception_from_fire_ang, fire_angle, current_ang))

        #############################################################

        # 3. account for existing interceptors
        if i_data_list is not None:

            interceptions_by_time_steps_dict = {}

            # assess interception for every I-R pair
            for i_index, i_data in enumerate(i_data_list):
                if i_data[1] == 0:  # if i_y == 0 : an interceptor that was just fired
                    break

                interceptor_rockets_data = np.zeros((len(r_data_list), 8), dtype=np.float32)
                for r_index, r_data in enumerate(r_data_list):
                    interceptor_rockets_data[r_index] = np.array([*r_data[:4], *i_data], dtype=np.float32)

                interception_classification_list = self.agent.predict_interception(interceptor_rockets_data)
                time_steps_to_interception_list = self.agent.predict_time_steps_to_interception(interceptor_rockets_data)

                for r_index, r_data in enumerate(r_data_list):
                    if interception_classification_list[r_index] == INTERCEPTION_TYPE_HIT:

                        time_steps_to_interception = time_steps_to_interception_list[r_index][0]
                        time_steps_to_interception = 1 if time_steps_to_interception < 1 else int(round(time_steps_to_interception))

                        if time_steps_to_interception in interceptions_by_time_steps_dict:
                            interceptions_by_i_dict = interceptions_by_time_steps_dict[time_steps_to_interception]
                        else:
                            interceptions_by_i_dict = {}
                            interceptions_by_time_steps_dict[time_steps_to_interception] = interceptions_by_i_dict

                        if i_index in interceptions_by_i_dict:
                            r_indices_list = interceptions_by_i_dict[i_index]
                        else:
                            r_indices_list = []
                            interceptions_by_i_dict[i_index] = r_indices_list

                        r_indices_list.append(r_index)

            # account for early interceptions only
            if len(interceptions_by_time_steps_dict) > 0:

                # update rocket's final_preference_score according to interception status
                time_steps_list = list(interceptions_by_time_steps_dict.keys())
                time_steps_list.sort()
                for time_steps_to_interception in time_steps_list:
                    if time_steps_to_interception in interceptions_by_time_steps_dict:  # some may be removed in the process

                        empty_t_steps = set()
                        empty_i = set()

                        # go over every interceptor and its future intercepted rockets:
                        for i_index, r_indices_list in interceptions_by_time_steps_dict[time_steps_to_interception].items():

                            # remove that interceptor's following interceptions (in following time-steps):
                            for t_step, interceptions_by_i_dict in interceptions_by_time_steps_dict.items():
                                if t_step != time_steps_to_interception and i_index in interceptions_by_i_dict:
                                    del interceptions_by_i_dict[i_index]
                                    if len(interceptions_by_i_dict) == 0:
                                        empty_t_steps.add(t_step)

                            # handle that interceptor's future intercepted rockets':
                            #   1) change their final_preference_score.
                            #   2) remove them from other interceptions lists.
                            for r_index in r_indices_list:
                                r_objects_list[r_index].final_preference_score = -1

                                for t_step, interceptions_by_i_dict in interceptions_by_time_steps_dict.items():
                                    if t_step != time_steps_to_interception:
                                        for i, r_indices_list in interceptions_by_i_dict.items():
                                            if r_index in r_indices_list:
                                                r_indices_list.remove(r_index)
                                                if len(r_indices_list) == 0:
                                                    empty_i.add(i)

                        # remove current time-step from dict:
                        del interceptions_by_time_steps_dict[time_steps_to_interception]

                        # remove empty from dict:
                        for t_step, interceptions_by_i_dict in interceptions_by_time_steps_dict.items():
                            for i in empty_i:
                                if i in interceptions_by_i_dict:
                                    del interceptions_by_i_dict[i]
                            if len(interceptions_by_i_dict) == 0:
                                empty_t_steps.add(t_step)

                        # remove empty time-step from dict:
                        for t_step in empty_t_steps:
                            del interceptions_by_time_steps_dict[t_step]

        #############################################################

        # 4. choose rocket according to final_preference_score & choose action accordingly
        a = ACTION_NONE
        curr_i_rockets_arr = np.array([(r_index, r_object.final_preference_score, r_object.t_to_fire_angle)
                                       for r_index, r_object in enumerate(r_objects_list)
                                       if r_object.is_interceptable])

        # if there are immediately interceptable rockets (rockets with a valid\sure-hit fire-ang) -
        #   1) check if they're already on the way to be intercepted
        #   2) if not - set the high-profile one (city hit + closer fire_ang) as a FIRE target
        if len(curr_i_rockets_arr) > 0 and curr_i_rockets_arr[:, 1].max() > 0:
            max_preference_i_r_indices = np.where(
                curr_i_rockets_arr[:, 1] == curr_i_rockets_arr[:, 1].max()
            )[0]

            if len(max_preference_i_r_indices) > 1:
                # if multiple rockets have the same final_preference_score, choose the one closest to FIRE:
                i = np.argmin([curr_i_rockets_arr[index, 2] for index in max_preference_i_r_indices])
            else:
                i = 0
            index = max_preference_i_r_indices[i]
            chosen_rocket_index = int(curr_i_rockets_arr[index, 0])
            a = self.get_action_according_to_ang(r_objects_list[chosen_rocket_index].fire_angle, enable_fire=True)

        if config == 0:
            # if none of the rockets are immediately interceptable -
            #   1) check if they're already on the way to be intercepted
            #   2) if not - set the high-profile one (non-intercepted + city hit + closer collision) as a MOVE target
            if a == ACTION_NONE and len(r_objects_list) > len(curr_i_rockets_arr):  # no target rocket & non-interceptable rocket exists

                # no interceptable rockets --> move towards a city threat rocket of them
                curr_non_i_rockets_arr = np.array([(r_index, r_object.t_to_collision)
                                                   for r_index, r_object in enumerate(r_objects_list)
                                                   if (not r_object.is_interceptable
                                                       and r_object.collision_classification == COLLISION_TYPE_CITY
                                                       and r_object.final_preference_score != -1)])

                # check if any rockets answer the above criteria (city hit that's not about to be intercepted)
                if len(curr_non_i_rockets_arr) > 0:
                    if len(curr_non_i_rockets_arr) > 1:
                        # if multiple rockets, choose the one closest to collision:
                        index = np.argmin(curr_non_i_rockets_arr[:, 1])
                    else:
                        index = 0
                    chosen_rocket_index = int(curr_non_i_rockets_arr[index, 0])
                    a = self.get_action_according_to_ang(fire_angle_list[chosen_rocket_index], enable_fire=False)

        #############################################################

        # 5. increment steps_since_fire if needed, and update fire_action_range
        global steps_since_fire
        if a == ACTION_FIRE and steps_since_fire >= fire_threshold:  # meanning: a successful FIRE action (it'll actually FIRE)
            steps_since_fire = 1
        else:  # meaning: either a non-FIRE action or unsuccessful FIRE action (it'll try, but nothing will happen)
            steps_since_fire += 1

        global fire_action_range
        fire_action_range = fire_threshold - steps_since_fire

        return a

