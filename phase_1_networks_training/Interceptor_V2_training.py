import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
# from PIL import Image


ACTION_LEFT = 0  # Change turret angle one step left
ACTION_NONE = 1  # Do nothing in the current game step
ACTION_RIGHT = 2  # Change turret angle one step right
ACTION_FIRE = 3

COLLISION_TYPE_GROUND = 0
COLLISION_TYPE_CITY = 1
COLLISION_TYPE_ROCKET = 2


class World:

    def __init__(self):
        self.width = 10000  # [m]
        self.height = 4000  # [m]
        self.dt = 0.2  # [sec]
        self.g = 9.8  # Gravity [m/sec**2]
        self.fric = 5e-7  # Air friction [Units of Science]
        self.rocket_prob = 1  # expected rockets per sec

        # # original:
        # self.reward_city = -15
        # self.reward_open = -1
        # self.reward_fire = -1
        # self.reward_intercept = 4

        # mine:
        self.reward_city = -99
        self.reward_open = -99
        self.reward_fire = 0
        self.reward_intercept = 101
        self.reward_time_step = -1  # mine

        # # mine ang assessment:
        # self.reward_city = -100
        # self.reward_open = -100
        # self.reward_fire = 0
        # self.reward_intercept = 100
        # self.reward_time_step = 0  # mine

        self.reward_illogical_action = -1  # mine

        self.time = 0  # [sec]
        self.time_step_index = -1         # mine
        self.step_reward = 0
        self.game_score = 0         # mine

    def reset(self):
        self.time = 0  # [sec]
        self.time_step_index = -1
        self.step_reward = 0
        self.game_score = 0


class Turret:

    def __init__(self, friendly):
        self.x = -2000 if friendly else 4800    # [m]
        self.y = 0                              # [m]
        self.v0 = 800  # Initial speed [m/sec]. only a placeholder for hostile (updated randomly with each rocket fired)
        self.ang = 0  # Turret angle [deg]. only a placeholder for hostile (updated randomly with each rocket fired)

    def update_angle_left(self, world, ang_vel):
        self.ang = self.ang - ang_vel * world.dt
        if self.ang < -90:
            world.step_reward += world.reward_illogical_action
            self.ang = -90

    def update_angle_right(self, world, ang_vel):
        self.ang = self.ang + ang_vel * world.dt
        if self.ang > 90:
            world.step_reward += world.reward_illogical_action
            self.ang = 90

    def update_angle_random(self):
        self.ang = -88 + np.random.rand() * 68  # [deg]

    def update_v0_random(self):
        self.v0 = 700 + np.random.rand() * 300  # [m/sec]


class HostileTurret:

    def __init__(self):
        self.turret = Turret(friendly=False)

    def update_turret(self):
        self.turret.update_angle_random()
        self.turret.update_v0_random()
        x = 1


class FriendlyTurret:

    def __init__(self, assessment=False):
        self.turret = Turret(friendly=True)

        self.ang_vel = 30           # Turret angular speed [deg/sec]
        self.prox_radius = 150      # detonation proximity radius [m]
        if assessment:
            self.reload_time = 0
        else:
            self.reload_time = 1.5      # [sec]

        self.last_shot_time = -3    # [sec]

    def reset(self):
        self.turret.ang = 0
        self.last_shot_time = -3

    def update_turret(self, action_button, world):
        fire_interceptor = False

        if action_button == ACTION_LEFT:
            self.turret.update_angle_left(world, self.ang_vel)

        elif action_button == ACTION_RIGHT:
            self.turret.update_angle_right(world, self.ang_vel)

        elif action_button == ACTION_FIRE:
            if (world.time - self.last_shot_time) > self.reload_time:
                self.last_shot_time = world.time  # [sec]
                fire_interceptor = True

        return fire_interceptor


class Rocket:

    def __init__(self, index, current_time_step, turret, other_turret=None, rocket_data_when_fired=None):
        self.total_list_index = index  # return it every step
        self.appearance_time_step = current_time_step

        self.fire_ang = turret.ang
        if rocket_data_when_fired is not None:
            self.rocket_data_when_fired = rocket_data_when_fired

        self.collision_time_step = -1  # disappearance_time_step
        self.collision_classification = -1
        # intercepting interceptor's index (for rockets) \ intercepted rocket's index (for interceptors):
        self.colliding_rocket_index = -1  # colliding rocket total_list_index

        self.x = turret.x  # [m]
        self.y = turret.y  # [m]
        self.vx = turret.v0 * np.sin(np.deg2rad(turret.ang))
        self.vy = turret.v0 * np.cos(np.deg2rad(turret.ang))

        self.x_other = other_turret.x if other_turret is not None else None
        self.y_other = other_turret.y if other_turret is not None else None

        self.dist = None
        self.ang = None

        self.data = []

    def update_rocket(self, world):
        v_loss = (self.vx ** 2 + self.vy ** 2) * world.fric * world.dt
        self.vx = self.vx * (1 - v_loss)
        self.vy = self.vy * (1 - v_loss) - world.g * world.dt
        dx = self.vx * world.dt
        dy = self.vy * world.dt
        self.x += dx
        self.y += dy

        if self.x_other is not None:
            delta_x = self.x - self.x_other
            delta_y = self.y - self.y_other
            self.dist = math.sqrt(math.pow(delta_x, 2) + math.pow(delta_y, 2))
            # math.degrees() - from theta do degrees.
            # theta is measured counter-clockwise from the +x axis. we need clockwise from the +y axis
            # self.ang = math.degrees(math.atan2(-delta_y, delta_x)) + 90  # range: -90 to 270
            self.ang = math.degrees(math.atan2(delta_y, -delta_x)) - 90  # range: -270 to 90
            self.data.append((self.x, self.y, dx, dy, self.dist, self.ang))
            # print('Ang: ' + str(self.ang) + ' ' + str(another_ang))
        else:
            self.data.append((self.x, self.y, dx, dy))

    def collision(self, current_time_step, collision_classification, colliding_rocket=None):
        self.collision_time_step = current_time_step
        self.collision_classification = collision_classification
        if colliding_rocket is not None:  # COLLISION_TYPE_ROCKET
            self.colliding_rocket_index = colliding_rocket.total_list_index


class City:

    def __init__(self, x1, x2):
        self.x = np.random.randint(x1, x2)  # [m]
        self.width = 800  # [m]


class Game:

    def __init__(self, single_rocket_restriction=False, assessment=False, visualize=False):
        self.ep_max_steps = 1000
        self.single_rocket_restriction = single_rocket_restriction
        self.assessment = assessment

        self.world = World()
        self.turret_friendly = FriendlyTurret(self.assessment)     # mine
        self.turret_hostile = HostileTurret()       # mine

        self.cities = []
        self.add_cities()

        self.rockets = []
        self.interceptors = []

        self.total_rockets = []         # mine
        self.total_interceptors = []    # mine

        self.first_rocket_fired = False

    def add_cities(self):
        self.cities.append(City(-self.world.width * 0.50 + 400, -self.world.width * 0.25 - 400))
        self.cities.append(City(-self.world.width * 0.25 + 400, -400))

    def reset(self):
        self.world.reset()
        self.turret_friendly = FriendlyTurret(self.assessment)
        self.turret_hostile = HostileTurret()

        self.cities.clear()
        self.add_cities()

        self.rockets.clear()
        self.interceptors.clear()

        self.total_rockets.clear()
        self.total_interceptors.clear()

        self.first_rocket_fired = False

        observation, step_reward, done = self.step(ACTION_RIGHT)

        return observation

    def step(self, action_button):
        # mine
        self.world.step_reward = 0

        self.step_world(action_button)

        # check rockets interactions
        self.check_interception()
        self.check_ground_hit()

        self.world.game_score += self.world.step_reward

        observation = self.get_observation()

        # check game end
        if self.single_rocket_restriction:
            done = self.first_rocket_fired and not self.rockets
        else:
            done = self.world.time_step_index == self.ep_max_steps - 1

        return observation, self.world.step_reward, done

    def step_world(self, action_button):
        self.world.time += self.world.dt
        self.world.time_step_index += 1

        if self.single_rocket_restriction:
            self.world.step_reward += self.world.reward_time_step

        if not self.rockets or not self.single_rocket_restriction:
            if np.random.rand() < self.world.rocket_prob * self.world.dt:
                self.create_rocket()

        for r in self.rockets:
            r.update_rocket(self.world)

        for i in self.interceptors:
            i.update_rocket(self.world)
            if i.y < 0 or np.abs(i.x) > self.world.width / 2:
                i.collision(self.world.time_step_index, COLLISION_TYPE_GROUND if i.y < 0 else -1)
                self.interceptors.remove(i)

        if self.turret_friendly.update_turret(action_button, self.world):
            self.create_interceptor()

    def create_rocket(self):
        if not self.first_rocket_fired:
            self.first_rocket_fired = True
        self.turret_hostile.update_turret()
        rocket = Rocket(len(self.total_rockets), self.world.time_step_index, self.turret_hostile.turret,
                        other_turret=self.turret_friendly.turret)
        self.rockets.append(rocket)
        self.total_rockets.append(rocket)

    def create_interceptor(self):
        interceptor = Rocket(len(self.total_interceptors), self.world.time_step_index, self.turret_friendly.turret,
                             rocket_data_when_fired=self.total_rockets[0].data[-2])
        self.interceptors.append(interceptor)
        self.total_interceptors.append(interceptor)
        self.world.step_reward += self.world.reward_fire

    def check_interception(self):
        for i in self.interceptors:
            for r in self.rockets:
                if ((r.x - i.x) ** 2 + (r.y - i.y) ** 2) ** 0.5 < self.turret_friendly.prox_radius:
                    r.collision(self.world.time_step_index, COLLISION_TYPE_ROCKET, i)
                    i.collision(self.world.time_step_index, COLLISION_TYPE_ROCKET, r)
                    self.rockets.remove(r)
                    if i in self.interceptors:
                        self.interceptors.remove(i)
                    self.world.step_reward += self.world.reward_intercept

    def check_ground_hit(self):
        for r in self.rockets:
            if r.y < 0:
                city_hit = False
                for c in self.cities:
                    if np.abs(r.x - c.x) < c.width:
                        city_hit = True
                r.collision(self.world.time_step_index, COLLISION_TYPE_CITY if city_hit else COLLISION_TYPE_GROUND)
                self.rockets.remove(r)
                self.world.step_reward += (self.world.reward_city if city_hit else self.world.reward_open)

    def get_observation(self):
        r_locs = np.zeros(shape=(len(self.rockets), 2))
        r_locs_total_list_index = np.zeros(shape=len(self.rockets), dtype=np.int16)
        for i in range(len(self.rockets)):
            r_locs_total_list_index[i] = self.rockets[i].total_list_index
            r_locs[i, :] = [self.rockets[i].x, self.rockets[i].y]

        i_locs = np.zeros(shape=(len(self.interceptors), 2))
        i_locs_total_list_index = np.zeros(shape=len(self.interceptors), dtype=np.int16)
        for i in range(len(self.interceptors)):
            i_locs_total_list_index[i] = self.interceptors[i].total_list_index
            i_locs[i, :] = [self.interceptors[i].x, self.interceptors[i].y]

        c_locs = np.zeros(shape=(len(self.cities), 2))
        for i in range(len(self.cities)):
            c_locs[i, :] = [self.cities[i].x, self.cities[i].width]

        return r_locs, i_locs, c_locs, self.turret_friendly.turret.ang, r_locs_total_list_index, i_locs_total_list_index

    def get_score(self):
        return self.world.game_score

    def get_state_single_rocket(self):
        if self.total_rockets:
            return np.array([*self.total_rockets[0].data[-1], self.turret_friendly.turret.ang])
        else:
            if self.assessment:
                return self.turret_friendly.turret.ang,
            else:
                return None

    def get_cities_data(self):
        return np.array([self.cities[0].x - self.cities[0].width / 2,
                         self.cities[0].x + self.cities[0].width / 2,
                         self.cities[1].x - self.cities[1].width / 2,
                         self.cities[1].x + self.cities[1].width / 2])
