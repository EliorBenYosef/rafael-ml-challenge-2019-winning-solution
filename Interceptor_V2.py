# -*- coding: utf-8 -*-
"""
Created on 27/5/2019
Interceptor V2
@author: I. 
This ML challenge was created by SAMLA (National Electronic Warfare, Cyber & Intelligence Research Center) - a national research lab at Rafael http://portal.rafael.co.il/mlchallenge2019/Documents/samla.html



The goal of the game:
    Getting highest score in 100 games each running for 1000 steps.
    The player have access to 3 functions:

        Init(): This function initializes the game. It should be called before each game.

        Game_step(action_button): This function performs an action as described:
            action_button = 0: Change turret angle one step left
            action_button = 1: Do nothing in the current game step
            action_button = 2: Change turret angle one step right
            action_button = 3: Fire

            This function returns several variables:
                r_locs: Location of each rocket in the game (x,y)
                i_locs: Location of each interceptor in the game (x,y)
                c_locs: Location of each city in the game (x, width)
                ang: Turret angle
                score: Current player score

        Draw(): This function displays current game state (slows down your program. Not a must)

    Score is as follows:
        Rocket hits city: -15 points
        Rocket hits open field: -1 point
        Firing an interceptor: -1 point
        Intercepting a rocket: +4 points

In order to play, do the following:
***********************************

from Interceptor_V2 import Init, Draw, Game_step

Init()
for stp in range(1000):
    action_button = *** Insert your logic here: 0,1,2 or 3 ***
    r_locs, i_locs, c_locs, ang, score = Game_step(action_button)
    Draw()

*************************************

Don't forget to play by the rules described in the website.
"""

import numpy as np
import matplotlib.pyplot as plt


class World():
    width = 10000  # [m]
    height = 4000  # [m]
    dt = 0.2  # [sec]
    time = 0  # [sec]
    score = 0
    reward_city = -15
    reward_open = -1
    reward_fire = -1
    reward_intercept = 4
    g = 9.8  # Gravity [m/sec**2]
    fric = 5e-7  # Air friction [Units of Science]
    rocket_prob = 1  # expected rockets per sec


class Turret():
    x = -2000  # [m]
    y = 0  # [m]
    x_hostile = 4800
    y_hostile = 0
    ang_vel = 30  # Turret angular speed [deg/sec]
    ang = 0  # Turret angle [deg]
    v0 = 800  # Initial speed [m/sec]
    prox_radius = 150  # detonation proximity radius [m]
    reload_time = 1.5  # [sec]
    last_shot_time = -3  # [sec]

    def update(self, action_button):
        if action_button == 0:
            self.ang = self.ang - self.ang_vel * world.dt
            if self.ang < -90: self.ang = -90

        if action_button == 1:
            pass

        if action_button == 2:
            self.ang = self.ang + self.ang_vel * world.dt
            if self.ang > 90: self.ang = 90

        if action_button == 3:
            if world.time - self.last_shot_time > self.reload_time:
                Interceptor()
                self.last_shot_time = world.time  # [sec]


class Interceptor():
    def __init__(self):
        self.x = turret.x
        self.y = turret.y
        self.vx = turret.v0 * np.sin(np.deg2rad(turret.ang))
        self.vy = turret.v0 * np.cos(np.deg2rad(turret.ang))
        world.score = world.score + world.reward_fire
        interceptor_list.append(self)

    def update(self):
        self.v_loss = (self.vx ** 2 + self.vy ** 2) * world.fric * world.dt
        self.vx = self.vx * (1 - self.v_loss)
        self.vy = self.vy * (1 - self.v_loss) - world.g * world.dt
        self.x = self.x + self.vx * world.dt
        self.y = self.y + self.vy * world.dt
        if self.y < 0:
            Explosion(self.x, self.y)
            interceptor_list.remove(self)
        if np.abs(self.x) > world.width / 2:
            interceptor_list.remove(self)


class Rocket():
    def __init__(self, world):
        self.x = turret.x_hostile  # [m]
        self.y = turret.y_hostile  # [m]
        self.v0 = 700 + np.random.rand() * 300  # [m/sec]
        self.ang = -88 + np.random.rand() * 68  # [deg]
        self.vx = self.v0 * np.sin(np.deg2rad(self.ang))
        self.vy = self.v0 * np.cos(np.deg2rad(self.ang))
        rocket_list.append(self)

    def update(self):
        self.v_loss = (self.vx ** 2 + self.vy ** 2) * world.fric * world.dt
        self.vx = self.vx * (1 - self.v_loss)
        self.vy = self.vy * (1 - self.v_loss) - world.g * world.dt
        self.x = self.x + self.vx * world.dt
        self.y = self.y + self.vy * world.dt


class City():
    def __init__(self, x1, x2, width):
        self.x = np.random.randint(x1, x2)  # [m]
        self.width = width  # [m]
        city_list.append(self)
        self.img = np.zeros((200, 800))
        for b in range(60):
            h = np.random.randint(30, 180)
            w = np.random.randint(30, 80)
            x = np.random.randint(1, 700)
            self.img[0:h, x:x + w] = np.random.rand()
        self.img = np.flipud(self.img)


class Explosion():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = 500
        self.duration = 0.4  # [sec]
        self.verts1 = (np.random.rand(30, 2) - 0.5) * self.size
        self.verts2 = (np.random.rand(20, 2) - 0.5) * self.size / 2
        self.verts1[:, 0] = self.verts1[:, 0] + x
        self.verts1[:, 1] = self.verts1[:, 1] + y
        self.verts2[:, 0] = self.verts2[:, 0] + x
        self.verts2[:, 1] = self.verts2[:, 1] + y
        self.hit_time = world.time
        explosion_list.append(self)

    def update(self):
        if world.time - self.hit_time > self.duration:
            explosion_list.remove(self)


def Check_interception():
    for intr in interceptor_list:
        for r in rocket_list:
            if ((r.x - intr.x) ** 2 + (r.y - intr.y) ** 2) ** 0.5 < turret.prox_radius:
                rocket_list.remove(r)
                Explosion(intr.x, intr.y)
                if intr in interceptor_list: interceptor_list.remove(intr)
                world.score = world.score + world.reward_intercept


def Check_ground_hit():
    for r in rocket_list:
        if r.y < 0:
            city_hit = False
            for c in city_list:
                if np.abs(r.x - c.x) < c.width:
                    city_hit = True
            if city_hit == True:
                world.score = world.score + world.reward_city
            else:
                world.score = world.score + world.reward_open
            Explosion(r.x, r.y)
            rocket_list.remove(r)


def Draw():
    plt.cla()
    plt.rcParams['axes.facecolor'] = 'black'
    for r in rocket_list:
        plt.plot(r.x, r.y, '.y')
    for intr in interceptor_list:
        plt.plot(intr.x, intr.y, 'or')
        C1 = plt.Circle((intr.x, intr.y), radius=turret.prox_radius, linestyle='--', color='gray', fill=False)
        ax = plt.gca()
        ax.add_artist(C1)
    for c in city_list:
        plt.imshow(c.img, extent=[c.x - c.width / 2, c.x + c.width / 2, 0, c.img.shape[0]])
        plt.set_cmap('bone')
    for e in explosion_list:
        P1 = plt.Polygon(e.verts1, True, color='yellow')
        P2 = plt.Polygon(e.verts2, True, color='red')
        ax = plt.gca()
        ax.add_artist(P1)
        ax.add_artist(P2)
    plt.plot(turret.x, turret.y, 'oc', markersize=12)
    plt.plot([turret.x, turret.x + 100 * np.sin(np.deg2rad(turret.ang))],
             [turret.y, turret.y + 100 * np.cos(np.deg2rad(turret.ang))], 'c', linewidth=3)
    plt.plot(turret.x_hostile, turret.y_hostile, 'or', markersize=12)
    plt.axes().set_aspect('equal')
    plt.axis([-world.width / 2, world.width / 2, 0, world.height])
    plt.title('Score: ' + str(world.score))
    plt.draw()
    plt.pause(0.001)


def Init():
    global world, turret, rocket_list, interceptor_list, city_list, explosion_list
    world = World()
    rocket_list = []
    interceptor_list = []
    turret = Turret()
    city_list = []
    explosion_list = []
    City(-world.width * 0.5 + 400, -world.width * 0.25 - 400, 800)
    City(-world.width * 0.25 + 400, -400, 800)
    plt.rcParams['axes.facecolor'] = 'black'


def Game_step(action_button):
    world.time = world.time + world.dt

    if np.random.rand() < world.rocket_prob * world.dt:
        Rocket(world)

    for r in rocket_list:
        r.update()

    for intr in interceptor_list:
        intr.update()

    for e in explosion_list:
        e.update()

    turret.update(action_button)
    Check_interception()
    Check_ground_hit()

    r_locs = np.zeros(shape=(len(rocket_list), 2))
    for ind in range(len(rocket_list)):
        r_locs[ind, :] = [rocket_list[ind].x, rocket_list[ind].y]

    i_locs = np.zeros(shape=(len(interceptor_list), 2))
    for ind in range(len(interceptor_list)):
        i_locs[ind, :] = [interceptor_list[ind].x, interceptor_list[ind].y]

    c_locs = np.zeros(shape=(len(city_list), 2))
    for ind in range(len(city_list)):
        c_locs[ind, :] = [city_list[ind].x, city_list[ind].width]

    return r_locs, i_locs, c_locs, turret.ang, world.score
