# gym==0.21.0
import psutil
import csv
import os
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from codecarbon import EmissionsTracker

tracker = EmissionsTracker()
tracker.start()

class CartPoleEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02
        self.kinematics_integrator = "euler"
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self._validate_action(action)
        x, x_dot, theta, theta_dot = self.state
        force = self._get_force(action)
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (
            force + self.polemass_length * theta_dot ** 2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        self._integrate_state(x_dot, xacc, theta_dot, thetaacc)
        self.state = (x, x_dot, theta, theta_dot)
        done = self._is_done(x, theta)
        reward = self._get_reward(done)
        return np.array(self.state, dtype=np.float32), reward, done, {}

    def _validate_action(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

    def _get_force(self, action):
        return self.force_mag if action == 1 else -self.force_mag

    def _integrate_state(self, x_dot, xacc, theta_dot, thetaacc):
        nonlocal x, theta, x_dot, theta_dot
        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

    def _is_done(self, x, theta):
        return (
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

    def _get_reward(self, done):
        if not done:
            return 1.0
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            return 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            return 0.0

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32)

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400
        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0
        if self.viewer is None:
            self._init_viewer(screen_width, screen_height)
        if self.state is None:
            return None
        self._update_pole_vertex()
        self._update_cart_and_pole(scale, screen_width, carty)
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def _init_viewer(self, screen_width, screen_height):
        from gym.envs.classic_control import rendering

        self.viewer = rendering.Viewer(screen_width, screen_height)
        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        self.carttrans = rendering.Transform()
        cart.add_attr(self.carttrans)
        self.viewer.add_geom(cart)
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )
        pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        pole.set_color(0.8, 0.6, 0.4)
        self.poletrans = rendering.Transform(translation=(0, axleoffset))
        pole.add_attr(self.poletrans)
        pole.add_attr(self.carttrans)
        self.viewer.add_geom(pole)
        self.axle = rendering.make_circle(polewidth / 2)
        self.axle.add_attr(self.poletrans)
        self.axle.add_attr(self.carttrans)
        self.axle.set_color(0.5, 0.5, 0.8)
        self.viewer.add_geom(self.axle)
        self.track = rendering.Line((0, carty), (screen_width, carty))
        self.track.set_color(0, 0, 0)
        self.viewer.add_geom(self.track)
        self._pole_geom = pole

    def _update_pole_vertex(self):
        polewidth = 10.0
        polelen = self.scale * (2 * self.length)
        pole = self._pole_geom
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

    def _update_cart_and_pole(self, scale, screen_width, carty):
        x = self.state
        cartx = x[0] * scale + screen_width / 2.0
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

class myCartPoleEnv(CartPoleEnv):
    def __init__(self):
        super().__init__()
        self.episode_number = 0
        self.test_gen = []
        current_directory = os.getcwd()
        folder_path = os.path.join(current_directory, "testing")
        files = os.listdir(folder_path)
        txt_files = [file for file in files if file.endswith(".txt")][-1]
        file_path = os.path.join(folder_path, txt_files)
        file = open(str(file_path))
        content = file.readlines()
        for line in content:
            list_string = line.strip().split("#")
            list_float = [float(i) for i in list_string]
            self.test_gen.append(np.array(list_float))

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        self.modified_state = self.test_gen[self.episode_number]
        self.episode_number += 1
        return np.array(self.modified_state, dtype=np.float32)
