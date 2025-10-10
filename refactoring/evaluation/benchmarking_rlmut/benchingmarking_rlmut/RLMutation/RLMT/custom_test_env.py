import psutil
import csv
import os
import math
import warnings
from typing import TYPE_CHECKING, Optional
import numpy as np
import sys
import gym
from gym import spaces, logger, error
from gym.utils import seeding, EzPickle
import Box2D
from Box2D.b2 import (
    edgeShape,
    circleShape,
    fixtureDef,
    polygonShape,
    revoluteJointDef,
    contactListener,
)
from codecarbon import EmissionsTracker

if TYPE_CHECKING:
    import pygame


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if (
            self.env.lander == contact.fixtureA.body
            or self.env.lander == contact.fixtureB.body
        ):
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False


class CustomLunarLander(gym.Env, EzPickle):
    FPS = 50
    SCALE = 30.0
    MAIN_ENGINE_POWER = 13.0
    SIDE_ENGINE_POWER = None
    INITIAL_RANDOM = 1000.0
    LANDER_POLY = [(-14, +17), (-17, 0), (-17, -10), (+17, -10), (+17, 0), (+14, +17)]
    LEG_AWAY = 20
    LEG_DOWN = 18
    LEG_W, LEG_H = 2, 8
    LEG_SPRING_TORQUE = 40
    SIDE_ENGINE_HEIGHT = 14.0
    SIDE_ENGINE_AWAY = 12.0
    VIEWPORT_W = 600
    VIEWPORT_H = 400
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": FPS}
    continuous = False

    def __init__(self, params):
        EzPickle.__init__(self)
        self.my_custom_params = params
        self.SIDE_ENGINE_POWER = self.my_custom_params[1]
        self.seed()
        self.viewer = None
        self.world = Box2D.b2World((0, self.my_custom_params[0]))
        self.moon = None
        self.lander = None
        self.particles = []
        self.prev_reward = None
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(8,), dtype=np.float32
        )
        if self.continuous:
            self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(4)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy_bodies(self):
        if self.moon:
            self.world.DestroyBody(self.moon)
            self.moon = None
        if self.lander:
            self.world.DestroyBody(self.lander)
            self.lander = None
        if hasattr(self, 'legs'):
            for leg in self.legs:
                self.world.DestroyBody(leg)
            self.legs = []

    def _destroy(self):
        self._destroy_bodies()
        self._clean_particles(True)
        self.world.contactListener = None

    def _create_terrain(self):
        W = self.VIEWPORT_W / self.SCALE
        H = self.VIEWPORT_H / self.SCALE
        CHUNKS = 11
        height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
        self.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
        self.helipad_y = H / 4
        height[CHUNKS // 2 - 2] = self.helipad_y
        height[CHUNKS // 2 - 1] = self.helipad_y
        height[CHUNKS // 2 + 0] = self.helipad_y
        height[CHUNKS // 2 + 1] = self.helipad_y
        height[CHUNKS // 2 + 2] = self.helipad_y
        smooth_y = [
            0.33 * (height[i - 1] + height[i + 0] + height[i + 1])
            for i in range(CHUNKS)
        ]
        self.moon = self.world.CreateStaticBody(
            shapes=edgeShape(vertices=[(0, 0), (W, 0)])
        )
        self.sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])
        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

    def _create_lander(self):
        initial_y = self.VIEWPORT_H / self.SCALE
        self.lander = self.world.CreateDynamicBody(
            position=(self.VIEWPORT_W / self.SCALE / 2, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=[(x / self.SCALE, y / self.SCALE) for x, y in self.LANDER_POLY]
                ),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,
                restitution=0.0,
            ),
        )
        self.lander.color1 = (0.5, 0.4, 0.9)
        self.lander.color2 = (0.3, 0.3, 0.5)
        self.lander.ApplyForceToCenter(
            (
                self.np_random.uniform(-self.INITIAL_RANDOM, self.INITIAL_RANDOM),
                self.np_random.uniform(-self.INITIAL_RANDOM, self.INITIAL_RANDOM),
            ),
            True,
        )

    def _create_legs(self):
        self.legs = []
        initial_y = self.VIEWPORT_H / self.SCALE
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(self.VIEWPORT_W / self.SCALE / 2 - i * self.LEG_AWAY / self.SCALE, initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(self.LEG_W / self.SCALE, self.LEG_H / self.SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001,
                ),
            )
            leg.ground_contact = False
            leg.color1 = (0.5, 0.4, 0.9)
            leg.color2 = (0.3, 0.3, 0.5)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * self.LEG_AWAY / self.SCALE, self.LEG_DOWN / self.SCALE),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=self.LEG_SPRING_TORQUE,
                motorSpeed=+0.3 * i,
            )
            if i == -1:
                rjd.lowerAngle = +0.9 - 0.5
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

    def _prepare_reset(self):
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None

    def reset(self):
        self._prepare_reset()
        self._create_terrain()
        self._create_lander()
        self._create_legs()
        self.drawlist = [self.lander] + self.legs
        return self.step(np.array([0, 0]) if self.continuous else 0)[0]

    def _create_particle(self, mass, x, y, ttl):
        p = self.world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=2 / self.SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,
                restitution=0.3,
            ),
        )
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all):
        while self.particles and (all or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    def _apply_main_engine(self, action, m_power):
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) / self.SCALE for _ in range(2)]
        ox = (
            tip[0] * (4 / self.SCALE + 2 * dispersion[0]) + side[0] * dispersion[1]
        )
        oy = -tip[1] * (4 / self.SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
        impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
        p = self._create_particle(
            3.5,
            impulse_pos[0],
            impulse_pos[1],
            m_power,
        )
        p.ApplyLinearImpulse(
            (ox * self.MAIN_ENGINE_POWER * m_power, oy * self.MAIN_ENGINE_POWER * m_power),
            impulse_pos,
            True,
        )
        self.lander.ApplyLinearImpulse(
            (-ox * self.MAIN_ENGINE_POWER * m_power, -oy * self.MAIN_ENGINE_POWER * m_power),
            impulse_pos,
            True,
        )

    def _apply_side_engine(self, action, s_power):
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) / self.SCALE for _ in range(2)]
        direction = np.sign(action[1]) if self.continuous else action - 2
        ox = tip[0] * dispersion[0] + side[0] * (
            3 * dispersion[1] + direction * self.SIDE_ENGINE_AWAY / self.SCALE
        )
        oy = -tip[1] * dispersion[0] - side[1] * (
            3 * dispersion[1] + direction * self.SIDE_ENGINE_AWAY / self.SCALE
        )
        impulse_pos = (
            self.lander.position[0] + ox - tip[0] * 17 / self.SCALE,
            self.lander.position[1] + oy + tip[1] * self.SIDE_ENGINE_HEIGHT / self.SCALE,
        )
        p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
        p.ApplyLinearImpulse(
            (ox * self.SIDE_ENGINE_POWER * s_power, oy * self.SIDE_ENGINE_POWER * s_power),
            impulse_pos,
            True,
        )
        self.lander.ApplyLinearImpulse(
            (-ox * self.SIDE_ENGINE_POWER * s_power, -oy * self.SIDE_ENGINE_POWER * s_power),
            impulse_pos,
            True,
        )

    def _calculate_state(self):
        pos = self.lander.position
        vel = self.lander.linearVelocity
        state = [
            (pos.x - self.VIEWPORT_W / self.SCALE / 2) / (self.VIEWPORT_W / self.SCALE / 2),
            (pos.y - (self.helipad_y + self.LEG_DOWN / self.SCALE)) / (self.VIEWPORT_H / self.SCALE / 2),
            vel.x * (self.VIEWPORT_W / self.SCALE / 2) / self.FPS,
            vel.y * (self.VIEWPORT_H / self.SCALE / 2) / self.FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / self.FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ]
        return np.array(state, dtype=np.float32)

    def _calculate_reward(self, state, m_power, s_power):
        reward = 0
        shaping = (
            -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
            - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
            - 100 * abs(state[4])
            + 10 * state[6]
            + 10 * state[7]
        )
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping
        reward -= m_power * 0.30
        reward -= s_power * 0.03
        return reward

    def _is_done(self, state):
        if self.game_over or abs(state[0]) >= 1.0:
            return True, -100
        if not self.lander.awake:
            return True, +100
        return False, None

    def step(self, action):
        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float32)
        else:
            assert self.action_space.contains(action), "%r (%s) invalid " % (
                action,
                type(action),
            )
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (
            not self.continuous and action == 2
        ):
            if self.continuous:
                m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5
                assert m_power >= 0.5 and m_power <= 1.0
            else:
                m_power = 1.0
            self._apply_main_engine(action, m_power)
        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (
            not self.continuous and action in [1, 3]
        ):
            if self.continuous:
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
                assert s_power >= 0.5 and s_power <= 1.0
            else:
                s_power = 1.0
            self._apply_side_engine(action, s_power)
        self.world.Step(1.0 / self.FPS, 6 * 30, 2 * 30)
        state = self._calculate_state()
        reward = self._calculate_reward(state, m_power, s_power)
        done, reward_override = self._is_done(state)
        if reward_override is not None:
            reward = reward_override
        return np.array(state, dtype=np.float32), reward, done, {}

    def _update_particle_colors(self):
        for obj in self.particles:
            obj.ttl -= 0.15
            obj.color1 = (
                max(0.2, 0.2 + obj.ttl),
                max(0.2, 0.5 * obj.ttl),
                max(0.2, 0.5 * obj.ttl),
            )
            obj.color2 = (
                max(0.2, 0.2 + obj.ttl),
                max(0.2, 0.5 * obj.ttl),
                max(0.2, 0.5 * obj.ttl),
            )

    def _draw_body(self, obj, viewer):
        for f in obj.fixtures:
            trans = f.body.transform
            if type(f.shape) is circleShape:
                t = rendering.Transform(translation=trans * f.shape.pos)
                viewer.draw_circle(
                    f.shape.radius, 20, color=obj.color1
                ).add_attr(t)
                viewer.draw_circle(
                    f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2
                ).add_attr(t)
            else:
                path = [trans * v for v in f.shape.vertices]
                viewer.draw_polygon(path, color=obj.color1)
                path.append(path[0])
                viewer.draw_polyline(path, color=obj.color2, linewidth=2)

    def _draw_helipad(self, viewer):
        for x in [self.helipad_x1, self.helipad_x2]:
            flagy1 = self.helipad_y
            flagy2 = flagy1 + 50 / self.SCALE
            viewer.draw_polyline([(x, flagy1), (x, flagy2)], color=(1, 1, 1))
            viewer.draw_polygon(
                [
                    (x, flagy2),
                    (x, flagy2 - 10 / self.SCALE),
                    (x + 25 / self.SCALE, flagy2 - 5 / self.SCALE),
                ],
                color=(0.8, 0.8, 0),
            )

    def render(self, mode="human"):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.VIEWPORT_W, self.VIEWPORT_H)
            self.viewer.set_bounds(0, self.VIEWPORT_W / self.SCALE, 0, self.VIEWPORT_H / self.SCALE)
        self._update_particle_colors()
        self._clean_particles(False)
        for p in self.sky_polys:
            self.viewer.draw_polygon(p, color=(0, 0, 0))
        for obj in self.particles + self.drawlist:
            self._draw_body(obj, self.viewer)
        self._draw_helipad(self.viewer)
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class CustomCartPole(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self, params):
        self.gravity = 9.8
        self.masscart = params[0]
        self.masspole = params[1]
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

    def _calculate_derivatives(self, state, force):
        x, x_dot, theta, theta_dot = state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (
            force + self.polemass_length * theta_dot ** 2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        return xacc, thetaacc

    def _integrate_state(self, state, xacc, thetaacc):
        x, x_dot, theta, theta_dot = state
        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        return x, x_dot, theta, theta_dot

    def _is_episode_over(self, x, theta):
        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        return done

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        force = self.force_mag if action == 1 else -self.force_mag
        xacc, thetaacc = self._calculate_derivatives(self.state, force)
        x, x_dot, theta, theta_dot = self._integrate_state(self.state, xacc, thetaacc)
        self.state = (x, x_dot, theta, theta_dot)
        done = self._is_episode_over(x, theta)
        reward = 1.0
        if not done:
            pass
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0
        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32)

    def _create_cart_pole_geometries(self):
        from gym.envs.classic_control import rendering
        screen_width = 600
        screen_height = 400
        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0
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

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self._create_cart_pole_geometries()
        if self.state is None:
            return None
        pole = self._pole_geom
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )
        pole.v = [(l, b), (l, t), (r, t), (r, b)]
        x = self.state
        cartx = x[0] * scale + screen_width / 2.0
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])
        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

tracker = EmissionsTracker(project_name=r"C:\Users\guicu\OneDrive\Documentos\prog\aise\artifact\artifacts\benchmarking_rlmut\RLMutation\RLMT\custom_test_env.py")
tracker.start()
mem_start = psutil.virtual_memory().used / (1024**2)
cpu_start = psutil.cpu_percent(interval=None)
tracker = EmissionsTracker()
tracker.start()
tracker.stop()
