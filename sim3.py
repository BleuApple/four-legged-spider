#!/usr/bin/env python3
"""
walk_spider.py
Working crawl gait for MuJoCo >= 3.1 (pip package `mujoco`).
"""

import numpy as np
import math
import mujoco
import mujoco.viewer
import time

# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------
def project_point_into_triangle_with_margin(A, B, C, P, eps=1e-1):
    def shift_vertex(V, Vprev, Vnext):
        e1 = (Vprev - V) / np.linalg.norm(Vprev - V)
        e2 = (Vnext - V) / np.linalg.norm(Vnext - V)
        bis = e1 + e2
        n = np.linalg.norm(bis)
        return V if n < 1e-9 else V + (eps * bis / n)

    A2 = shift_vertex(A, B, C)
    B2 = shift_vertex(B, C, A)
    C2 = shift_vertex(C, A, B)

    if np.linalg.norm(np.cross(B2 - A2, C2 - A2)) < 1e-9:
        A2, B2, C2 = A, B, C

    v0, v1, v2 = B2 - A2, C2 - A2, P - A2
    d00, d01 = np.dot(v0, v0), np.dot(v0, v1)
    d11, d20 = np.dot(v1, v1), np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01

    if abs(denom) < 1e-12:
        return A2

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w

    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1 - u)
    w = 1 - u - v

    return u * A2 + v * B2 + w * C2


# --------------------------------------------------------------------------
# Spider Math
# --------------------------------------------------------------------------
class SpiderRobotMath:
    def __init__(self, femur_length, tibia_length, base_length):
        self.f = femur_length
        self.t = tibia_length
        self.b = base_length

    def dh_parameter_to_homo_transform(self, d, theta, a, alpha):
        return np.array([
            [np.cos(theta), -np.sin(theta) * np.cos(alpha),  np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
            [np.sin(theta),  np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
            [0,             np.sin(alpha),                  np.cos(alpha),                 d],
            [0, 0, 0, 1]
        ])

    def leg_fwd_kinem(self, th1, th2, th3):
        T1 = self.dh_parameter_to_homo_transform(0, th1, 0, math.pi/2)
        T2 = self.dh_parameter_to_homo_transform(0, th2, self.f, 0)
        T3 = self.dh_parameter_to_homo_transform(0, th3, self.t, 0)
        T = T1 @ T2 @ T3
        v = T @ np.array([0, 0, 0, 1])
        return np.array([v[0], v[1], v[2]])

    def leg_inv_kinem(self, x, y, z):
        l = math.hypot(x, y)
        r = math.sqrt(l*l + z*z)

        if r > (self.f + self.t) - 1e-6:
            r = (self.f + self.t) - 1e-6

        theta1 = math.atan2(y, x)

        cos_phi = (r*r + self.f*self.f - self.t*self.t) / (2 * r * self.f)
        cos_phi = np.clip(cos_phi, -1, 1)
        phi = math.acos(cos_phi)

        theta2 = math.atan2(z, l) - phi
        theta3 = math.atan2(z - self.f * math.sin(theta2),
                            l - self.f * math.cos(theta2)) - theta2

        return theta1, -theta2, -theta3

    @staticmethod
    def qpos_to_leg_joints(qpos):
        base = 7
        return [qpos[base + i*3 : base + i*3 + 3] for i in range(4)]

    @staticmethod
    def hip_frame_to_body_xy(idx, vec):
        ang = math.radians(90 * idx - 45)
        R = np.array([[math.cos(ang), -math.sin(ang), 0],
                      [math.sin(ang),  math.cos(ang), 0],
                      [0,0,1]])

        offsets = [
            np.array([1,1,0]),
            np.array([-1,1,0]),
            np.array([1,-1,0]),
            np.array([-1,-1,0])
        ]
        return R @ vec + offsets[idx]

    @staticmethod
    def body_xy_to_hip_frame(idx, target):
        ang = math.radians(90 * idx - 45)
        R = np.array([[math.cos(ang), -math.sin(ang), 0],
                      [math.sin(ang),  math.cos(ang), 0],
                      [0,0,1]])

        offsets = [
            np.array([1,1,0]),
            np.array([-1,1,0]),
            np.array([1,-1,0]),
            np.array([-1,-1,0])
        ]
        return np.linalg.inv(R) @ (target - offsets[idx])

    def stable_centre_coords(self, qpos):
        joints = SpiderRobotMath.qpos_to_leg_joints(qpos)

        foot_world = []
        for i in range(4):
            local = self.leg_fwd_kinem(*joints[i])
            foot_world.append(self.hip_frame_to_body_xy(i, local))
        foot_world = np.array(foot_world)

        centre = np.mean(foot_world, axis=0)

        stable = []
        for i in range(4):
            others = [j for j in range(4) if j != i]
            tri = [foot_world[j][:2] for j in others]
            proj = project_point_into_triangle_with_margin(tri[0], tri[1], tri[2],
                                                           centre[:2], eps=0.12)
            z = np.mean([foot_world[j][2] for j in others])
            stable.append(np.array([proj[0], proj[1], z]))

        return np.array(stable), centre


# --------------------------------------------------------------------------
# Gait controller
# --------------------------------------------------------------------------
def run_crawl_gait(xml_path="Spider_v1.xml"):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    mathlib = SpiderRobotMath(1.5, 2.5, 1.0)

    leg_act = {0:0, 1:3, 2:6, 3:9}
    gait_seq = [0,3,2,1]

    print("Starting crawl gait...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for _ in range(30):
            mujoco.mj_step(model, data)

        while viewer.is_running():
            for leg in gait_seq:
                qpos = data.qpos.copy()
                legs = SpiderRobotMath.qpos_to_leg_joints(qpos)

                # compute stable com
                stable, cur = mathlib.stable_centre_coords(qpos)
                target_com = stable[leg]

                # move base
                for s in range(30):
                    a = (s+1)/30
                    data.qpos[0] = (1-a)*qpos[0] + a*target_com[0]
                    data.qpos[1] = (1-a)*qpos[1] + a*target_com[1]
                    data.qpos[2] = qpos[2]
                    mujoco.mj_forward(model, data)
                    mujoco.mj_step(model, data)
                    viewer.sync()

                # swing step
                joints = legs
                foots = []
                for i in range(4):
                    f_local = mathlib.leg_fwd_kinem(*joints[i])
                    f_body = mathlib.hip_frame_to_body_xy(i, f_local)
                    foots.append(f_body)
                foots = np.array(foots)

                start = foots[leg]
                end = start + np.array([0.5, 0.5, 0])

                for s in range(40):
                    t = s/39
                    lift = (1 - (2*(t-0.5))**2) * 0.6
                    xy = (1-t)*start[:2] + t*end[:2]
                    z = start[2] + lift
                    target = np.array([xy[0], xy[1], z])

                    hip = mathlib.body_xy_to_hip_frame(leg, target)
                    th1,th2,th3 = mathlib.leg_inv_kinem(*hip)

                    a0 = leg_act[leg]
                    data.ctrl[a0+0] = th1
                    data.ctrl[a0+1] = th2
                    data.ctrl[a0+2] = th3

                    mujoco.mj_step(model, data)
                    viewer.sync()


# --------------------------------------------------------------------------
if __name__ == "__main__":
    run_crawl_gait("Spider_v1.xml")
