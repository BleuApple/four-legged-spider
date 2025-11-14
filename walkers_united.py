import mujoco
import mujoco.viewer
import numpy as np
import time
from robokinemath import *

model = mujoco.MjModel.from_xml_path("Spider_v1.xml")
data = mujoco.MjData(model)

key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "start_pose")
mujoco.mj_resetDataKeyframe(model, data, key_id)

indToCtrl = {1:1, 2:3, 3:2, 4:4}

target_qpos_1 = np.array([
    0.0,  0.64411, -1.86921,
    -0.4,  0.64411, -1.86921,
    0.4,  0.64411, -1.86921,
    0.0,  0.64411, -1.86921
])

FR_hip_up = np.array([
    0.0,  1, 0.0,
    0.0,  0.0, 0.0,
    0.0,  0.0, 0.0,
    0.0,  0.0, 0.0
])

FR_hip_rot = np.array([
    1.0,  0.0, 0.0,
    0.0,  0.0, 0.0,
    0.0,  0.0, 0.0,
    0.0,  0.0, 0.0
])

span_len = 2
height = 1.2
femur_len = 3
tibia_len = 4
base_len = 1
step_len = 1.5
spy = SpiderRobotMath(femur_len, tibia_len, base_len, height, step_len, 0.6, span_len)
def_params = spy.leg_inv_kinem(span_len, 0, -1* height)
print(def_params, spy.leg_fwd_kinem(def_params[0], def_params[1], def_params[2]))

target_qpos = np.array([
             def_params[0], def_params[1], def_params[2],
             def_params[0], def_params[1], def_params[2],
             def_params[0], def_params[1], def_params[2],
             def_params[0], def_params[1], def_params[2],
             ])

target_qpos_2 = np.array([
                0,  1.15, -2.55,
                0,  1.15, -2.55,
                0,  1.15, -2.55,
                0,  1.15, -2.55
                ])
target_qpos_ = np.array([
             0, 0, 0,
             1, 0, 0, 0,
             def_params[0], def_params[1], def_params[2],
             def_params[0], def_params[1], def_params[2],
             def_params[0], def_params[1], def_params[2],
             def_params[0], def_params[1], def_params[2],
             ])


run_mode = 4
iterara = 0
leg_cycle = 15
le_direction = [1, 1]
normalised = np.array(le_direction)/np.linalg.norm(np.array(le_direction))
print("Normalised target direction", normalised)

with mujoco.viewer.launch_passive(model, data) as viewer:
    print(data.qpos)
    prev_leg_ind = -1
    init_time = 0
    curr_qpos = target_qpos
    prev_seconds_passed = time.time()//1
    start_time = time.time()
    phase = -1
    while viewer.is_running():
        if time.time() - start_time < 3:
            data.ctrl[:] = target_qpos_2
        elif run_mode == 1:
            data.ctrl[:] = target_qpos + 0.1 * (np.sin(2 * np.pi * 0.5 * time.time()))**2
        elif run_mode == 2:
            data.ctrl[:] = target_qpos + \
                FR_hip_up * 0.3 * (np.sin(2 * np.pi * 0.5 * time.time()))**2 + \
                FR_hip_rot * 0.3 * np.sin(2 * np.pi * 0.2 * time.time())
            if (time.time()//1 - time.time())**2 < 0.025 and iterara < 5:
                print(data.qpos)
                iterara += 1
        elif run_mode == 3:
            seconds_passed = (time.time()-start_time)//1



        mujoco.mj_step(model, data)
        viewer.sync()
