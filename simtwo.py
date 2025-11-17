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


run_mode = 3
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
            seconds_passed = (time.time() - start_time)//1
            curr_leg_ind = int(((seconds_passed % (leg_cycle*4)) // leg_cycle) + 1)
            if prev_leg_ind != curr_leg_ind:
                phase = 1
                print("Current leg index is:",curr_leg_ind, time.time(), seconds_passed)

                curr_qpos = np.copy(data.ctrl)
                print("current qpos is", curr_qpos)
                init_time = time.time() # init time is per leg
                
                final, init = spy.stable_centre_coords(data.qpos, curr_leg_ind)
                deltavec_wt = (final - init).flatten()

                prev_leg_ind = curr_leg_ind
                print("Delta vec for wt shift is", deltavec_wt)
                target_ctrl = spy.in_body_inv_kinem(deltavec_wt[0], deltavec_wt[1], deltavec_wt[2], np.copy(data.qpos))
                
                for i in range(4):
                    legsss = ["FR", "BR", "FL", "BL"]
                    print(legsss[i], spy.leg_fwd_kinem(target_ctrl[i][0], target_ctrl[i][1], target_ctrl[i][2]))

            if prev_seconds_passed != seconds_passed and False: 
                print(np.array(target_ctrl))
                print("delta addn is ", ((np.array(target_ctrl).flatten() - np.array(curr_qpos).flatten()) * (time.time() - (time.time()//30) * 30))/30, "at time", time.time())
            time.sleep(0.01)
            time_lapse = time.time() - init_time
            

            change = deltavec_wt * time_lapse / leg_cycle
            target_change = spy.in_body_inv_kinem(change[0], change[1], change[2], target_qpos_)

            data.ctrl[:] = np.array(target_change).flatten()
            if prev_seconds_passed != seconds_passed: print("updating to ", change,target_change)


            #data.ctrl[:] = curr_qpos + (np.array(target_ctrl).flatten() - np.array(curr_qpos)) * (time_lapse) / 30
            #if prev_seconds_passed != seconds_passed: print("updating to ", curr_qpos + (np.array(target_ctrl).flatten() - np.array(curr_qpos)) * (time_lapse) / 30)
            #print((time.time() - (time.time()//30)*30))
            prev_seconds_passed = seconds_passed
        
        elif run_mode == 4:
            seconds_passed = (time.time()-start_time)//1
            curr_leg_ind = int(((seconds_passed % (leg_cycle*4)) // leg_cycle) + 1)

            if prev_leg_ind != curr_leg_ind:
                print("\n\n\nLEG CHANGED\n\n\n")
                print("\n\n\n\nPHASE CHANGEE\n\n\n\n")
                if phase == 3: 
                    """target_qpos_ = np.array([0,0,0,
                                             1,0,0,0,
                                             target_change[0][0],  target_change[0][1], target_change[0][2],
                                             target_change[1][0],  target_change[1][1], target_change[1][2],
                                             target_change[2][0],  target_change[2][1], target_change[2][2],
                                             target_change[3][0],  target_change[3][1], target_change[3][2],
                                             ])"""
                phase = 1
                print("Current leg index is:",curr_leg_ind, time.time(), seconds_passed)

                curr_qpos = np.copy(data.ctrl)
                print("current qpos is", curr_qpos, "phase is ", phase)

                init_time = time.time()
                final, init = spy.stable_centre_coords(data.qpos, curr_leg_ind,False)
                deltavec = (final - init).flatten()

                prev_leg_ind = curr_leg_ind
                print("Delta vec is", deltavec)
                print("current qpos", init)
                print("target qpos", final)
                target_ctrl = spy.in_body_inv_kinem(deltavec[0], deltavec[1], deltavec[2], np.copy(data.qpos),False)
                im2sleepy = [0,0,0,
                             1,0,0,0,
                             target_ctrl[0][0],target_ctrl[0][1],target_ctrl[0][2],
                             target_ctrl[1][0],target_ctrl[1][1],target_ctrl[1][2],
                             target_ctrl[2][0],target_ctrl[2][1],target_ctrl[2][2],
                             target_ctrl[3][0],target_ctrl[3][1],target_ctrl[3][2],]
                print("projected qpos", spy.stable_centre_coords(im2sleepy,None,False)[1])

                print("motor vals", target_ctrl)
                
                for i in range(4):
                    legsss = ["FR", "BR", "FL", "BL"]
                    print(legsss[i], spy.leg_fwd_kinem(target_ctrl[i][0], target_ctrl[i][1], target_ctrl[i][2]))

            if prev_seconds_passed != seconds_passed and False: 
                print(np.array(target_ctrl))
                print("delta addn is ", ((np.array(target_ctrl).flatten() - np.array(curr_qpos).flatten()) * (time.time() - (time.time()//30) * 30))/30, "at time", time.time())
            time.sleep(0.01)
            time_lapse = time.time() - init_time

            if time_lapse > 2 * leg_cycle / 3:
                if phase == 2:
                    print("\n\n\n\nPHASE CHANGEE\n\n\n\n")
                    curr_qpos = np.copy(data.ctrl)
                    print("current qpos is", curr_qpos, "phase is ", phase)
                    rota = lambda x, y, z, leg_index : np.array([[np.cos(np.deg2rad(45 - 90 * leg_index)), -np.sin(np.deg2rad(45 - 90 * leg_index)), 0],
                                                     [np.sin(np.deg2rad(45 - 90 * leg_index)),  np.cos(np.deg2rad(45 - 90 * leg_index)), 0],
                                                     [                                      0,                            0,             1]])\
                                                    @ np.array([[x], [y], [z]])
                
                    ultra_target = np.copy(target_change)
                    pos_to_get = np.array([[spy.span_len], [0] , [height]]) + rota(normalised[0] - deltavec[0], normalised[1] - deltavec[0], 0, curr_leg_ind) * spy.step_length
                    pos_to_get = pos_to_get.flatten()
                    ultra_target[curr_leg_ind-1] = list(spy.leg_inv_kinem(pos_to_get[0], pos_to_get[1], pos_to_get[2]))
                    ref_target = np.copy(target_change)
                phase = 3
            elif time_lapse > leg_cycle / 3:
                if phase == 1:
                    print("\n\n\n\nPHASE CHANGEE\n\n\n\n")
                    curr_qpos = np.copy(data.ctrl)
                    print("current qpos is", curr_qpos, "phase is ", phase)
                    print("Current Leg", curr_leg_ind, "about to lift", indToCtrl[[4, 2, 1, 3][curr_leg_ind - 1]])
                    curr_lift_angle = target_change[indToCtrl[curr_leg_ind] - 1][1]
                phase = 2
            
                    
            
            if phase == 1:
                change = deltavec * time_lapse * 3 / ( leg_cycle)
                target_change = spy.in_body_inv_kinem(change[0], change[1], change[2], target_qpos_)
                _, init__ = spy.stable_centre_coords(np.copy(data.qpos))
            elif phase == 2:
                #target_change = np.copy(curr_qpos)
                target_change = [list(target_change[0]), list(target_change[1]), list(target_change[2]), list(target_change[3])]
                opp_leg_ind = [3, 4, 1, 2][curr_leg_ind - 1]
                #print("curr, opp, ctrl", curr_leg_ind, opp_leg_ind, indToCtrl[opp_leg_ind])
                #target_change[indToCtrl[opp_leg_ind]][1] = curr_qpos[indToCtrl[opp_leg_ind]*3 - 1] + \
                #    (2.7 - curr_qpos[indToCtrl[curr_leg_ind]*3 - 1])* (time_lapse - leg_cycle/3) * 3/( leg_cycle)

                target_change[indToCtrl[curr_leg_ind] - 1][1] = curr_lift_angle + (2 - curr_lift_angle) * (time_lapse - leg_cycle/3) * 3 / leg_cycle

            else:
                target_change = np.array(ref_target) + (np.array(ultra_target) - np.array(ref_target)) * (time_lapse - 2 * leg_cycle/3) * 3/ leg_cycle

            data.ctrl[:] = np.array(target_change).flatten()
            if prev_seconds_passed != seconds_passed: 
                print("updating to ", target_change, "seconds passed,", seconds_passed)
                if phase ==1 :
                    print("centre loc,",init__)
                if phase == 3:
                    print("reference", ref_target)
                    print("ultra", ultra_target)


            #data.ctrl[:] = curr_qpos + (np.array(target_ctrl).flatten() - np.array(curr_qpos)) * (time_lapse) / 30
            #if prev_seconds_passed != seconds_passed: print("updating to ", curr_qpos + (np.array(target_ctrl).flatten() - np.array(curr_qpos)) * (time_lapse) / 30)
            #print((time.time() - (time.time()//30)*30))
            prev_seconds_passed = seconds_passed


        mujoco.mj_step(model, data)
        viewer.sync()
