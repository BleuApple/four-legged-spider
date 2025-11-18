import mujoco
import mujoco.viewer
import numpy as np
import time
from robokinemath import *

model = mujoco.MjModel.from_xml_path("Spider_v1.xml")
data = mujoco.MjData(model)

key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "start_pose")
mujoco.mj_resetDataKeyframe(model, data, key_id)


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
def_params = spy.leg_inv_kinem(span_len, 0,  height)
print(def_params, spy.leg_fwd_kinem(def_params[0], def_params[1], def_params[2]))


target_qpos_cut = np.array([
             def_params[0], def_params[1], def_params[2],
             def_params[0], def_params[1], def_params[2],
             def_params[0], def_params[1], def_params[2],
             def_params[0], def_params[1], def_params[2],
             ])

target_qpos_cut_2 = np.array([
                0,  1.15, -2.55,
                0,  1.15, -2.55,
                0,  1.15, -2.55,
                0,  1.15, -2.55
                ])
target_qpos_cut_ = np.array([
             0, 0, 0,
             1, 0, 0, 0,
             def_params[0], def_params[1], def_params[2],
             def_params[0], def_params[1], def_params[2],
             def_params[0], def_params[1], def_params[2],
             def_params[0], def_params[1], def_params[2],
             ])

data.qpos[:] = target_qpos_cut_

run_mode = 5
iterara = 0
leg_cycle = 15
phase_cycle = 1
le_direction = [1, 1]
targeted_lift_per_leg = 2
target_real_eps = 0.1
msg_wait_time = 1
normalised = np.array(le_direction)/np.linalg.norm(np.array(le_direction))
delta_vec_accumulated = np.zeros(3)
print("Normalised target direction", normalised)
print("Leg fwd kinem")
print(spy.leg_fwd_kinem(target_qpos_cut_2[0], target_qpos_cut_2[1], target_qpos_cut_2[2]))

# Trying to introduce artificial leg 5 which has only one phase which is to centre itself

with mujoco.viewer.launch_passive(model, data) as viewer:
    print(data.qpos)
    prev_leg_ind = -1
    init_time = 0
    curr_ctrl = target_qpos_cut
    prev_seconds_passed = time.time()//1
    start_time = time.time()
    phase = -1
    phase_time = -1
    last_msg_time = -1
    while viewer.is_running():
        if time.time() - start_time < 3:
            data.ctrl[:] = target_qpos_cut
        elif run_mode == 1:
            data.ctrl[:] = target_qpos_cut + 0.1 * (np.sin(2 * np.pi * 0.5 * time.time()))**2
        elif run_mode == 2:
            data.ctrl[:] = target_qpos_cut + \
                FR_hip_up * 0.3 * (np.sin(2 * np.pi * 0.5 * time.time()))**2 + \
                FR_hip_rot * 0.3 * np.sin(2 * np.pi * 0.2 * time.time())
            if (time.time()//1 - time.time())**2 < 0.025 and iterara < 5:
                print(data.qpos)
                iterara += 1       
        elif run_mode == 5:
            seconds_passed = (time.time()-start_time)//1
            time.sleep(0.02)
            time_lapse = time.time() - start_time

            #Initialising first phase
            if phase == -1:
                phase = 1
                leg_init_time = time.time()
                curr_leg_ind = 1
                print("\n\nBeginning with LEG",curr_leg_ind,"\n\n")
                phase_init_time = time.time()
                print("\nStarting phase 1\n")
                curr_ctrl = np.copy(data.ctrl)

                init_qpos_ph1 = np.copy(data.qpos)

                fin_body_cen_ph1, init_body_cen_ph1 = spy.stable_centre_coords(data.qpos, curr_leg_ind, to_print=True)
                #deltavec_ph1 = (fin_body_cen_ph1 - init_body_cen_ph1).flatten()
                deltavec_ph1 = fin_body_cen_ph1.flatten()
                delta_vec_accumulated += deltavec_ph1

                target_ctrl_ph1 = spy.in_body_inv_kinem(deltavec_ph1[0], deltavec_ph1[1], deltavec_ph1[2], np.copy(data.qpos), False)
                target_ctrl_ph1_extended = np.hstack(([0,0,0,1,0,0,0], np.array(target_ctrl_ph1).flatten()))

                print("Delta vec is", deltavec_ph1, "current body pos", init_body_cen_ph1, "target body pos", fin_body_cen_ph1, \
                      "projected body pos", spy.stable_centre_coords(target_ctrl_ph1_extended,None,False)[1],
                      "target ctrl", target_ctrl_ph1)
                for i in range(4):
                    legsss = ["FR", "FL", "BL", "BR"]
                    print(legsss[i], spy.leg_fwd_kinem(target_ctrl_ph1[i][0], target_ctrl_ph1[i][1], target_ctrl_ph1[i][2]))

            leg_lapse = time.time() - leg_init_time
            phase_lapse = time.time() - phase_init_time    

            

            if phase_lapse < phase_cycle:
                if phase == 1:
                    change = deltavec_ph1 * phase_lapse / phase_cycle
                    

                    #IGNORE THIS IF FOR NOW (TESTING)
                    if ((time.time() - last_msg_time > msg_wait_time) and False):
                        print("Start Target")
                        target_change = spy.in_body_inv_kinem(change[0], change[1], 0, data.qpos, to_print = True) # TARGET_QPOS_CUT_ should be replaced with np.qpos(data.qpos)
                        print("Start Pseudo")
                        pseudo_change = spy.in_body_inv_kinem(change[0], change[1], 0, np.copy(data.qpos), to_print = True)
                        last_msg_time = time.time()
                        print("Target_qpos_cut is")
                        bleppindo = True
                        if bleppindo: 
                            print(np.round(target_qpos_cut_[ 0: 3], decimals=4))
                            print(np.round(target_qpos_cut_[ 3: 7], decimals=4))
                            print(np.round(target_qpos_cut_[ 7:10], decimals=4))
                            print(np.round(target_qpos_cut_[10:13], decimals=4))
                            print(np.round(target_qpos_cut_[13:16], decimals=4))
                            print(np.round(target_qpos_cut_[16:19], decimals=4))   
                        else:
                            print(target_change)
                        
                        print("data.qpos is")
                        if bleppindo:
                            print(np.round(data.qpos[ 0: 3], decimals=4))
                            print(np.round(data.qpos[ 3: 7], decimals=4))
                            print(np.round(data.qpos[ 7:10], decimals=4))
                            print(np.round(data.qpos[10:13], decimals=4))
                            print(np.round(data.qpos[13:16], decimals=4))
                            print(np.round(data.qpos[16:19], decimals=4))
                        else:
                            print(pseudo_change)
                    
                    else:
                        target_change = spy.in_body_inv_kinem(change[0], change[1], 0, np.copy(data.qpos)) # TARGET_QPOS_CUT_ should be replaced with np.qpos(data.qpos)
                        pseudo_change = spy.in_body_inv_kinem(change[0], change[1], 0, np.copy(data.qpos))

                    _, init__ = spy.stable_centre_coords(np.copy(data.qpos))
                elif phase == 2:
                    target_change = np.array(target_change)
                    target_change[curr_leg_ind - 1][1] = init_lift_angle_ph2 + (targeted_lift_per_leg - init_lift_angle_ph2) * phase_lapse / phase_cycle
                elif phase == 3:
                    target_change = np.array(init_ctrl_ph3) + (np.array(target_ctrl_ph3) - np.array(init_ctrl_ph3)) * phase_lapse / phase_cycle
                #elif phase == 4:
                #    target_change = np.array(init_ctrl_ph4) + (np.array(target_qpos_cut_2) - np.array(init_ctrl_ph4)) * phase_lapse / phase_cycle
                else:
                    raise ValueError(f"How did we get the phase to be {phase}")
            else:
                if np.all(np.abs(data.qpos[7:] - np.array(target_change).flatten()) < target_real_eps):
                    print("Proceeding to next phase")

                    if phase == 1:
                        print("\nStarting Phase 2\n")
                        phase_init_time = time.time()
                        phase = 2

                        curr_ctrl = np.copy(data.ctrl)
                        print("current qpos_cut is", data.qpos[7:])
                        print("Current Leg", curr_leg_ind, "about to lift", curr_leg_ind)
                        init_lift_angle_ph2 = target_change[curr_leg_ind - 1][1]
                    
                    elif phase == 2:
                        print("\nStarting Phase 3\n")
                        phase_init_time = time.time()
                        phase = 3

                        curr_ctrl = np.copy(data.ctrl)
                        print("current qpos_cut is", data.qpos[7:])
                        rota = lambda x, y, z, leg_index : np.array([[np.cos(np.deg2rad(45 - 90 * leg_index)), -np.sin(np.deg2rad(45 - 90 * leg_index)), 0],
                                                        [np.sin(np.deg2rad(45 - 90 * leg_index)),  np.cos(np.deg2rad(45 - 90 * leg_index)), 0],
                                                        [                                      0,                            0,             1]])\
                                                        @ np.array([[x], [y], [z]])
                    
                        target_ctrl_ph3 = np.copy(target_change) # final ctrl for phase 3
                        pos_to_get_ph3 = np.array([[spy.span_len], [0] , [height]]) + rota(normalised[0] - delta_vec_accumulated[0], normalised[1] - delta_vec_accumulated[0], 0, curr_leg_ind) * spy.step_length
                        pos_to_get_ph3 = pos_to_get_ph3.flatten()
                        target_ctrl_ph3[curr_leg_ind-1] = list(spy.leg_inv_kinem(pos_to_get_ph3[0], pos_to_get_ph3[1], pos_to_get_ph3[2]))
                        print("Pos to get for phase 3", pos_to_get_ph3)
                        print("Target Projected for leg", curr_leg_ind,\
                               "is", spy.leg_fwd_kinem(target_ctrl_ph3[curr_leg_ind-1][0], target_ctrl_ph3[curr_leg_ind-1][1], target_ctrl_ph3[curr_leg_ind-1][2]))
                        init_ctrl_ph3 = np.copy(target_change) # init ctrl for phase 3

                    #elif phase == 3:
                    #    print("\nStarting Phase 4\n")
                    #    phase_init_time = time.time()
                    #    phase = 4
                    #    curr_ctrl = np.copy(data.ctrl)
                    #    print("current qpos_cut is", data.qpos[7:])
                    #    init_ctrl_ph4 = np.copy(target_change) # init ctrl for phase 4

                    elif phase == 3: # Todo -> check phase 4
                        print("\n\nBeginning with LEG",curr_leg_ind,"\n\n")
                        leg_init_time = time.time()
                        if curr_leg_ind == 1:
                            curr_leg_ind = 3
                        elif curr_leg_ind == 3:
                            curr_leg_ind = 2
                        elif curr_leg_ind == 2:
                            curr_leg_ind = 4
                        elif curr_leg_ind == 4:
                            curr_leg_ind = 1
                            _, resetter_init = spy.stable_centre_coords(data.qpos, curr_leg_ind, to_print=True)
                            delta_vec_accumulated = - np.array(resetter_init).flatten()

                        phase_init_time = time.time()
                        phase = 1
                        print("\nStarting phase 1\n")

                        curr_ctrl = np.copy(data.ctrl)
                        
                        init_qpos_ph1 = np.copy(data.qpos)


                        fin_body_cen_ph1, init_body_cen_ph1 = spy.stable_centre_coords(data.qpos, curr_leg_ind, to_print=True)
                        #deltavec_ph1 = (fin_body_cen_ph1 - init_body_cen_ph1).flatten()
                        deltavec_ph1 = fin_body_cen_ph1.flatten()
                        delta_vec_accumulated += deltavec_ph1
                        
                        target_ctrl_ph1 = spy.in_body_inv_kinem(deltavec_ph1[0], deltavec_ph1[1], deltavec_ph1[2], np.copy(data.qpos), False)
                        target_ctrl_ph1_extended = np.hstack(([0,0,0,1,0,0,0], np.array(target_ctrl_ph1).flatten()))
                        print("Delta vec is", deltavec_ph1, "current body pos", init_body_cen_ph1, "target body pos", fin_body_cen_ph1, \
                            "projected body pos", spy.stable_centre_coords(target_ctrl_ph1_extended,None,False)[1],
                            "target ctrl", target_ctrl_ph1)
                        for i in range(4):
                            legsss = ["FR", "FL", "BL", "BR"]
                            print(legsss[i], spy.leg_fwd_kinem(target_ctrl_ph1[i][0], target_ctrl_ph1[i][1], target_ctrl_ph1[i][2]))

                    else:
                        raise ValueError(f"Again, how did we get phase to be {phase}")

                else:
                     if (time.time() - last_msg_time > msg_wait_time) and False:
                         last_msg_time = time.time()
                         print(f"Waiting to reach target pos. Phase Time - {phase_lapse}, Seconds passed - {seconds_passed}")
                         print("Robot Qpos")
                         print(np.array(data.qpos[7:]).reshape(4,-1))
                         print("Targeted Ctrl")
                         print(np.array(target_change).reshape(4,-1))
                         
                data.ctrl[:] = np.array(target_change).flatten()


            

        mujoco.mj_step(model, data)
        viewer.sync()
