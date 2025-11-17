import mujoco
import mujoco.viewer
import numpy as np
import time
from robokinemath import * # Assuming this is your SpiderRobotMath class

model = mujoco.MjModel.from_xml_path("Spider_v1.xml")
data = mujoco.MjData(model)

key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "start_pose")
mujoco.mj_resetDataKeyframe(model, data, key_id)

# --- Kinematic/Pose Definitions (Unchanged) ---
span_len = 2
height = 1.2
femur_len = 3
tibia_len = 4
base_len = 1
step_len = 1.5 # The desired foot displacement distance
spy = SpiderRobotMath(femur_len, tibia_len, base_len, height, step_len, 0.6, span_len)
def_params = spy.leg_inv_kinem(span_len, 0, -1* height)

# Target pose 2, used as the standing/base position
target_qpos_2 = np.array([
    0.0, 1.15, -2.55,
    0.0, 1.15, -2.55,
    0.0, 1.15, -2.55,
    0.0, 1.15, -2.55
])

run_mode = 3

# --- Smooth Motion Parameters ---
cycle_duration = 2.0  # Time for one full cycle (G1 lift + body trans + G2 lift + body trans)
step_duration = cycle_duration / 4.0 # Duration of each of the 4 phases
amplitude_lift = 0.4  # Max lift for the foot
amplitude_swing = 1.0 # Max rotation for the hip (forward/backward)

# Base joint positions from target_qpos_2 (Assuming 3 joints per leg)
hip_base_rot = target_qpos_2[0] # Joint 1: Rotation
hip_base_up = target_qpos_2[1]  # Joint 2: Up/Down
knee_base = target_qpos_2[2]    # Joint 3: Knee

# --- Body Control Parameters ---
# The index for the floating body's X position (forward/backward movement)
# Assuming the first three joints in the model are the floating base translation (X, Y, Z)
body_x_ctrl_idx = 0 
body_step_distance = 0.5 # How far the body translates during each translation phase

# Grouping indices for control (qpos is 0-11 for 4 legs x 3 joints)
# Group 1: Front-Right (0-2) and Back-Left (9-11)
G1_indices = [0, 1, 2, 9, 10, 11]
# Group 2: Front-Left (3-5) and Back-Right (6-8)
G2_indices = [3, 4, 5, 6, 7, 8]
# --- End Smooth Motion Parameters ---


with mujoco.viewer.launch_passive(model, data) as viewer:
    print(data.qpos)
    start_time = time.time()
    
    # Initialize control to the standing position
    data.ctrl[:] = target_qpos_2 
    data.ctrl[body_x_ctrl_idx] = data.qpos[body_x_ctrl_idx] # Initialize body X-control to current body X-position

    while viewer.is_running():
        elapsed_time = time.time() - start_time
        
        # Calculate time within the current cycle (from 0 to cycle_duration)
        t = elapsed_time % cycle_duration
        
        # --- Smooth Gait Control ---
        if run_mode == 3:
            
            # --- 1. Reset all controls to base position (important for joints not being moved) ---
            data.ctrl[:] = target_qpos_2
            data.ctrl[body_x_ctrl_idx] = data.qpos[body_x_ctrl_idx] # Keep body translation control separate
            
            # --- Body Translation Tracking ---
            # Use the initial body position as the origin for forward movement
            initial_body_x = data.qpos[body_x_ctrl_idx] # Use data.qpos as starting reference for forward motion
            
            # --- Phase Logic ---
            
            # PHASE 1: Group 1 (FR, BL) Swings Forward (t = 0 to step_duration)
            if t < step_duration:
                # Group 1 lifts and swings forward (use smooth sine profile)
                time_in_phase = t
                phase_progress = time_in_phase / step_duration # 0 to 1
                
                # Swing (Joint 1 - Hip Rotation)
                swing_rot = -amplitude_swing * np.cos(np.pi * phase_progress) + amplitude_swing # 0 -> +2*amplitude_swing
                # Lift (Joint 2 - Hip Up/Down)
                swing_lift = amplitude_lift * np.sin(np.pi * phase_progress) # 0 -> +amplitude_lift -> 0
                
                # Apply motion to Group 1
                for i in [0, 9]: # Hip Rotation (G1)
                    data.ctrl[i] = hip_base_rot + swing_rot
                for i in [1, 10]: # Hip Up/Down (G1)
                    data.ctrl[i] = hip_base_up + swing_lift
                for i in [2, 11]: # Knee (G1) - compensate lift
                    data.ctrl[i] = knee_base - swing_lift
                
                # Body stays put (controlled by data.ctrl[body_x_ctrl_idx] = initial_body_x)
                
            # PHASE 2: Body Translates Forward (Group 2 grounded) (t = step_duration to 2*step_duration)
            elif t < 2 * step_duration:
                # Body translation phase. Both groups are nominally grounded.
                time_in_phase = t - step_duration
                phase_progress = time_in_phase / step_duration # 0 to 1
                
                # Body moves forward linearly. Start at 0, end at body_step_distance.
                # data.ctrl[body_x_ctrl_idx] smoothly drives the body forward
                data.ctrl[body_x_ctrl_idx] = initial_body_x + body_step_distance * phase_progress
                
                # Grounded legs (G1 and G2) should remain at their last commanded position
                
            # PHASE 3: Group 2 (FL, BR) Swings Forward (t = 2*step_duration to 3*step_duration)
            elif t < 3 * step_duration:
                # Group 2 lifts and swings forward
                time_in_phase = t - 2 * step_duration
                phase_progress = time_in_phase / step_duration # 0 to 1
                
                # Swing (Hip Rotation)
                swing_rot = -amplitude_swing * np.cos(np.pi * phase_progress) + amplitude_swing
                # Lift (Hip Up/Down)
                swing_lift = amplitude_lift * np.sin(np.pi * phase_progress)
                
                # Apply motion to Group 2
                for i in [3, 6]: # Hip Rotation (G2)
                    data.ctrl[i] = hip_base_rot + swing_rot
                for i in [4, 7]: # Hip Up/Down (G2)
                    data.ctrl[i] = hip_base_up + swing_lift
                for i in [5, 8]: # Knee (G2) - compensate lift
                    data.ctrl[i] = knee_base - swing_lift

                # Body stays put
                
            # PHASE 4: Body Translates Forward (Group 1 grounded) (t = 3*step_duration to cycle_duration)
            else: # t < cycle_duration
                # Body translation phase.
                time_in_phase = t - 3 * step_duration
                phase_progress = time_in_phase / step_duration # 0 to 1
                
                # Body moves forward linearly. Start at body_step_distance, end at 2 * body_step_distance.
                data.ctrl[body_x_ctrl_idx] = initial_body_x + body_step_distance + body_step_distance * phase_progress
                
                # Grounded legs (G1 and G2) should remain at their last commanded position
            
            
            # --- End Phase Logic ---
            
        mujoco.mj_step(model, data)
        viewer.sync()