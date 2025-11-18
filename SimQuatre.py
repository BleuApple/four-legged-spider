import mujoco
import mujoco.viewer
import numpy as np
import time
from robokinemath import *

# Initialize MuJoCo model
model = mujoco.MjModel.from_xml_path("Spider_v1.xml")
data = mujoco.MjData(model)

# Reset to start pose
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "start_pose")
mujoco.mj_resetDataKeyframe(model, data, key_id)

# Robot parameters
span_len = 2
height = 1.2
femur_len = 3
tibia_len = 4
base_len = 1
lift_height = 2  # How high to lift legs during movement

# Initialize spider math helper
spy = SpiderRobotMath(femur_len, tibia_len, base_len, height, 0, 0.6, span_len)
def_params = spy.leg_inv_kinem(span_len, 0, height)

print("Default leg parameters:", def_params)
print("Default position:", spy.leg_fwd_kinem(def_params[0], def_params[1], def_params[2]))

# Initial position - all legs at default stance
initial_qpos = np.array([
    0, 0, 0, 1, 0, 0, 0,  # Body position and orientation (quaternion)
    def_params[0], def_params[1], def_params[2],  # FR
    def_params[0], def_params[1], def_params[2],  # FL
    def_params[0], def_params[1], def_params[2],  # BL
    def_params[0], def_params[1], def_params[2],  # BR
])

data.qpos[:] = initial_qpos

# Phase definitions
# Each phase: (type, leg_index, dx, dy, duration)
# type: 'leg' for leg movement, 'body' for body movement
# leg_index: 1=FR, 2=FL, 3=BL, 4=BR (ignored for body movements)
PHASES = [
    # Phase 0: Move FR leg by (1, 1)
    ('body', None, -0.1, -0.1, 1.0),

    ('leg', 1, 1.0, 1.0, 1.0),
    # Phase 1: Move body by (1, 1)
    ('body', None, 1.1, 1.1, 1.0),
    # Phase 2: Move BL leg by (1, 1)
    ('leg', 3, 1.0, 1.0, 1.0),
    # Phase 3: Move body by (-0.2, 0.2)
    ('body', None, -0.2, 0.2, 1.0),
    # Phase 4: Move BR leg by (1, 1)
    ('leg', 4, 1.0, 1.0, 1.0),
    # Phase 5: Move body by (0.4, -0.4)
    ('body', None, 0.4, -0.4, 1.0),
    # Phase 6: Move FL leg by (1, 1)
    ('leg', 2, 1.0, 1.0, 1.0),
    # Phase 7: Move body by (-0.2, 0.2) - recentering
    ('body', None, -0.2, 0.2, 1.0),
]

def interpolate_leg_movement(spy, curr_qpos, leg_index, target_dx, target_dy, progress):
    """
    Move a leg through a lift-move-lower trajectory.
    Progress goes from 0 to 1.
    """
    curr_qpos = np.copy(curr_qpos)
    
    # Get current leg position
    leg_qpos_idx = 7 + (leg_index - 1) * 3
    current_leg_angles = curr_qpos[leg_qpos_idx:leg_qpos_idx + 3]
    start_pos = spy.leg_fwd_kinem(current_leg_angles[0], current_leg_angles[1], current_leg_angles[2])
    
    # Rotation for leg coordinate system
    rota = lambda x, y, leg_idx: np.array([
        [np.cos(np.deg2rad(45 - 90 * leg_idx)), -np.sin(np.deg2rad(45 - 90 * leg_idx))],
        [np.sin(np.deg2rad(45 - 90 * leg_idx)),  np.cos(np.deg2rad(45 - 90 * leg_idx))]
    ]) @ np.array([[x], [y]])
    
    # Calculate target position in leg frame
    delta_rotated = rota(target_dx, target_dy, leg_index)
    target_x = start_pos[0] + delta_rotated[0, 0]
    target_y = start_pos[1] + delta_rotated[1, 0]
    target_z = start_pos[2]  # Keep same height when on ground
    
    # Three-phase movement: lift (0-0.33), move (0.33-0.66), lower (0.66-1.0)
    if progress < 0.33:
        # Lift phase
        lift_progress = progress / 0.33
        x = start_pos[0]
        y = start_pos[1]
        z = start_pos[2] - lift_height * lift_progress
    elif progress < 0.66:
        # Move phase (while lifted)
        move_progress = (progress - 0.33) / 0.33
        x = start_pos[0] + (target_x - start_pos[0]) * move_progress
        y = start_pos[1] + (target_y - start_pos[1]) * move_progress
        z = start_pos[2] - lift_height
    else:
        # Lower phase
        lower_progress = (progress - 0.66) / 0.34
        x = target_x
        y = target_y
        z = start_pos[2] - lift_height * (1 - lower_progress)
    
    # Calculate inverse kinematics for new position
    new_angles = spy.leg_inv_kinem(x, y, z)
    
    # Update only this leg's angles
    result_qpos = np.copy(curr_qpos)
    result_qpos[leg_qpos_idx:leg_qpos_idx + 3] = new_angles
    
    return result_qpos[7:]  # Return only joint angles (ctrl format)

def interpolate_body_movement(spy, initial_qpos, target_qpos, progress):
    """
    Move the body (keeping all legs stationary relative to ground).
    This means moving all legs in the opposite direction in the body frame.
    Progress interpolates between initial_qpos and target_qpos.
    """
    # Simple linear interpolation between initial and target positions
    initial_ctrl = initial_qpos[7:]
    target_ctrl = target_qpos[7:]
    
    interpolated = initial_ctrl + (target_ctrl - initial_ctrl) * progress
    return interpolated

# Main simulation loop
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Starting simulation. Holding initial position for 2 seconds...")
    
    start_time = time.time()
    current_phase = -1  # -1 means in initial hold period
    phase_start_time = start_time
    target_ctrl = initial_qpos[7:]  # Initial control target
    initial_qpos_for_phase = np.copy(data.qpos)
    target_qpos_for_phase = None  # Will store the target qpos for body movements
    
    while viewer.is_running():
        current_time = time.time()
        elapsed_time = current_time - start_time
        
        # Initial hold period
        if elapsed_time < 2.0:
            data.ctrl[:] = initial_qpos[7:]
        else:
            # Start cycling through phases
            if current_phase == -1:
                current_phase = 0
                phase_start_time = current_time
                initial_qpos_for_phase = np.copy(data.qpos)
                phase_type, leg_idx, dx, dy, duration = PHASES[current_phase]
                
                # Pre-calculate target for body movements
                if phase_type == 'body':
                    target_qpos_for_phase = np.copy(initial_qpos_for_phase)
                    target_ctrl_calc = spy.in_body_inv_kinem(dx, dy, 0, initial_qpos_for_phase, to_print=False)
                    target_qpos_for_phase[7:] = np.array(target_ctrl_calc).flatten()
                
                print(f"\nPhase {current_phase}: {phase_type}", end="")
                if phase_type == 'leg':
                    print(f" - Moving leg {leg_idx} by ({dx}, {dy})")
                else:
                    print(f" - Moving body by ({dx}, {dy})")
            
            phase_elapsed = current_time - phase_start_time
            phase_type, leg_idx, dx, dy, duration = PHASES[current_phase]
            
            # Calculate progress (0 to 1)
            progress = min(phase_elapsed / duration, 1.0)
            
            # Calculate target control based on phase type
            if phase_type == 'leg':
                target_ctrl = interpolate_leg_movement(
                    spy, initial_qpos_for_phase, leg_idx, dx, dy, progress
                )
            else:  # body movement
                target_ctrl = interpolate_body_movement(
                    spy, initial_qpos_for_phase, target_qpos_for_phase, progress
                )
            
            # Apply control
            data.ctrl[:] = target_ctrl
            
            # Check if phase is complete
            if progress >= 1.0 and np.all(np.abs(data.qpos[7:] - target_ctrl) < 0.08):
                print(f"Phase {current_phase} complete!")
                
                # Move to next phase
                current_phase = (current_phase + 1) % len(PHASES)
                phase_start_time = current_time
                initial_qpos_for_phase = np.copy(data.qpos)
                
                phase_type, leg_idx, dx, dy, duration = PHASES[current_phase]
                
                # Pre-calculate target for body movements
                if phase_type == 'body':
                    target_qpos_for_phase = np.copy(initial_qpos_for_phase)
                    target_ctrl_calc = spy.in_body_inv_kinem(dx, dy, 0, initial_qpos_for_phase, to_print=False)
                    target_qpos_for_phase[7:] = np.array(target_ctrl_calc).flatten()
                
                print(f"\nPhase {current_phase}: {phase_type}", end="")
                if phase_type == 'leg':
                    print(f" - Moving leg {leg_idx} by ({dx}, {dy})")
                else:
                    print(f" - Moving body by ({dx}, {dy})")
        
        # Step simulation
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)  # Small delay for visualization