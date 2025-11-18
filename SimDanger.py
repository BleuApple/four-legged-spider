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

x_o = -1
y_o = 1

# Robot parameters
span_len = 2
height = 1.2
femur_len = 3
tibia_len = 4
base_len = 1
lift_height = 2  # How high to lift legs during movement
settling_time = 0.3  # Time to wait after each phase for stabilization

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
# type: 'leg' for leg movement, 'body' for body movement, 'reset' for returning to initial position
# leg_index: 1=FR, 2=FL, 3=BL, 4=BR (ignored for body movements and reset)

def get_phases_for_direction(x_o, y_o):
    """
    Returns PHASES array depending on the direction (quadrant)
    defined by x_o and y_o.
    
    Leg indices:
        1 = FR
        2 = FL
        3 = BL
        4 = BR
    """
    # Default order = x>0, y>0  (FR, BL, BR, FL)
    if x_o >= 0 and y_o >= 0:
        leg_order = [1, 3, 4, 2]

    # x < 0, y > 0  → Moving left-up → FL moves first
    elif x_o < 0 and y_o >= 0:
        leg_order = [2, 4, 3, 1]

    # x < 0, y < 0  → Moving back-left → BL moves first
    elif x_o < 0 and y_o < 0:
        leg_order = [3, 1, 2, 4]

    # x > 0, y < 0 → Moving back-right → BR moves first
    else:
        leg_order = [4, 2, 1, 3]

    # Build PHASES using this order
    return [
        ('body', None, -0.1 * x_o, -0.1 * y_o, 1.0),

        ('leg', leg_order[0], 1.0 * x_o, 1.0 * y_o, 1.0),
        ('body', None, 1.1 * x_o, 1.1 * y_o, 1.0),

        ('leg', leg_order[1], 1.0 * x_o, 1.0 * y_o, 1.0),
        ('body', None, -0.2 * x_o, 0.2 * y_o, 1.0),

        ('leg', leg_order[2], 1.0 * x_o, 1.0 * y_o, 1.0),
        ('body', None, 0.4 * x_o, -0.4 * y_o, 1.0),

        ('leg', leg_order[3], 1.0 * x_o, 1.0 * y_o, 1.0),
        ('body', None, -0.2 * x_o, 0.2 * y_o, 1.0),

        ('reset', None, 0, 0, 1.0),
    ]

# Replace this line:
# PHASES = [...]

# With this:
PHASES = get_phases_for_direction(x_o, y_o)

"""
PHASES = [
    # Phase 0: Initial body shift
    ('body', None, -0.1*x_o, -0.1*y_o, 1.0),
    # Phase 1: Move FR leg
    ('leg', 1, 1.0*x_o, 1.0*y_o, 1.0),
    # Phase 2: Move body
    ('body', None, 1.1*x_o, 1.1*y_o, 1.0),
    # Phase 3: Move BL leg
    ('leg', 3, 1.0*x_o, 1.0*y_o, 1.0),
    # Phase 4: Move body
    ('body', None, -0.2*x_o, 0.2*y_o, 1.0),
    # Phase 5: Move BR leg
    ('leg', 4, 1.0*x_o, 1.0*y_o, 1.0),
    # Phase 6: Move body
    ('body', None, 0.4*x_o, -0.4*y_o, 1.0),
    # Phase 7: Move FL leg
    ('leg', 2, 1.0*x_o, 1.0*y_o, 1.0),
    # Phase 8: Recentering
    ('body', None, -0.2*x_o, 0.2*y_o, 1.0),
    # Phase 9: Reset to initial position (height readjustment)
    ('reset', None, 0, 0, 1.0),
]
"""

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

def interpolate_reset_movement(initial_qpos_phase, target_qpos, progress):
    """
    Reset all legs to initial stance position.
    Progress interpolates between current position and initial_qpos.
    """
    initial_ctrl = initial_qpos_phase[7:]
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
    
    # Settling state
    in_settling = False
    settling_start_time = None
    
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
                elif phase_type == 'body':
                    print(f" - Moving body by ({dx}, {dy})")
                else:
                    print(f" - Resetting to initial position")
            
            # Handle settling period
            if in_settling:
                settling_elapsed = current_time - settling_start_time
                if settling_elapsed < settling_time:
                    # Just hold position during settling
                    pass
                else:
                    print(f"Settling complete. Body height: {data.qpos[2]:.3f}")
                    in_settling = False
                    
                    # Now move to next phase
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
                    elif phase_type == 'body':
                        print(f" - Moving body by ({dx}, {dy})")
                    else:
                        print(f" - Resetting to initial position")
            
            # Normal phase execution
            else:
                phase_elapsed = current_time - phase_start_time
                phase_type, leg_idx, dx, dy, duration = PHASES[current_phase]
                
                # Calculate progress (0 to 1)
                progress = min(phase_elapsed / duration, 1.0)
                
                # Calculate target control based on phase type
                if phase_type == 'leg':
                    target_ctrl = interpolate_leg_movement(
                        spy, initial_qpos_for_phase, leg_idx, dx, dy, progress
                    )
                elif phase_type == 'body':
                    target_ctrl = interpolate_body_movement(
                        spy, initial_qpos_for_phase, target_qpos_for_phase, progress
                    )
                else:  # reset phase
                    target_ctrl = interpolate_reset_movement(
                        initial_qpos_for_phase, initial_qpos, progress
                    )
                
                # Apply control
                data.ctrl[:] = target_ctrl
                
                # Check if phase is complete
                if progress >= 1.0 and np.all(np.abs(data.qpos[7:] - target_ctrl) < 0.08):
                    print(f"Phase {current_phase} complete!")
                    
                    if phase_type == 'reset':
                        print(f"Reset complete! Body height: {data.qpos[2]:.3f}")
                    
                    # Start settling period
                    in_settling = True
                    settling_start_time = current_time
                    print(f"Settling for {settling_time}s...")
        
        # Step simulation
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)  # Small delay for visualization