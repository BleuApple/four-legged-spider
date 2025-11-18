import math
import numpy as np

import numpy as np

class SpiderRobotMath:
    """
    A collection of mathematical functions for a 3DOF robotic leg.
    """
    def __init__(self, femur_length:float, tibia_length:float, base_length:float, \
                 default_height:float = None, step_length:float = None, qpos = None,\
                 span_len:float = None):
        self.f = femur_length
        self.t = tibia_length
        self.b = base_length
        self.h = default_height if default_height is not None else (tibia_length - femur_length) * 0.5
        self.step_length = step_length if step_length is not None else femur_length * 0.75
        self.lift_angle = 85 # degrees
        self.qpos = qpos
        self.span_len = span_len if span_len is not None else (tibia_length + femur_length) * 0.5

        thet1, thet2, thet3 = self.leg_inv_kinem(self.span_len, 0, -self.h)

        self.def_qpos = [0,0,0,
                         1,0,0,0,
                         thet1, thet2, thet3,
                         thet1, thet2, thet3,
                         thet1, thet2, thet3]
        

    def update_qpos(self, qpos):
        self.qpos = qpos
        
    def dh_parameter_to_homo_transform(self, d:float, theta:float, a:float, alpha:float):
        return np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                         [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                         [            0,                np.sin(alpha),                np.cos(alpha),               d],
                         [            0,                            0,                            0,               1]])

    def leg_fwd_kinem(self, theta1: float, theta2:float, theta3:float):
        """
        Forward Kinematics for 3DOF robotic leg.
        Takes theta1, theta2, theta3 for the leg in question
        """

        T1 = self.dh_parameter_to_homo_transform(0, theta1, 0, math.pi/2)
        T2 = self.dh_parameter_to_homo_transform(0, theta2, self.f, 0)
        T3 = self.dh_parameter_to_homo_transform(0, theta3, self.t, 0)

        totalT = T1@T2@T3

        final_vector = totalT@np.array([[0],
                                        [0],
                                        [0],
                                        [1]])
        
        x, y, z = final_vector[0][0], final_vector[1][0], final_vector[2][0]


        return x, y, z


    def leg_inv_kinem(self, x:float, y:float, z:float):
        """
        Inverse kinematics for a 3DOF robotic leg.
        Takes x y z in reference to the hip joint.
        """

        #print("x, y z are", x, y, z)
        l = math.sqrt(x**2 + y**2)
        r = math.sqrt(l**2 + z**2)

        if r > (self.f + self.t):
            raise ValueError("Target point is out of reach for the leg.")
        if r < abs(self.f - self.t):
            raise ValueError("Target point is too close for the leg.")
        if l == 0:
            raise ValueError("Undefined hip yaw angle for vertical position.")
        
        theta1 = math.atan2(y, x)

        theta2 =  math.atan2(z, l) - math.acos((r**2 + self.f**2 - self.t**2 ) / (2 * r * self.f))

        theta3 = math.atan2(z - self.f * math.sin(theta2), l - self.f * math.cos(theta2)) - theta2

        return theta1, -theta2, -theta3
    
    def total_inv_kinem(self, x:float, y:float, z:float, leg_index:int):
        """
        Inverse kinematics for the leg.
        Takes x y z in reference to the robot center.
        """

        x_ = x - self.b/2 if leg_index % 2 != 0 else x + self.b/2

        y_ = y - self.b/2 if leg_index < 3 else y + self.b/2

        rotAngle = 45 - 90 * leg_index

        x_, y_ = np.array([[np.cos(rotAngle), -np.sin(rotAngle)],[np.sin(rotAngle), np.cos(rotAngle)]]) @ np.array([[x_], [y_]])

        return self.leg_inv_kinem(x_, y_, z)
    
    def in_body_inv_kinem(self, body_dx:float, body_dy:float, body_dz:float, curr_qpos=None,to_print=False):
        """Returns ctrl param order format"""
        if curr_qpos is None: curr_qpos = self.qpos
        curr_qpos = np.copy(curr_qpos)
        #print("dx, dy, dz", dx, dy, dz)
        dx, dy, dz = -body_dx, -body_dy, -body_dz # convert body movement to how legs should actuall move


        # Getting Relative Coordinates
        FR_pos_ = self.leg_fwd_kinem(curr_qpos[7],  curr_qpos[8],  curr_qpos[9])
        FL_pos_ = self.leg_fwd_kinem(curr_qpos[10], curr_qpos[11], curr_qpos[12])
        BL_pos_ = self.leg_fwd_kinem(curr_qpos[13], curr_qpos[14], curr_qpos[15])
        BR_pos_ = self.leg_fwd_kinem(curr_qpos[16], curr_qpos[17], curr_qpos[18])

        if to_print:
            print("calculating in body inv kinem")
            print("FR_pos_:", FR_pos_)
            print("BR_pos_:", BR_pos_)
            print("FL_pos_:", FL_pos_)
            print("BL_pos_:", BL_pos_)

        rota = lambda x, y, z, leg_index : np.array([[np.cos(np.deg2rad(45 - 90 * leg_index)), -np.sin(np.deg2rad(45 - 90 * leg_index)), 0],
                                                     [np.sin(np.deg2rad(45 - 90 * leg_index)),  np.cos(np.deg2rad(45 - 90 * leg_index)), 0],
                                                     [                                      0,                            0,             1]])\
              @ np.array([[x], [y], [z]])
        
        FR_pos = np.array(FR_pos_) + rota(dx, dy, dz, 1).flatten()
        FL_pos = np.array(FL_pos_) + rota(dx, dy, dz, 2).flatten()
        BL_pos = np.array(BL_pos_) + rota(dx, dy, dz, 3).flatten()
        BR_pos = np.array(BR_pos_) + rota(dx, dy, dz, 4).flatten()
        
        if to_print:
            print("FR_pos:", FR_pos)
            print("FL_pos:", FL_pos)
            print("BL_pos:", BL_pos)
            print("BR_pos:", BR_pos)

        FR_pos[2] = -1 * FR_pos[2] if FR_pos[2] < 0 else FR_pos[2]
        FL_pos[2] = -1 * FL_pos[2] if FL_pos[2] < 0 else FL_pos[2]
        BL_pos[2] = -1 * BL_pos[2] if BL_pos[2] < 0 else BL_pos[2]
        BR_pos[2] = -1 * BR_pos[2] if BR_pos[2] < 0 else BR_pos[2]



        FR_ctrl = self.leg_inv_kinem(FR_pos[0], FR_pos[1], FR_pos[2])
        FL_ctrl = self.leg_inv_kinem(FL_pos[0], FL_pos[1], FL_pos[2])
        BL_ctrl = self.leg_inv_kinem(BL_pos[0], BL_pos[1], BL_pos[2])
        BR_ctrl = self.leg_inv_kinem(BR_pos[0], BR_pos[1], BR_pos[2])

        if to_print:
            print("FR_fwd:", self.leg_fwd_kinem(FR_ctrl[0], FR_ctrl[1], FR_ctrl[2]))
            print("FL_fwd:", self.leg_fwd_kinem(FL_ctrl[0], FL_ctrl[1], FL_ctrl[2]))
            print("BL_fwd:", self.leg_fwd_kinem(BL_ctrl[0], BL_ctrl[1], BL_ctrl[2]))
            print("BR_fwd:", self.leg_fwd_kinem(BR_ctrl[0], BR_ctrl[1], BR_ctrl[2]))

        ctrl_params = [FR_ctrl, FL_ctrl, BL_ctrl, BR_ctrl]

        return ctrl_params
    
    def stable_centre_coords(self, curr_qpos=None, leg_index=None,to_print=False):
        """Returns according to leg indices """

        if curr_qpos is None: curr_qpos = self.qpos
        curr_qpos = np.copy(curr_qpos)


        # Getting Relative Coordinates
        FR_pos_ = self.leg_fwd_kinem(curr_qpos[7],  curr_qpos[8],  curr_qpos[9])
        FL_pos_ = self.leg_fwd_kinem(curr_qpos[10], curr_qpos[11], curr_qpos[12])
        BL_pos_ = self.leg_fwd_kinem(curr_qpos[13], curr_qpos[14], curr_qpos[15])
        BR_pos_ = self.leg_fwd_kinem(curr_qpos[16], curr_qpos[17], curr_qpos[18])

        # Rotation Function
        rota = lambda x, y, z, leg_index : np.array([[np.cos(np.deg2rad(90 * leg_index - 45)), -np.sin(np.deg2rad(90 * leg_index - 45)), 0],\
                                                     [np.sin(np.deg2rad(90 * leg_index - 45)),  np.cos(np.deg2rad(90 * leg_index - 45)), 0],
                                                     [0,                            0,                           1]])\
              @ np.array([[x], [y], [z]])

        # Rotate then add
        FR_pos = rota(FR_pos_[0], FR_pos_[1], FR_pos_[2], 1) + np.array([[ self.b * 0.5],[ self.b * 0.5],[0]])
        FL_pos = rota(FL_pos_[0], FL_pos_[1], FL_pos_[2], 2) + np.array([[-self.b * 0.5],[ self.b * 0.5],[0]])
        BL_pos = rota(BL_pos_[0], BL_pos_[1], BL_pos_[2], 3) + np.array([[-self.b * 0.5],[-self.b * 0.5],[0]])
        BR_pos = rota(BR_pos_[0], BR_pos_[1], BR_pos_[2], 4) + np.array([[ self.b * 0.5],[-self.b * 0.5],[0]])

        pos_sum = FR_pos + BR_pos + FL_pos + BL_pos

        current_centre = pos_sum / 4

        # Ordered according to leg indexes
        # Move nth of the way to opposite leg
        nth = 0

        """
        + np.array([[BL_pos[0][0]], [BL_pos[1][0]], [0]]) * nth
        + np.array([[BR_pos[0][0]], [BR_pos[1][0]], [0]]) * nth
        + np.array([[FL_pos[0][0]], [FL_pos[1][0]], [0]]) * nth
        + np.array([[FR_pos[0][0]], [FR_pos[1][0]], [0]]) * nth
        """
        stable_centres = np.array([pos_sum - FR_pos + np.array([[BL_pos[0][0]], [BL_pos[1][0]], [0]]) * nth ,\
                                   pos_sum - FL_pos + np.array([[BR_pos[0][0]], [BR_pos[1][0]], [0]]) * nth ,\
                                   pos_sum - BL_pos + np.array([[FR_pos[0][0]], [FR_pos[1][0]], [0]]) * nth ,\
                                   pos_sum - BR_pos + np.array([[FL_pos[0][0]], [FL_pos[1][0]], [0]]) * nth ]) / 3

        if to_print:
            print("Stable centre coords prints")
            print("FR_pos:", FR_pos.flatten())
            print("FL_pos:", FL_pos.flatten())
            print("BL_pos:", BL_pos.flatten())
            print("BR_pos:", BR_pos.flatten())
            print("FR_pos_:", FR_pos_)
            print("FL_pos_:", FL_pos_)
            print("BR_pos_:", BR_pos_)
            print("BL_pos_:", BL_pos_)

            print("current centre is", current_centre)
            print("stable centres is", stable_centres)

        if leg_index is None:
            return stable_centres , current_centre
        
        return stable_centres[leg_index - 1], current_centre





        
if __name__ == "__main__":        
    spy = SpiderRobotMath(1.5, 2.5, 1, None, 0.3, None)

    print(spy.leg_inv_kinem(3, 0, -2))

    print(180*np.array(spy.leg_inv_kinem(3, 0, -2))/math.pi)
    print(spy.leg_fwd_kinem(0.0, -1.176005207095135, 0.9272952180016121))