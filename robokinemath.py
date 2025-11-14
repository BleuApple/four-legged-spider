import math
import numpy as np

import numpy as np

def project_point_into_triangle_with_margin(A, B, C, P, eps=1e-1):
    """Projects P into triangle ABC with an epsilon interior margin."""

    # ---- Compute inner (shrunk) triangle ----
    def shift_vertex(V, Vprev, Vnext):
        e1 = (Vprev - V) / np.linalg.norm(Vprev - V)
        e2 = (Vnext - V) / np.linalg.norm(Vnext - V)
        bis = e1 + e2
        n = np.linalg.norm(bis)
        return V if n < 1e-9 else V + (eps * bis / n)

    A2 = shift_vertex(A, B, C)
    B2 = shift_vertex(B, C, A)
    C2 = shift_vertex(C, A, B)

    # If shrinking kills triangle → fallback to original
    if np.linalg.norm(np.cross(B2 - A2, C2 - A2)) < 1e-9:
        A2, B2, C2 = A, B, C

    # ---- Barycentric projection ----
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

    # clamp inside triangle
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1 - u)
    w = 1 - u - v

    return u * A2 + v * B2 + w * C2

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
    
    def in_body_inv_kinem(self, dx:float, dy:float, dz:float, curr_qpos=None,to_print=False):
        """Returns ctrl param order format"""
        if curr_qpos is None: curr_qpos = self.qpos
        curr_qpos = np.copy(curr_qpos)
        #print("dx, dy, dz", dx, dy, dz)

        # Getting Relative Coordinates
        FR_pos_ = self.leg_fwd_kinem(curr_qpos[7],  curr_qpos[8],  curr_qpos[9])
        BR_pos_ = self.leg_fwd_kinem(curr_qpos[10], curr_qpos[11], curr_qpos[12])
        FL_pos_ = self.leg_fwd_kinem(curr_qpos[13], curr_qpos[14], curr_qpos[15])
        BL_pos_ = self.leg_fwd_kinem(curr_qpos[16], curr_qpos[17], curr_qpos[18])

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

        FR_ctrl = self.leg_inv_kinem(FR_pos[0], FR_pos[1], FR_pos[2])
        FL_ctrl = self.leg_inv_kinem(FL_pos[0], FL_pos[1], FL_pos[2])
        BL_ctrl = self.leg_inv_kinem(BL_pos[0], BL_pos[1], BL_pos[2])
        BR_ctrl = self.leg_inv_kinem(BR_pos[0], BR_pos[1], BR_pos[2])

        if to_print:
            print("FR_fwd:", self.leg_fwd_kinem(FR_ctrl[0], FR_ctrl[1], FR_ctrl[2]))
            print("FL_fwd:", self.leg_fwd_kinem(FL_ctrl[0], FL_ctrl[1], FL_ctrl[2]))
            print("BL_fwd:", self.leg_fwd_kinem(BL_ctrl[0], BL_ctrl[1], BL_ctrl[2]))
            print("BR_fwd:", self.leg_fwd_kinem(BR_ctrl[0], BR_ctrl[1], BR_ctrl[2]))

        ctrl_params = [FR_ctrl, BR_ctrl, FL_ctrl, BL_ctrl]

        return ctrl_params
    
    def stable_centre_coords_old(self, curr_qpos=None, leg_index=None,to_print=False):
        """Returns according to leg indices """

        if curr_qpos is None: curr_qpos = self.qpos
        curr_qpos = np.copy(curr_qpos)


        # Getting Relative Coordinates
        FR_pos_ = self.leg_fwd_kinem(curr_qpos[7],  curr_qpos[8],  curr_qpos[9])
        BR_pos_ = self.leg_fwd_kinem(curr_qpos[10], curr_qpos[11], curr_qpos[12])
        FL_pos_ = self.leg_fwd_kinem(curr_qpos[13], curr_qpos[14], curr_qpos[15])
        BL_pos_ = self.leg_fwd_kinem(curr_qpos[16], curr_qpos[17], curr_qpos[18])

        # Rotation Function
        rota = lambda x, y, z, leg_index : np.array([[np.cos(np.deg2rad(90 * leg_index - 45)), -np.sin(np.deg2rad(90 * leg_index - 45)), 0],\
                                                     [np.sin(np.deg2rad(90 * leg_index - 45)),  np.cos(np.deg2rad(90 * leg_index - 45)), 0],
                                                     [0,                            0,                           1]])\
              @ np.array([[x], [y], [z]])

        # Rotate then add
        FR_pos = rota(FR_pos_[0], FR_pos_[1], FR_pos_[2], 1) + np.array([[ self.b * 0.5],[ self.b * 0.5],[0]])
        BR_pos = rota(BR_pos_[0], BR_pos_[1], BR_pos_[2], 4) + np.array([[ self.b * 0.5],[-self.b * 0.5],[0]])
        FL_pos = rota(FL_pos_[0], FL_pos_[1], FL_pos_[2], 2) + np.array([[-self.b * 0.5],[ self.b * 0.5],[0]])
        BL_pos = rota(BL_pos_[0], BL_pos_[1], BL_pos_[2], 3) + np.array([[-self.b * 0.5],[-self.b * 0.5],[0]])

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
                                   pos_sum - BR_pos + np.array([[FL_pos[0][0]], [FL_pos[1][0]], [0]]) * nth ,\
                                   pos_sum - BL_pos + np.array([[FR_pos[0][0]], [FR_pos[1][0]], [0]]) * nth ]) / 3

        if to_print:
            print("Stable centre coords prints")
            print("FR_pos:", FR_pos.flatten())
            print("FL_pos:", FL_pos.flatten())
            print("BR_pos:", BR_pos.flatten())
            print("BL_pos:", BL_pos.flatten())
            print("FR_pos_:", FR_pos_)
            print("FL_pos_:", FL_pos_)
            print("BR_pos_:", BR_pos_)
            print("BL_pos_:", BL_pos_)

            print("current centre is", current_centre)
            print("stable centres is", stable_centres)

        if leg_index is None:
            return stable_centres , current_centre
        
        return stable_centres[leg_index - 1], current_centre

 
    def stable_centre_coords(self, curr_qpos=None, leg_index=None, to_print=False, epsilon=0.1):
        """Returns according to leg indices, with closest point to COM inside support triangle 
        that is at least epsilon away from edges"""
        
        if curr_qpos is None: 
            curr_qpos = self.qpos
        curr_qpos = np.copy(curr_qpos)

        # Getting Relative Coordinates
        FR_pos_ = self.leg_fwd_kinem(curr_qpos[7],  curr_qpos[8],  curr_qpos[9])
        BR_pos_ = self.leg_fwd_kinem(curr_qpos[10], curr_qpos[11], curr_qpos[12])
        FL_pos_ = self.leg_fwd_kinem(curr_qpos[13], curr_qpos[14], curr_qpos[15])
        BL_pos_ = self.leg_fwd_kinem(curr_qpos[16], curr_qpos[17], curr_qpos[18])

        # Rotation Function
        rota = lambda x, y, z, leg_index : np.array([[np.cos(np.deg2rad(90 * leg_index - 45)), -np.sin(np.deg2rad(90 * leg_index - 45)), 0],\
                                                    [np.sin(np.deg2rad(90 * leg_index - 45)),  np.cos(np.deg2rad(90 * leg_index - 45)), 0],
                                                    [0,                            0,                           1]])\
            @ np.array([[x], [y], [z]])

        # Rotate then add
        FR_pos = rota(FR_pos_[0], FR_pos_[1], FR_pos_[2], 1) + np.array([[ self.b * 0.5],[ self.b * 0.5],[0]])
        BR_pos = rota(BR_pos_[0], BR_pos_[1], BR_pos_[2], 4) + np.array([[ self.b * 0.5],[-self.b * 0.5],[0]])
        FL_pos = rota(FL_pos_[0], FL_pos_[1], FL_pos_[2], 2) + np.array([[-self.b * 0.5],[ self.b * 0.5],[0]])
        BL_pos = rota(BL_pos_[0], BL_pos_[1], BL_pos_[2], 3) + np.array([[-self.b * 0.5],[-self.b * 0.5],[0]])

        pos_sum = FR_pos + BR_pos + FL_pos + BL_pos
        current_centre = pos_sum / 4

        # Define support triangles for each leg
        support_triangles = [
            [FL_pos, BL_pos, BR_pos],  # Triangle for FR leg (opposite vertices)
            [FR_pos, BR_pos, BL_pos],  # Triangle for FL leg  
            [FL_pos, FR_pos, BL_pos],  # Triangle for BR leg
            [FR_pos, FL_pos, BR_pos]   # Triangle for BL leg
        ]
        
        def point_to_line_distance(point, line_p1, line_p2):
            """Calculate distance from point to line segment defined by p1 and p2"""
            p1 = line_p1.flatten()[:2]  # Use only x,y coordinates
            p2 = line_p2.flatten()[:2]
            pt = point.flatten()[:2]
            
            line_vec = p2 - p1
            line_len = np.linalg.norm(line_vec)
            line_unitvec = line_vec / line_len
            vec_from_p1 = pt - p1
            
            # Project point onto line
            projection_length = np.dot(vec_from_p1, line_unitvec)
            
            if projection_length < 0:
                # Closest to p1
                return np.linalg.norm(pt - p1)
            elif projection_length > line_len:
                # Closest to p2  
                return np.linalg.norm(pt - p2)
            else:
                # Closest to line segment
                closest_point = p1 + projection_length * line_unitvec
                return np.linalg.norm(pt - closest_point)
        
        def is_point_in_triangle(point, triangle):
            """Check if point is inside triangle using barycentric coordinates"""
            p = point.flatten()[:2]
            a, b, c = [v.flatten()[:2] for v in triangle]
            
            # Barycentric coordinate method
            v0 = b - a
            v1 = c - a
            v2 = p - a
            
            dot00 = np.dot(v0, v0)
            dot01 = np.dot(v0, v1)
            dot02 = np.dot(v0, v2)
            dot11 = np.dot(v1, v1)
            dot12 = np.dot(v1, v2)
            
            inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
            u = (dot11 * dot02 - dot01 * dot12) * inv_denom
            v = (dot00 * dot12 - dot01 * dot02) * inv_denom
            
            return (u >= 0) and (v >= 0) and (u + v <= 1)
        
        def project_point_to_triangle_interior(point, triangle, epsilon):
            """Find closest point to given point inside triangle that is at least epsilon from edges"""
            a, b, c = [v.flatten()[:2] for v in triangle]
            p = point.flatten()[:2]
            
            # Store original z-value to preserve it
            original_z = point[2][0]
            
            # First check if point is already inside and satisfies epsilon constraint
            if is_point_in_triangle(point, triangle):
                # Check distances to all edges
                dist_to_ab = point_to_line_distance(point, a, b)
                dist_to_bc = point_to_line_distance(point, b, c)  
                dist_to_ca = point_to_line_distance(point, c, a)
                
                if min(dist_to_ab, dist_to_bc, dist_to_ca) >= epsilon:
                    return np.array([[p[0]], [p[1]], [original_z]])
            
            # If not, we need to find the closest valid point
            # Use the centroid of the triangle shrunk by epsilon as a reasonable candidate
            edges = [(a, b), (b, c), (c, a)]
            
            # Calculate inward normals for each edge to shrink the triangle
            shrunken_vertices = []
            for i, (v1, v2, v3) in enumerate([(a, b, c), (b, c, a), (c, a, b)]):
                edge_vec = v2 - v1
                edge_len = np.linalg.norm(edge_vec)
                if edge_len > 0:  # Avoid division by zero
                    edge_unit = edge_vec / edge_len
                    
                    # Calculate inward normal (perpendicular to edge, pointing inside triangle)
                    to_third = v3 - v1
                    cross_val = np.cross(edge_vec, to_third)
                    normal = np.array([-edge_unit[1], edge_unit[0]]) if cross_val > 0 else np.array([edge_unit[1], -edge_unit[0]])
                    
                    # Move vertex inward by epsilon along the normal
                    shrunken_vertex = v1 + epsilon * normal
                    shrunken_vertices.append(shrunken_vertex)
                else:
                    shrunken_vertices.append(v1)
            
            # Use centroid of shrunken triangle as candidate point
            if len(shrunken_vertices) > 0:
                candidate = np.mean(shrunken_vertices, axis=0)
                candidate_3d = np.array([[candidate[0]], [candidate[1]], [original_z]])
                
                # Verify candidate is valid
                if is_point_in_triangle(candidate_3d, triangle):
                    return candidate_3d
            
            # Fallback: use the original triangle centroid with original z-value
            centroid = np.mean([a, b, c], axis=0)
            return np.array([[centroid[0]], [centroid[1]], [original_z]])
        
        # Calculate stable centres using the new method
        stable_centres = []
        for i, triangle in enumerate(support_triangles):
            stable_point = project_point_to_triangle_interior(current_centre, triangle, epsilon)
            stable_centres.append(stable_point)
        
        stable_centres = np.array(stable_centres)

        if to_print:
            print("Stable centre coords prints")
            print("FR_pos:", FR_pos.flatten())
            print("FL_pos:", FL_pos.flatten())
            print("BR_pos:", BR_pos.flatten())
            print("BL_pos:", BL_pos.flatten())
            print("FR_pos_:", FR_pos_)
            print("FL_pos_:", FL_pos_)
            print("BR_pos_:", BR_pos_)
            print("BL_pos_:", BL_pos_)
            print("current centre is", current_centre)
            print("stable centres is", stable_centres)

        if leg_index is None:
            return stable_centres, current_centre
        
        return stable_centres[leg_index - 1], current_centre

       
        
    
    def crawl_gait(self, t:float, direction:tuple, curr_qpos:float = None, phase_offset:float = 0.0, step_length:float = None):
        direction_magnitude = math.sqrt(direction[0]**2 + direction[1]**2)
        direction_unit = (direction[0]/direction_magnitude, direction[1]/direction_magnitude)\
                         if direction_magnitude != 0 else (0.0, 0.0)
        step_length = step_length if step_length is not None else self.step_length
        curr_qpos = curr_qpos if curr_qpos is not None else self.qpos
        if curr_qpos == None:
            raise ValueError("Missing qpos data")
        
        self.qpos = curr_qpos

        if direction[0] > 0 and direction_unit[1] > 0:
            rel_FR_ind = 0
        elif direction_unit[0] <= 0 and direction_unit[1] > 0:
            rel_FR_ind = 1
        elif direction_unit[0] <= 0 and direction_unit[1] <= 0:
            rel_FR_ind = 2
        elif direction_unit[0] > 0 and direction_unit[1] <= 0:
            rel_FR_ind = 3
        else:
            rel_FL_ind = None
            raise ValueError("Doesn't lie in any quadrants -> HOW on earth did this happen")
        
        rel_FL_ind, rel_BL_ind, rel_BR_ind = (rel_FR_ind + 1) % 4, (rel_FR_ind + 2)%4, (rel_FR_ind + 3)%4

        qposes = []
        traj_type = []
        
        # 13 step process -> 12 steps per leg, 1 step to come back to original pos
        rotAngle = 45 - 90 * rel_FR_ind

        rel_dir = np.array([[np.cos(rotAngle), -np.sin(rotAngle)],[np.sin(rotAngle), np.cos(rotAngle)]]) @ direction_unit


        # Moving relative FR first
        rel_FR_pos = self.leg_fwd_kinem(curr_qpos[7 + (rel_FR_ind*3), 7 + (rel_FR_ind*3 + 1), 7 + (rel_FR_ind*3 + 2)])
        ref_FR_x_, rel_FR_y_ = rel_FR_pos[0] + rel_dir * step_length, rel_FR_pos[1] + rel_dir * step_length

        rel_FR_theta1, rel_FR_theta2, rel_FR_theta3 = self.leg_inv_kinem(ref_FR_x_, rel_FR_y_, rel_FR_pos[2])

        # SCRAPPED THE IDEA TO ADD IT HERE
        # FUNCTIONALITY IN SIMULATOR SCRIPT
        
        raise NotImplementedError("Functionality in simtwo.py")

        return ...






        
if __name__ == "__main__":        
    spy = SpiderRobotMath(1.5, 2.5, 1, None, 0.3, None)

    print(spy.leg_inv_kinem(3, 0, -2))

    print(180*np.array(spy.leg_inv_kinem(3, 0, -2))/math.pi)
    print(spy.leg_fwd_kinem(0.0, -1.176005207095135, 0.9272952180016121))


