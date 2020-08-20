
import time
import oreo
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

pos_list = []
# Habitatai coordinate frame (z+ive, y+ive, xive) is Pybullet's frame (x+ive,z+ive, y+ive)
# The rotation matrix R to express a point (x,y,z) in Habitat's frame in Pybullet's frame
# It is the rotation matrix from Pybullet to Habitatai  row 1 [0 0 1], row 2 [1 0 0], row 3 [0,1,0]]
# Given a point in Habitat, we do a rotation to pybullet, determine if there is collision, find out
# the orientation of the eyes. This orientation has to be rotated using R_inverse.

def move_in_yaw(pitch):
    yaw = np.linspace(-25, 25, 50)
    for i in yaw:
        my_angles =[i,pitch,0.0,pitch]
        actuator_pos = robot.get_actuator_positions_for_a_given_yaw_pitch(my_angles)
        if actuator_pos[0] == 1:
            my_pos = actuator_pos[1:]
            print("My yaw-pitch angles={} and actuator positions = {}".format(my_angles,my_pos))
            pos_list.append(my_pos)
        else:
            print("Interpolator functions not computed")
            return 0
    return 1
def generate_interpolated_actuator_values():
    lefteye_l = []
    lefteye_r = []
    righteye_l = []
    righteye_r = []

    lefteye_l_outside_range = 0
    lefteye_r_outside_range = 0
    righteye_l_outside_range = 0
    righteye_r_outside_range = 0

    yaw = np.linspace(-25, 25, 100)
    pitch = np.linspace(75,125,100)

    for i in pitch:
        for j in yaw:
            my_angles = [j,i,j,i]
            actuator_pos = robot.get_actuator_positions_for_a_given_yaw_pitch(my_angles)
            if actuator_pos[0] == 1:
                if (-0.03<actuator_pos[1]<0.03):
                    lefteye_l.append([i,j,actuator_pos[1]])
                else:
                    lefteye_l_outside_range += 1

                if (-0.025 < actuator_pos[2] < 0.025):
                    lefteye_r.append([i, j, actuator_pos[2]])
                else:
                    lefteye_r_outside_range += 1

                if (-0.03<actuator_pos[3]<0.03):
                    righteye_l.append([i,j,actuator_pos[3]])
                else:
                    righteye_l_outside_range += 1

                if (-0.025 < actuator_pos[4] < 0.025):
                    righteye_r.append([i, j, actuator_pos[4]])
                else:
                    righteye_r_outside_range += 1

            else:
                print("Interpolator functions not computed")
                return 0,
    print("Bad Actuator values for LeftEye Left{} Right{}  RightEye Left{} Right".format(
        lefteye_l_outside_range, lefteye_r_outside_range, righteye_l_outside_range, righteye_r_outside_range
    ))
    return 1,lefteye_l, lefteye_r, righteye_l, righteye_r

def plot_interpolated(interp_val):
    if interp_val[0] == 1:
        lefteye_l_interp = np.array(interp_val[1])

        pit = lefteye_l_interp[:, 0]
        yaw = lefteye_l_interp[:, 1]
        left_a = lefteye_l_interp[:, 2]
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.scatter(yaw, pit, left_a, 'z', c='coral', )
        ax1.set_xlabel('Yaw in degrees')
        ax1.set_ylabel('Pitch in degrees')
        ax1.set_zlabel('Left Actuator - coral')
        ax1.set_title('Left Eye Left Actuator - Normal')

        lefteye_r_interp = np.array(interp_val[2])
        pit = lefteye_r_interp[:, 0]
        yaw = lefteye_r_interp[:, 1]
        left_a = lefteye_r_interp[:, 2]
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.scatter(yaw, pit, left_a, 'z', c='blue', )
        ax1.set_xlabel('Yaw in degrees')
        ax1.set_ylabel('Pitch in degrees')
        ax1.set_zlabel('Left Actuator - Blue')
        ax1.set_title('Left Eye Right Actuator')

    pass

if __name__ == "__main__":
    robot = oreo.Oreo_Robot(True, True, "/home/oreo/Documents/oreo_sim/oreo/sim", "assembly.urdf", True)
    robot.InitModel()
    print(robot.GetJointNames())
    robot.InitManCtrl()
    robot.RegKeyEvent(['c', 'q', 'p'])

    '''
    my_points = [[0.4, 0.0, 0.0], [0.5, 0.0, 0.0], [0.7, 0.0, 0.0], \
                 [0.4, 0.1, 0.0], [0.5, 0.1, 0.0], [0.7, 0.1, 0.0], \
                 [0.4, 0.5, 0.3], [0.5, 0.5, 0.3], [0.7, 0.5, 0.3], \
                 [0.4, -0.5, 0.3], [0.5, -0.5, 0.3], [0.7, -0.5, 0.3], \
                 [0.4, 0.5, -0.3], [0.5, 0.5, -0.3], [0.7, 0.5, -0.3], \
                 [0.4, -0.5, -0.3], [0.5, -0.5, -0.3], [0.7, -0.5, -0.3]]

    point_list = []
    for p in my_points:
        my_angles = robot.compute_yaw_pitch_for_given_point(p)
        if oreo.is_within_limits(my_angles):
            temp = p.copy()
            temp.append(my_angles)
            point_list.append(temp)
    '''

    l = 0
    #read table data from pickle file
    a = robot.read_oreo_yaw_pitch_actuator_data()
    if a == 0:
        print("Building scan data takes minutes ....")
        robot.build_oreo_scan_yaw_pitch_actuator_data()

    robot.get_max_min_yaw_pitch_values()
    # read the interpolator function from pickle file
    b = robot.read_interpolator_functions()
    if b == 0:
        robot.produce_interpolators()

    #robot.plot_interpolator_functions()
    #robot.compare_actuator_values()
    interp_val = generate_interpolated_actuator_values()
    plot_interpolated(interp_val)
    move_in_yaw(90.0)


    while (1):
        #robot.UpdManCtrl()
        #robot.UpdManCtrl_new()
        #robot.UpdManCtrl_test()
        if l < 1 :
            for p in pos_list:
                robot.move_eyes_to_pos(p)
                time.sleep(1.0)
            l +=1

        k = cv2.waitKey(0)
        if k == ord('q'):
            break


        keys = robot.GetKeyEvents()
        if 'c' in keys:
            robot.CheckAllCollisions()
        if 'p' in keys:
            robot.GetLinkPosOrn('neck_joint')
        if 'q' in keys:
            # quit
            break
        #robot.final_pose()
    robot.Cleanup()
