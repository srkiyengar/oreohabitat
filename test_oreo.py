
import time
import oreo
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pybullet as p

import numpy as np
import quaternion

pos_list = []
angle_pos_list=[]


def test_IK(this_robot):
    link_name = "left_eye_joint"
    idx1 = this_robot.jointDict[link_name]
    pos = list(this_robot.initPosOrn[idx1][this_robot.POS_IDX])
    quat1_pybullet = [0.0,0.0,0.0871557,0.9961947]  # axis (0,0,1) angle 10 degrees
    my_joints_pos1 = p.calculateInverseKinematics(this_robot.linkage, idx1, pos, quat1_pybullet)

    quat2_pybullet = [0.0, 0.0, 0.1736482, 0.9848078]  # axis (0,0,1) angle 30 degrees
    my_joints_pos2 = p.calculateInverseKinematics(this_robot.linkage, idx1, pos, quat2_pybullet)

    quat3_pybullet = [0.0, 0.0, 0.258819, 0.9659258]  # axis (0,0,1) angle 30 degrees
    my_joints_pos3 = p.calculateInverseKinematics(this_robot.linkage, idx1, pos, quat3_pybullet)

    return [my_joints_pos1,my_joints_pos2,my_joints_pos3]


'''
Pybullet - the eyes are looking in the direction of +ive X, and Z is up.
Habitat - the eyes are looking in the direction of +ive Z and Y is up. 
Habitatai coordinate frame (x+ive,y+ive,z -ive) is Pybullet's frame (y-ive, z+ive, x+ive)
A point (x,y,z) in Habitat's frame when rotated by R will provide it in Pybullet's frame
It is the rotation matrix from Pybullet to Habitatai  row 0 [0 0 -1], row 1 [-1 0 0], row 2 [0,1,0]]
Habitat to PyBullet R_inverse row 0 = [0.-1,0], row 1 = [0,0,1], row 2 = [-1,0,0]
Given a point in Habitat, we do a rotation to pybullet, determine if there is collision and get 
the orientation of the eyes. This orientation has to be rotated using R_inverse.
'''

def move_in_yaw(pitch):
    yaw = np.linspace(-25, 25, 50)
    for i in yaw:
        my_angles =[i,pitch,i,pitch]
        actuator_pos = robot.get_actuator_positions_for_a_given_yaw_pitch(my_angles)
        if actuator_pos[0] == 1:
            my_pos = actuator_pos[1:]
            print("My yaw-pitch angles={} and actuator positions = {}".format(my_angles,my_pos))
            pos_list.append(my_pos)
            angle_pos_list.append(my_angles+my_pos)
        else:
            print("Actuator positions not computed")
    return 1


def move_x_and_z(x_val, y_mid, z_val):
    y_val = np.linspace(y_mid-2.0,y_mid+2.0,50)
    for i in y_val:
        view_point = [x_val, i, z_val]
        my_angles = robot.compute_yaw_pitch_for_given_point(view_point)
        actuator_pos = robot.get_actuator_positions_for_a_given_yaw_pitch(my_angles)
        if actuator_pos[0] == 1:
            my_pos = actuator_pos[1:]
            #print("My yaw-pitch angles={} and actuator positions = {}".format(my_angles, my_pos))
            pos_list.append(my_pos)
            angle_pos_list.append(list(my_angles) + my_pos)
        else:
            print("Actuator positions not computed")
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
    #yaw = np.linspace(-10, 0, 100)
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
    print("Bad Actuator values for LeftEye Left = {} Right = {}  RightEye Left = {} Right = {}".format(
        lefteye_l_outside_range, lefteye_r_outside_range, righteye_l_outside_range, righteye_r_outside_range
    ))
    return 1,lefteye_l, lefteye_r, righteye_l, righteye_r

def plot_interpolated_lefteye(interp_val):
    if interp_val[0] == 1:
        lefteye_l_interp = np.array(interp_val[1])

        pit = lefteye_l_interp[:, 0]
        yaw = lefteye_l_interp[:, 1]
        left_a = lefteye_l_interp[:, 2]
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.scatter(yaw, pit, left_a, 'z', c='coral')
        ax1.set_xlabel('Yaw in degrees')
        ax1.set_ylabel('Pitch in degrees')
        ax1.set_zlabel('Left Actuator - coral')
        ax1.set_title('Left Eye Left Actuator')
        plt.show()
        lefteye_r_interp = np.array(interp_val[2])
        pit = lefteye_r_interp[:, 0]
        yaw = lefteye_r_interp[:, 1]
        right_a = lefteye_r_interp[:, 2]
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.scatter(yaw, pit, right_a, 'z', c='blue')
        ax1.set_xlabel('Yaw in degrees')
        ax1.set_ylabel('Pitch in degrees')
        ax1.set_zlabel('Left Actuator - Blue')
        ax1.set_title('Left Eye Right Actuator')
    return
def plot_interpolated_righteye(interp_val):
    if interp_val[0] == 1:
        righteye_l_interp = np.array(interp_val[1])

        pit = righteye_l_interp[:, 0]
        yaw = righteye_l_interp[:, 1]
        left_a = righteye_l_interp[:, 2]
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.scatter(yaw, pit, left_a, 'z', c='red')
        ax1.set_xlabel('Yaw in degrees')
        ax1.set_ylabel('Pitch in degrees')
        ax1.set_zlabel('Left Actuator - red')
        ax1.set_title('Right Eye Left Actuator')
        plt.show()
        righteye_r_interp = np.array(interp_val[2])
        pit = righteye_r_interp[:, 0]
        yaw = righteye_r_interp[:, 1]
        right_a = righteye_r_interp[:, 2]
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.scatter(yaw, pit, right_a, 'z', c='green')
        ax1.set_xlabel('Yaw in degrees')
        ax1.set_ylabel('Pitch in degrees')
        ax1.set_zlabel('Left Actuator - green')
        ax1.set_title('Right Eye Right Actuator')
        plt.show()
    return
def test_joint_info (myrobot):
    num_of_joints = myrobot.numJoints
    print("Number of Joints = {}".format(num_of_joints))
    print("Joint Dictionary {}\n".format(myrobot.jointDict))
    for i in range(num_of_joints):
        jointInfo = p.getJointInfo(myrobot.linkage, i)
        print("****** Joint index {} Joint name = {} Link name {}".format(jointInfo[0],
                                    jointInfo[1].decode('UTF-8'), jointInfo[12].decode('UTF-8')))
        print("Joint type is {}".format(jointInfo[2]))
        print("Joint Lower Limit {} Upper Limit {}".format(jointInfo[8],jointInfo[9]))
        print("Position wrt to previous frame {}".format(jointInfo[14]))
        print("Orientation wrt to previous frame {}".format(jointInfo[15]))

        state = p.getLinkState(myrobot.linkage, i, computeForwardKinematics=True)
        print("Position {} Orientation {}".format(state[4],state[5]))

        print("XXXXXXX End of joint information for {}\n".format(jointInfo[1].decode('UTF-8')))

    new_direction = np.array([0.5,0.5,0.0])
    new_direction = new_direction/np.linalg.norm(new_direction)
    myrobot.compute_IK_for_actuators(new_direction)

    return

if __name__ == "__main__":
    robot = oreo.Oreo_Robot(True, True, "./", "assembly.urdf", True)
    #robot = oreo.Oreo_Robot(True, True, "/home/oreo/Documents/oreo_sim/oreo/sim", "assembly.urdf", True)
    robot.InitModel()
    print(robot.GetJointNames())
    robot.InitManCtrl()
    robot.RegKeyEvent(['c', 'q', 'p'])

    #test_joint_info(robot)
    #my_results = test_IK(robot)
    #read table data from pickle file

    a = robot.read_oreo_yaw_pitch_actuator_data()
    if a == 0:
        print("Building scan data takes minutes ....")
        robot.build_oreo_scan_yaw_pitch_actuator_data()

    #robot.plot_scan_data()
    #robot.get_max_min_yaw_pitch_values()
    # read the interpolator function from pickle file
    b = robot.read_interpolator_functions()
    if b == 0:
        robot.produce_interpolators()

    '''
    #test
    my_point = [9.0,0.0,2.0]
    the_angles = robot.compute_yaw_pitch_for_given_point(my_point)
    val = robot.get_actuator_positions_for_a_given_yaw_pitch(the_angles)
    if val[0] == 1:
        a_pos = val[1:]
        collision = robot.move_eyes_to_pos(a_pos)
        if collision == 0:
            orn_lefteye = robot.GetLinkOrientationWCS("left_eye_joint")
            # convert to numpy quaternion (w,x,y,z) w is the real part.
            orientation_lefteye = np.quaternion(orn_lefteye[3], orn_lefteye[0], orn_lefteye[1], orn_lefteye[2])
            orn_righteye = robot.GetLinkOrientationWCS("right_eye_joint")
            # convert to numpy quaternion (w,x,y,z) w is the real part.
            orientation_righteye = np.quaternion(orn_righteye[3], orn_righteye[0], orn_righteye[1], orn_righteye[2])
    # end of test
    '''
    #robot.plot_interpolator_datapoints()
    #robot.compare_actuator_values()
    #interp_val = generate_interpolated_actuator_values()
    #plot_interpolated_lefteye(interp_val)
    #plot_interpolated_righteye(interp_val)
    #move_in_yaw(120.0)
    move_x_and_z(9.0,0.0,2.0)

    while (1):
        #robot.UpdManCtrl()
        #robot.UpdManCtrl_new()
        #robot.UpdManCtrl_test()


        for i in angle_pos_list:
            #robot.move_eyes_to_pos(i[4:8])
            collide, l_orn, right_orn = robot.move_eyes_to_position_and_return_orn(i[4:8])
            if collide == 0:
                i.append(l_orn)
                i.append(right_orn)
            
            #q, state = robot.GetLinkOrientationWCS_test("left_eye_joint")
            #print("quat = {}".format(q))
            #print("Angles = {}".format(i[:4]))
            #print("Angles = {} - Rotation of Left eye q {}".format(i[:4],q))
            #print("State = {}".format(state))


        keys = robot.GetKeyEvents()
        if 'c' in keys:
            robot.CheckAllCollisions()
        if 'p' in keys:
            robot.GetLinkPosOrn('neck_joint')
        if 'q' in keys:
            quit
            break
        #robot.final_pose()

    robot.Cleanup()
