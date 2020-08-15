
import time
import oreo
import sys
import numpy as np
import cv2

pos_list = []

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
            return
    return 1

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
    move_in_yaw(70.0)

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
