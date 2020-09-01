import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys

# reads the pickled data
def read_oreo_yaw_pitch_actuator_data():
    global left_eye_scan_data, right_eye_scan_data
    try:
        with open("eye_scan_data.pkl", "rb") as f:
            my_data = pickle.load(f)
            # left and right eye [left_actuator,right_actuator,yaw,pitch]
            left_eye_scan_data = my_data[0]
            right_eye_scan_data = my_data[1]
            return 1
    except IOError as e:
        print("Failure: Opening pickle file {}".format("eye_scan_data.pkl"))
        return 0

# reads pickled function
def read_interpolator_functions():
    global left_eye_interpolator_left, left_eye_interpolator_right, right_eye_interpolator_left, right_eye_interpolator_right
    try:
        with open("interp_file.pkl", "rb") as f:
            my_interp = pickle.load(f)
            # left and right eye [left_actuator,right_actuator,yaw,pitch]
            left_eye_interpolator_left = my_interp[0]
            left_eye_interpolator_right = my_interp[1]
            right_eye_interpolator_left = my_interp[2]
            right_eye_interpolator_right = my_interp[3]
            return 1
    except IOError as e:
        print("Failure: Opening pickle file {}".format("interp_file.pkl"))
        return 0


def plot_interpolator_datapoints():
    # left eye
    my_lefteye_table = np.array(left_eye_scan_data)
    x = my_lefteye_table[:, 2]  # all yaw
    y = my_lefteye_table[:, 3]  # all pitch

    z_left = []
    z_right = []
    for xi, yi in zip(x, y):
        zi1 = left_eye_interpolator_left(xi, yi)[0]
        zi2 = left_eye_interpolator_right(xi, yi)[0]
        z_left.append(zi1)
        z_right.append(zi2)

    zpl = []
    zpr = []
    xp = []
    yp = []
    i = 0
    for j, k in zip(z_left, z_right):
        if (-0.015 < j < +0.015) and ((-0.015 < k < +0.015)):
            xp.append(x[i])
            yp.append(y[i])
            zpl.append(j)
            zpr.append(k)
        i += 1

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter(xp, yp, zpl, 'z', c='coral', )
    ax1.set_xlabel('Yaw in degrees')
    ax1.set_ylabel('Pitch in degrees')
    ax1.set_zlabel('Left Actuator - coral')
    ax1.set_title('Left Eye Left Actuator - Data points')
    plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter(xp, yp, zpr, 'z', c='blue', )
    ax1.set_xlabel('Yaw in degrees')
    ax1.set_ylabel('Pitch in degrees')
    ax1.set_zlabel('Right Actuator - blue')
    ax1.set_title('Left Eye Right Actuator - Data points')
    plt.show()
    # end of left eye

    # right eye
    my_righteye_table = np.array(right_eye_scan_data)
    x = my_righteye_table[:, 2]  # all yaw
    y = my_righteye_table[:, 3]  # all pitch
    z_left = []
    z_right = []
    for xi, yi in zip(x, y):
        zi1 = right_eye_interpolator_left(xi, yi)[0]
        z_left.append(zi1)
        zi2 = right_eye_interpolator_right(xi, yi)[0]
        z_right.append(zi2)

    zpl = []
    zpr = []
    xp = []
    yp = []
    i = 0
    for j, k in zip(z_left, z_right):
        if (-0.015 < j < +0.015) and ((-0.015 < k < +0.015)):
            xp.append(x[i])
            yp.append(y[i])
            zpl.append(j)
            zpr.append(k)
        i += 1

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter(xp, yp, zpl, 'z', c='red', )
    ax1.set_xlabel('Yaw in degrees')
    ax1.set_ylabel('Pitch in degrees')
    ax1.set_zlabel('Left Actuator - red')
    ax1.set_title('Right Eye Left Actuator - Data points')
    plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter(xp, yp, zpr, 'z', c='green', )
    ax1.set_xlabel('Yaw in degrees')
    ax1.set_ylabel('Pitch in degrees')
    ax1.set_zlabel('Right Actuator - green')
    ax1.set_title('Right Eye Right Actuator - Data points')
    plt.show()
    # end of 3D interpolated
    return

def get_actuator_positions_for_a_given_yaw_pitch(my_angles):
    if (left_eye_interpolator_left is None or left_eye_interpolator_right is None or \
            right_eye_interpolator_left is None or right_eye_interpolator_right is None):
        return [0, 0.0, 0.0, 0.0, 0.0]  # 0 followed by 4 floats

    left_actuator_pos_lefteye = left_eye_interpolator_left(my_angles[0], my_angles[1])[0]
    right_actuator_pos_lefteye = left_eye_interpolator_right(my_angles[0], my_angles[1])[0]

    left_actuator_pos_righteye = right_eye_interpolator_left(my_angles[2], my_angles[3])[0]
    right_actuator_pos_righteye = right_eye_interpolator_right(my_angles[2], my_angles[3])[0]
    return [1, left_actuator_pos_lefteye, right_actuator_pos_lefteye, left_actuator_pos_righteye,
            right_actuator_pos_righteye]

def generate_interpolated_actuator_values():
    lefteye_l = []
    lefteye_r = []
    righteye_l = []
    righteye_r = []

    lefteye_l_outside_range = 0
    lefteye_r_outside_range = 0
    righteye_l_outside_range = 0
    righteye_r_outside_range = 0

    #yaw = np.linspace(-25, 25, 100)
    #yaw = np.linspace(-10, 0, 100)
    yaw = np.linspace(-15.0, -5.0, 100)
    pitch = np.linspace(75,125,100)



    for i in pitch:
        for j in yaw:
            my_angles = [j,i,j,i]
            actuator_pos = get_actuator_positions_for_a_given_yaw_pitch(my_angles)
            if actuator_pos[0] == 1:
                if (-0.05<actuator_pos[1]<0.05):
                    lefteye_l.append([i,j,actuator_pos[1]])
                else:
                    lefteye_l_outside_range += 1

                if (-0.05 < actuator_pos[2] < 0.05):
                    lefteye_r.append([i, j, actuator_pos[2]])
                else:
                    lefteye_r_outside_range += 1

                if (-0.05<actuator_pos[3]<0.05):
                    righteye_l.append([i,j,actuator_pos[3]])
                else:
                    righteye_l_outside_range += 1

                if (-0.05 < actuator_pos[4] < 0.05):
                    righteye_r.append([i, j, actuator_pos[4]])
                else:
                    righteye_r_outside_range += 1

            else:
                print("Interpolator functions not computed")
                return 0,
    '''
    print("Bad Actuator values for LeftEye Left = {} Right = {}  RightEye Left = {} Right = {}".format(
        lefteye_l_outside_range, lefteye_r_outside_range, righteye_l_outside_range, righteye_r_outside_range
    ))
    '''
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
        ax1.set_title('Left Eye Left Actuator - Interpolated points')
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
        ax1.set_title('Left Eye Right Actuator - Interpolated points')
        plt.show()

    pass
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
        ax1.set_title('Right Eye Left Actuator - Interpolated points')
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
        ax1.set_title('Right Eye Right Actuator - Interpolated points')
        plt.show()


if __name__ == "__main__":
    print("The system version is {}".format(sys.version))
    read_oreo_yaw_pitch_actuator_data()         # reads the datapoints from pickled file "eye_scan_data.pkl"
    read_interpolator_functions()               # reads the interpolated functions from pickled file interp_file.pkl
    #plot_interpolator_datapoints()              # plots the interpolated function for the data points

    interp_val = generate_interpolated_actuator_values()    # generated interpolated values in the range of interest
    plot_interpolated_lefteye(interp_val)                   # plots both actuators for left eye
    plot_interpolated_righteye(interp_val)                  # right eye

    pass