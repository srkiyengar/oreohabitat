import pybullet as p
import time
import logging
import numpy as np
import quaternion
import pickle
from scipy import interpolate
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

yaw_lower_limit = -27.0
yaw_upper_limit = 41.0
pitch_lower_limit = 48.0
pitch_upper_limit = 128

def is_within_limits(angles):
    # angles is a tuple in the order yaw_lefteye, pitch_lefteye, yaw_righteye, pitch_righteye
    global yaw_lower_limit, yaw_upper_limit, pitch_lower_limit, pitch_upper_limit
    if yaw_lower_limit <= angles[0] <= yaw_upper_limit:
        if pitch_lower_limit <= angles[1] <= pitch_upper_limit:
            if yaw_lower_limit <= angles[2] <= yaw_upper_limit:
                if pitch_lower_limit <= angles[3] <= pitch_upper_limit:
                    print("$$$$$ All angle within limits")
                    return True
                else:
                    print("*** Not within limits - Righteye Pitch = {} ".format(angles[3]))
                    return False
            else:
                print("*** Not within limits - Righteye yaw = {}".format(angles[2]))
                return False
        else:
            print("*** Not within limits - Lefteye Pitch = {}".format(angles[1]))
            return False
    else:
        print("*** Not within limits - Lefteye yaw = {}".format(angles[0]))
        return False

def compute_yaw_pitch_from_vector(uvector):
    rx = uvector[0]
    ry = uvector[1]
    rz = uvector[2]

    pitch = np.degrees(np.arccos(rz))
    rxy = np.sqrt(rx * rx + ry * ry)
    if rxy != 0:
        yaw = np.degrees(np.arcsin(ry/rxy))
    else:
        yaw = 0
    return (yaw, pitch)


class Oreo_Robot(object):
    # Class which controls the oreo robot
    # All directions (far_left etc. are from robot's perspective (looking in +ve x-dir))
    # Member Vars
    linkage = 0
    physicsClient = 0
    numJoints = 0
    jointDict = {}  # map joint name to id
    linkJoints = []  # map link id to all its joints/constraints
    LINK_PARENT_IDX = 0  # joint ids where link is parent
    LINK_CHILD_IDX = 1  # joint ids where link is child
    initPosOrn = []
    POS_IDX = 0
    ORN_IDX = 1
    useRealTime = False
    linkVelo = []
    VELO_IDX = 0
    ANG_VELO_IDX = 1

    numConstraints = 4
    constraintDict = {}
    constraintLinks = [
        ['left_eye_joint', 'dogbone_joint_far_left', 'constraint_far_left'],
        ['left_eye_joint', 'dogbone_joint_mid_left', 'constraint_mid_left'],
        ['right_eye_joint', 'dogbone_joint_mid_right', 'constraint_mid_right'],
        ['right_eye_joint', 'dogbone_joint_far_right', 'constraint_far_right'],
    ]
    CONS_PARENT_IDX = 0
    CONS_CHILD_IDX = 1
    CONS_NAME_IDX = 2
    constraintParentPos = [[0.015, -0.0016, 0.0245], [0.0126, -0.0257, 0.0095], [0.0134, 0.0025, 0.0262],
                           [0.0143, -0.0224, 0.0124]]  # pos on eye
    constraintChildPos = [30.25e-3, 0, 0]  # pos on dogbone
    constraintAxis = [0, 0, 0]
    constraintType = p.JOINT_POINT2POINT

    # Joints modelled with motor
    actJointNames = ["neck_joint", "pitch_piece_joint", "skull_joint", "linear_motor_rod_joint_far_left",
                     "linear_motor_rod_joint_mid_left", \
                     "linear_motor_rod_joint_mid_right", "linear_motor_rod_joint_far_right"]
    actJointIds = []
    actJointPos = []
    actJointNum = 0
    actJointHome = 0
    prismaticLim = [-0.1, 0.1]
    revoluteLim = [-1.57, 1.57]
    LIM_MIN_IDX = 0
    LIM_MAX_IDX = 1
    manCtrl = []
    actJointControl = p.POSITION_CONTROL
    prev_pos = []   #rajan - to hold previous values of joints

    # "Dumb" joints with no actuation
    dumbJointNames = ["left_eye_yolk_joint", "left_eye_joint", "right_eye_yolk_joint", "right_eye_joint"]
    dumbJointHome = 0
    dumbJointIds = 0
    dumbJointNum = 0
    dumbJointVelo = 0
    dumbJointControl = p.VELOCITY_CONTROL
    dumbJointForce = 0

    # "Spherical" joints (no actuation)
    spherJointNames = ["dogbone_joint_far_left", "dogbone_joint_mid_left", "dogbone_joint_mid_right",
                       "dogbone_joint_far_right"]
    spherJointHome = [[0, 0, 0.00427], [0, 0, 0.00427], [0, 0, 0.00427], [0, 0, -0.00427], [0, 0, -0.00427]]
    spherJointIds = []
    spherJointNum = 0
    spherJointPos = [0]
    spherJointVelo = [0, 0, 0]
    spherJointControl = p.POSITION_CONTROL
    spherJointKp = 0
    spherJointKv = 1
    spherJointForce = [0, 0, 0]

    # Links which have collisions disabled
    disCollisionLinks = [
        ['dogbone_joint_far_left', 'left_eye_joint'],
        ['dogbone_joint_mid_left', 'left_eye_joint'],
        ['dogbone_joint_mid_right', 'right_eye_joint'],
        ['dogbone_joint_far_right', 'right_eye_joint'],
        ['dogbone_joint_far_left', 'left_eye_yolk_joint'],
        ['dogbone_joint_mid_left', 'left_eye_yolk_joint'],
        ['dogbone_joint_mid_right', 'right_eye_yolk_joint'],
        ['dogbone_joint_far_right', 'right_eye_yolk_joint'],
    ]

    # Links with collisions enabled
    toggCollisionLinks = [
        ['left_eye_joint', 'right_eye_joint'],
        ['left_eye_yolk_joint', 'right_eye_yolk_joint'],
        ['left_eye_yolk_joint', 'right_eye_joint'],
        ['right_eye_yolk_joint', 'left_eye_joint'],
        ['skull_joint', 'pitch_piece_joint'],
        ['dogbone_joint_far_right', 'pitch_piece_joint'],
        ['dogbone_joint_far_left', 'pitch_piece_joint']

    ]

    # Registered key events to look for
    keys = []

    # Constants
    INIT_POS = [0, 0, 0]
    INIT_ORN = [0, 0, 0]
    CONSTRAINT_MAX_FORCE = 10000000
    JOINT_MAX_FORCE = 100
    TORQUE_CONTROL = 0
    POSITION_CONTROL = 1
    TIME_STEP = 1 / 240
    DYN_STATE_SIZE = 6
    GRAV_ACCELZ = -9.8

    k = 0
    oreo_scan_data = "eye_scan_data.pkl"    # holds the pickled data for left and right acutator positions
    left_eye_scan_data = []                 # unpickled from the pickle file - left eye
    right_eye_scan_data = []                # unpickled from the pickle file - right eye
    left_eye_interpolator_left = None       # left actuator position interpolator function for left eye
    left_eye_interpolator_right = None      # right actuator position interpolator function for left eye
    right_eye_interpolator_left = None      # left actuator position interpolator function for right eye
    right_eye_interpolator_right = None     # right actuator position interpolator function for right eye

    interpolator_pickle_file = "interp_file.pkl"
    interp_functions = []

    # Constructor
    def __init__(self, enableDebug, enableGUI, urdfPath, urdfName, enableRealTime):
        # Init logging
        if (enableDebug):
            logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(message)s')
            logging.info('LOG_LEVEL: DEBUG')
        else:
            logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
            logging.info('LOG_LEVEL: INFO')

        # Setup environment
        if (enableGUI):
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(urdfPath)
        urdf_flags = p.URDF_USE_INERTIA_FROM_FILE | p.URDF_MAINTAIN_LINK_ORDER | p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        self.linkage = p.loadURDF(urdfName, self.INIT_POS, p.getQuaternionFromEuler(self.INIT_ORN), useFixedBase=1,
                                  flags=urdf_flags)
        self.useRealTime = enableRealTime


    # Go to home position
    def HomePos(self):
        p.setRealTimeSimulation(0)
        # Reset dumb joint positions
        for dumb_idx in range(self.dumbJointNum):
            p.resetJointState(self.linkage, self.dumbJointIds[dumb_idx], targetValue=self.dumbJointHome)

        # Reset actuated joint positions
        for act_idx in range(self.actJointNum):
            p.resetJointState(self.linkage, self.actJointIds[act_idx], targetValue=self.actJointHome)

        # Update position list
        self.actJointPos = [self.actJointHome] * self.actJointNum

        # Reset spherical joints
        for spher_idx in range(self.spherJointNum):
            p.resetJointStateMultiDof(self.linkage, self.spherJointIds[spher_idx],
                                      targetValue=self.spherJointHome[spher_idx])

    # Enable/Disable collision
    def ToggleCollision(self, enable):
        # Turn off some collisions
        for j in range(len(self.disCollisionLinks)):
            p.setCollisionFilterPair(self.linkage, self.linkage, self.jointDict[self.disCollisionLinks[j][0]],
                                     self.jointDict[self.disCollisionLinks[j][1]], 0)

        if enable > 0:
            logging.debug("Collision enabled")
            enable = 1
        else:
            logging.debug("Collision disabled")
            enable = 0
        for i in range(len(self.toggCollisionLinks)):
            p.setCollisionFilterPair(bodyUniqueIdA=self.linkage, bodyUniqueIdB=self.linkage,
                                     linkIndexA=self.jointDict[self.toggCollisionLinks[i][0]],
                                     linkIndexB=self.jointDict[self.toggCollisionLinks[i][1]], enableCollision=1)
            if enable > 0:
                logging.debug("Collision pair: %s %s", self.toggCollisionLinks[i][0], self.toggCollisionLinks[i][1])
        print()
        print()

    # Toggle Sim type
    def SetSimType(self, realTime):
        # Set sim type
        if realTime:
            self.useRealTime = True
            p.setRealTimeSimulation(1)
            logging.debug("Setting real time sim")
        else:
            self.useRealTime = False
            p.setRealTimeSimulation(0)
            p.setTimeStep(self.TIME_STEP)
            logging.debug("Setting step sim with timestep %f", self.TIME_STEP)

    # Reset actuated joint control
    def ResetActJointControl(self):
        # Must set to velocity control with 0 velocity and 0 force to "reset"
        p.setJointMotorControlArray(self.linkage, self.actJointIds, p.VELOCITY_CONTROL, forces=[0] * self.actJointNum)

    # Initialize robot for position control
    def InitPosnCtrl(self):
        self.SetActJointControlType(self.POSITION_CONTROL)
        self.ResetActJointControl()
        self.ControlActJoints([0] * self.actJointNum)

    # Initialize robot for torque control
    def InitTorqueCtrl(self):
        self.SetActJointControlType(self.TORQUE_CONTROL)
        self.ResetActJointControl()
        self.ControlActJoints([0] * self.actJointNum)

        # Set actuated joint control type

    def SetActJointControlType(self, ctrl_type):
        if ctrl_type == self.TORQUE_CONTROL:
            self.actJointControl = p.TORQUE_CONTROL
            logging.debug("Set torque control")
            # Must be step sim for torque control
            self.SetSimType(False)
        else:
            self.actJointControl = p.POSITION_CONTROL
            logging.debug("Set position control")

    # Set joint control for actuators
    def ControlActJoints(self, control):
        target = []
        links = []
        if isinstance(control, list):
            for idx in range(len(control)):
                target.append(control[idx])
            links = self.actJointIds
        elif isinstance(control, dict):
            for name in control:
                if name in self.actJointNames:
                    target.append(control[name])
                    links.append(self.jointDict[name])
                else:
                    logging.warning("Unknown dict key in control input for ControlActJoints")
        else:
            logging.error("Unknown control input type")

        if self.actJointControl == p.POSITION_CONTROL:
            pass
            p.setJointMotorControlArray(self.linkage, links, self.actJointControl, targetPositions=target,
                                        forces=[self.JOINT_MAX_FORCE] * len(target))
        elif self.actJointControl == p.TORQUE_CONTROL:
            p.setJointMotorControlArray(self.linkage, links, self.actJointControl, forces=target)

        # Update simulation
        if self.useRealTime == False:
            p.stepSimulation()

        return 0

    # Set joint control for dumb joints
    def ControlDumbJoints(self):
        p.setJointMotorControlArray(self.linkage, self.dumbJointIds, self.dumbJointControl,
                                    targetVelocities=[self.dumbJointVelo] * self.dumbJointNum,
                                    forces=[self.dumbJointForce] * self.dumbJointNum)

    # Set joint control for spherical joints
    def ControlSpherJoints(self):
        p.setJointMotorControlMultiDofArray(self.linkage, self.spherJointIds, self.spherJointControl,
                                            targetPositions=[self.spherJointPos] * self.spherJointNum,
                                            targetVelocities=[self.spherJointVelo] * self.spherJointNum,
                                            forces=[self.spherJointForce] * self.spherJointNum,
                                            positionGains=[self.spherJointKp] * self.spherJointNum,
                                            velocityGains=[self.spherJointKv] * self.spherJointNum)

    # Init manual control
    def InitManCtrl(self):
        self.InitPosnCtrl()
        for idx in range(self.actJointNum):
            name = self.actJointNames[idx]
            if "linear" in name:
                self.manCtrl.append(p.addUserDebugParameter(name, self.prismaticLim[self.LIM_MIN_IDX],
                                                            self.prismaticLim[self.LIM_MAX_IDX]))
            else:
                self.manCtrl.append(p.addUserDebugParameter(name, self.revoluteLim[self.LIM_MIN_IDX],
                                                            self.revoluteLim[self.LIM_MAX_IDX]))


    # Update manual control
    def UpdManCtrl(self):
        pos = [0] * self.actJointNum
        for idx in range(len(self.manCtrl)):
            pos[idx] = p.readUserDebugParameter(self.manCtrl[idx])

        if self.k == 0:
            for i in range(self.numJoints):
                my_joint = p.getJointInfo(self.linkage, i)
                my_joint_name = my_joint[1]
                my_joint_id = my_joint[0]
                state = p.getLinkState(self.linkage, my_joint_id, computeForwardKinematics=True)
                position = state[4]
                orientation = state[5]
                print("Link {} Position {} and Orientation {}".format(my_joint_name, position, orientation))

            self.k = 1
            '''
            position = self.GetLinkPosOrn("left_eye_joint")[0]
            orientation = self.GetLinkPosOrn("left_eye_joint")[1]
            print("Slider K0 {} Left eye Joint Position {} - Orientation - {}".format(pos, position, orientation))
            self.k = 1
            print("in first loop k={}".format(self.k))
            '''
        '''
        elif 8 > self.k > 0:
            pos[3] = (self.k)*0.003
            time.sleep(3)
            self.k+=1
            print("In second loop k={}".format(self.k))
        '''
        #printing when pos has changed
        if pos != self.prev_pos:
            self.prev_pos = pos.copy()
            self.ControlActJoints(pos)
            self.actJointPos = pos
            position = self.GetLinkPosOrn("left_eye_joint")[0]
            orientation = self.GetLinkPosOrn("left_eye_joint")[1]
            print("From Slider: {} Left eye Joint Orientation - {}".format(pos, orientation))
            if len(p.getContactPoints()) != 0:
                print("collision at {}".format(pos))
            else:
                print("No collision for {}".format(pos))

        #self.ControlActJoints(pos)
        #self.actJointPos = pos



        #self.k = 1
        #end of printing when pos has changed.

#****
        # Update manual control
    def UpdManCtrl_test(self):

        pos = [0] * self.actJointNum
        for idx in range(len(self.manCtrl)):
            pos[idx] = p.readUserDebugParameter(self.manCtrl[idx])

        # printing when pos has changed
        if pos != self.prev_pos:
            self.prev_pos = pos.copy()
            print("Actuator Positions = {}".format(pos))
            self.ControlActJoints(pos)
            self.actJointPos = pos
            time.sleep(0.1)
            # Left eye
            orn_lefteye = self.GetLinkOrientationWCS("left_eye_joint")
            # convert to numpy quaternion (w,x,y,z) w is the real part.
            orientation_lefteye = np.quaternion(orn_lefteye[3], orn_lefteye[0], orn_lefteye[1], orn_lefteye[2])
            my_rot_matrix_left = quaternion.as_rotation_matrix(orientation_lefteye)
            lefteye_x_uvector = my_rot_matrix_left[:, 0]
            ik_joint_info_lefteye = self.compute_IK_for_actuators(lefteye_x_uvector)

            # Right eye
            orn_righteye = self.GetLinkOrientationWCS("right_eye_joint")  # as a list in [x,y,z,w] order
            # convert tonge =  numpy array quaternion (w,x,y,z) - w is the real part
            orientation_righteye = np.quaternion(orn_righteye[3], orn_righteye[0], orn_righteye[1], orn_righteye[2])
            my_rot_matrix_right = quaternion.as_rotation_matrix(orientation_righteye)
            righteye_x_uvector = my_rot_matrix_right[:, 0]
            ik_joint_info_righteye = self.compute_IK_for_actuators(righteye_x_uvector)

            points = p.getContactPoints()
            if len(points) != 0:
                print("Collision for {}".format(pos))
            else:
                print(" No collision".format(pos))

    def compute_IK_for_actuators(self, uvector):
        v1 = np.array(uvector)
        v2 = np.array([1.0, 0.0, 0.0])
        # my_axis is v1 cross v2
        my_axis = np.array([0.0, -uvector[2], uvector[1]])
        my_angle = np.arccos(np.dot(v1, v2))
        my_axis_angle = my_angle * my_axis
        my_rot_quat = (quaternion.as_float_array(quaternion.from_rotation_vector(my_axis_angle))).tolist()
        quat_pybullet = my_rot_quat[1:]
        quat_pybullet.append(my_rot_quat[0])
        idx1 = self.jointDict["left_eye_joint"]
        my_joints = p.calculateInverseKinematics(self.linkage, idx1, self.initPosOrn[idx1][self.POS_IDX],
                                                 quat_pybullet)
        return my_joints

        # ****

    # Update manual control
    def UpdManCtrl_new(self):

        pos = [0] * self.actJointNum
        '''
        pos[3] = -0.014
        pos[4] = -0.012
        pos[5] = 0.015
        pos[6] = 0.013
        '''
        pos[3] = -0.0170896  # -0.017, -0.014
        pos[4] = -0.0156  # -0.017, -0.015
        # pos[5] = 0.1766523#0.035, -0.022
        # pos[6] = -0.00883694#0.020, -0.016

        if self.k == 0:
            print("Actuator Positions = {}".format(pos))
            self.ControlActJoints(pos)
            self.actJointPos = pos
            time.sleep(0.1)
            # Left eye
            orn_lefteye = self.GetLinkOrientationWCS("left_eye_joint")
            # convert to numpy quaternion (w,x,y,z) w is the real part.
            orientation_lefteye = np.quaternion(orn_lefteye[3], orn_lefteye[0], orn_lefteye[1], orn_lefteye[2])
            my_axis_angle_lefteye = quaternion.as_rotation_vector(orientation_lefteye)
            rotation_angle_lefteye = np.linalg.norm(my_axis_angle_lefteye)
            rotation_axis_lefteye = my_axis_angle_lefteye / rotation_angle_lefteye
            my_angles_left = np.degrees(quaternion.as_spherical_coords(orientation_lefteye))
            my_rot_matrix_left = quaternion.as_rotation_matrix(orientation_lefteye)

            # Right eye
            orn_righteye = self.GetLinkOrientationWCS("right_eye_joint")  # as a list in [x,y,z,w] order
            # convert tonge =  numpy array quaternion (w,x,y,z) - w is the real part
            orientation_righteye = np.quaternion(orn_righteye[3], orn_righteye[0], orn_righteye[1], orn_righteye[2])
            my_axis_angle_righteye = quaternion.as_rotation_vector(orientation_righteye)
            rotation_angle_righteye = np.linalg.norm(my_axis_angle_righteye)
            rotation_axis_righteye = my_axis_angle_righteye / rotation_angle_righteye
            my_angles_right = np.degrees(quaternion.as_spherical_coords(orientation_righteye))
            my_rot_matrix_right = quaternion.as_rotation_matrix(orientation_righteye)

            print("UpdManCtrl_new: Right Angles {} - Left angles {}".format(my_angles_right, my_angles_left))
            points = p.getContactPoints()
            if len(points) != 0:
                print("Collision for {}")
            else:
                print(" No collision".format(pos))

            self.k = 1

    ##
            # Update manual control
    def move_eyes_to_pos(self,new_pos):
        pos = [0] * self.actJointNum
        pos[3] = new_pos[0]
        pos[4] = new_pos[1]
        pos[5] = new_pos[2]
        pos[6] = new_pos[3]

        print("Move Eyes To Position: Actuator Positions = {}".format(pos))
        self.ControlActJoints(pos)
        self.actJointPos = pos
        time.sleep(0.1)
        collide = len(p.getContactPoints())
        orn_lefteye = self.GetLinkOrientationWCS("left_eye_joint")
        # convert to numpy quaternion (w,x,y,z) w is the real part.
        orientation_lefteye = np.quaternion(orn_lefteye[3], orn_lefteye[0], orn_lefteye[1], orn_lefteye[2])
        my_rot_matrix_left = quaternion.as_rotation_matrix(orientation_lefteye)
        yp1 = compute_yaw_pitch_from_vector(my_rot_matrix_left[:, 0])
        # Right eye
        orn_righteye = self.GetLinkOrientationWCS("right_eye_joint")  # as a list in [x,y,z,w] order
        # convert to numpy array quaternion (w,x,y,z) - w is the real part
        orientation_righteye = np.quaternion(orn_righteye[3], orn_righteye[0], orn_righteye[1],
                                             orn_righteye[2])
        my_rot_matrix_right = quaternion.as_rotation_matrix(orientation_righteye)
        yp2 = compute_yaw_pitch_from_vector(my_rot_matrix_right[:, 0])
        print("Move Eyes To Position: Left Eye Angles {} - Right Eye angles {} collision {}".format(yp1, yp2,collide))

    ##

    # final_pose
    def final_pose(self):
        if 50 >self.k>=1:
            orn_lefteye = self.GetLinkOrientationWCS("left_eye_joint")
            # convert to numpy quaternion (w,x,y,z) w is the real part.
            orientation_lefteye = np.quaternion(orn_lefteye[3], orn_lefteye[0], orn_lefteye[1], orn_lefteye[2])
            #my_axis_angle_lefteye = quaternion.as_rotation_vector(orientation_lefteye)
            #rotation_angle_lefteye = np.linalg.norm(my_axis_angle_lefteye)
            #rotation_axis_lefteye = my_axis_angle_lefteye / rotation_angle_lefteye
            #my_angles_left = np.degrees(quaternion.as_spherical_coords(orientation_lefteye))
            my_rot_matrix_left = quaternion.as_rotation_matrix(orientation_lefteye)
            my_left_computed = compute_yaw_pitch_from_vector(my_rot_matrix_left[:,0])

            # Right eye
            orn_righteye = self.GetLinkOrientationWCS("right_eye_joint")  # as a list in [x,y,z,w] order
            # convert tonge =  numpy array quaternion (w,x,y,z) - w is the real part
            orientation_righteye = np.quaternion(orn_righteye[3], orn_righteye[0], orn_righteye[1], orn_righteye[2])
            #my_axis_angle_righteye = quaternion.as_rotation_vector(orientation_righteye)
            #rotation_angle_righteye = np.linalg.norm(my_axis_angle_righteye)
            #rotation_axis_righteye = my_axis_angle_righteye / rotation_angle_righteye
            #my_angles_right = np.degrees(quaternion.as_spherical_coords(orientation_righteye))
            my_rot_matrix_right = quaternion.as_rotation_matrix(orientation_righteye)
            my_right_computed = compute_yaw_pitch_from_vector(my_rot_matrix_right[:, 0])

            #print("final_pose: k ={} right angles {} - left angles {}".format(self.k,my_angles_right, my_angles_left))
            print("final_pose Com: k ={} right angles {} - left angles {}".format(self.k, my_right_computed, my_left_computed))
            self.k +=1
        else:
            pass



    #****

    # Generate table of mid and far actuators positions and the corresponding yaw and pitch values.
    # record if there is collision

    def generate_actuator_positions(self, left, right, oreo_eye):
        # right, left are actuator pairs 3,4 or 5,6 - Right and left from oreo's perspective (not viewer)
        # 3,4 control yaw and pitch of oreo's left eye  while 5,6 for the right eye
        # orea_eye can be either " left_eye_joint" or "right_eye_joint"
        total_count = 0
        contact_count = 0
        contactless_count = 0
        my_value_l = np.linspace(-0.03,0.03, 100)
        my_value_r = np.linspace(-0.025,0.025,100)
        my_table = []
        pos = [0] * self.actJointNum
        wait_time = 0.2
        for left_actuator in my_value_l:
            pos[left] = left_actuator
            for right_actuator in my_value_r:
                total_count += 1
                pos[right] = right_actuator
                self.ControlActJoints(pos)
                self.actJointPos = pos
                time.sleep(wait_time)
                # specified eye
                eye_orientation = self.GetLinkOrientationWCS(oreo_eye)
                # convert to numpy quaternion (w,x,y,z) w is the real part.
                eye_orientation_quat = np.quaternion(eye_orientation[3], eye_orientation[0],eye_orientation[1], \
                                                     eye_orientation[2])
                my_rot_matrix = quaternion.as_rotation_matrix(eye_orientation_quat)
                computed_yaw, computed_pitch = compute_yaw_pitch_from_vector(my_rot_matrix[:, 0])
                my_table.append([left_actuator, right_actuator, computed_yaw, computed_pitch])
                if len(p.getContactPoints()) != 0:
                    contact_count += 1
                else:
                    contactless_count += 1
                    my_table.append([left_actuator, right_actuator, computed_yaw, computed_pitch])

        print("Total count = {}".format(total_count))
        print("Total contact less count = {}".format(contactless_count))
        print("Total contact count = {}".format(contact_count))
        with open(oreo_eye, "w") as f:
            for my_line in my_table:
                f.write(("left A ={} Right A ={} Com_yaw = {} Com_pitch = {}\n".format(my_line[0], \
                                                                    my_line[1],my_line[2],my_line[3])))
            f.write("Total count = {}".format(total_count))
            f.write("Total contact less count = {}".format(contactless_count))
            f.write("Total contact count = {}".format(contact_count))
        return my_table

    # The generate_actuator_positions_test tests the effect of delay after setting control act joints
    def generate_actuator_positions_test(self, left, right, oreo_eye):
        # right, left are actuator pairs 3,4 or 5,6 - Right and left from oreo's perspective (not viewer)
        # 3,4 control yaw and pitch of oreo's left eye  while 5,6 for the right eye
        # orea_eye can be either " left_eye_joint" or "right_eye_joint"
        total_count = 0
        contact_count = 0
        contactless_count = 0
        my_value = np.linspace(-0.1,0.1, 100)
        my_table = []
        pos = [0] * self.actJointNum
        temp_count = 0
        #wait_time = 0
        print("************ Wait_time none ***************")
        for left_actuator in my_value:
            if temp_count<50:
                pos[left] = left_actuator
                temp_inner_count = 0
                for right_actuator in my_value:
                    if temp_inner_count<5:
                        repeat_loop = 0
                        while repeat_loop < 15:
                            total_count += 1
                            pos[right] = right_actuator
                            self.ControlActJoints(pos)
                            self.actJointPos = pos
                            #time.sleep(0.20)
                            # specified eye
                            eye_orientation = self.GetLinkOrientationWCS(oreo_eye)
                            # convert to numpy quaternion (w,x,y,z) w is the real part.
                            eye_orientation_quat = np.quaternion(eye_orientation[3], eye_orientation[0],
                                                                 eye_orientation[1], \
                                                                 eye_orientation[2])
                            my_rot_matrix_left = quaternion.as_rotation_matrix(eye_orientation_quat)
                            computed_yaw, computed_pitch = compute_yaw_pitch_from_vector(my_rot_matrix_left[:, 0])
                            my_table.append([left_actuator, right_actuator, computed_yaw, computed_pitch])
                            #print("left A ={} Right A ={} Com_yaw = {} Com_pitch = {}".format(left_actuator, \
                            #                                        right_actuator,computed_yaw,computed_pitch))
                            if len(p.getContactPoints()) != 0:
                                contact_count += 1
                            else:
                                contactless_count += 1
                                # my_table.append([left_actuator, right_actuator, computed_yaw, computed_pitch])

                            repeat_loop += 1
                        temp_inner_count +=1
                        #print("Next Right Actuator setting ------------------------------")
                    else:
                        break

                temp_count += 1
                #print("Next Left Actuator setting --------------------------")
            else:
                break


        print("Total count = {}".format(total_count))
        print("Total contact less count = {}".format(contactless_count))
        print("Total contact count = {}".format(contact_count))
        with open("nowait_data", "w") as f:
            for my_line in my_table:
                f.write(("left A ={} Right A ={} Com_yaw = {} Com_pitch = {}\n".format(my_line[0], \
                                                                    my_line[1],my_line[2],my_line[3])))

        sys.exit()
        return my_table

    def build_oreo_scan_yaw_pitch_actuator_data(self):

        scan_data = []
        scan_data.append(self.generate_actuator_positions(3, 4, "left_eye_joint"))
        scan_data.append(self.generate_actuator_positions(5, 6, "right_eye_joint"))

        with open(self.oreo_scan_data,"wb") as f:
            pickle.dump(scan_data,f)


    # reads the pickled data
    def read_oreo_yaw_pitch_actuator_data(self):
        try:
            with open(self.oreo_scan_data, "rb") as f:
                my_data = pickle.load(f)
                # left and right eye [left_actuator,right_actuator,yaw,pitch]
                self.left_eye_scan_data = my_data[0]
                self.right_eye_scan_data = my_data[1]
                return 1
        except IOError as e:
            print("Failure: Opening pickle file {}".format(self.oreo_scan_data))
            return 0

    ##
    def get_max_min_yaw_pitch_values(self):
        my_lefteye_table = np.array(self.left_eye_scan_data)
        x = my_lefteye_table[:, 2]  # all yaw
        y = my_lefteye_table[:, 3]  # all pitch
        print("Left Eye scan data Max yaw {} Min yaw {}".format(max(x), min(x)))
        print("Left Eye scan data Max pitch {} Min pitch {}".format(max(y), min(y)))

        my_righteye_table = np.array(self.right_eye_scan_data)
        x = my_righteye_table[:, 2]  # all yaw
        y = my_righteye_table[:, 3]  # all pitch
        print("Right Eye scan data Max yaw {} Min yaw {}".format(max(x), min(x)))
        print("Right Eye scan data Max pitch {} Min pitch {}".format(max(y), min(y)))
        return

    ##


    # reads the pickled data
    def read_interpolator_functions(self):
        try:
            with open(self.interpolator_pickle_file, "rb") as f:
                my_interp = pickle.load(f)
                # left and right eye [left_actuator,right_actuator,yaw,pitch]
                self.left_eye_interpolator_left = my_interp[0]
                self.left_eye_interpolator_right = my_interp[1]
                self.right_eye_interpolator_left = my_interp[2]
                self.right_eye_interpolator_right = my_interp[3]
                return 1
        except IOError as e:
            print("Failure: Opening pickle file {}".format(self.interpolator_pickle_file))
            return 0


    # create two interpolator functions, one for the left and the other for the right oreo eye
    def produce_interpolators(self):
        self.read_oreo_yaw_pitch_actuator_data()

        my_lefteye_table = np.array(self.left_eye_scan_data)
        x = my_lefteye_table[:,2]   # all yaw
        y = my_lefteye_table[:,3]   # all pitch
        z_left = my_lefteye_table[:,0]  # all left_actuator positions
        z_right = my_lefteye_table[:,1]  # all right_actuator positions
        self.left_eye_interpolator_left = interpolate.interp2d(x, y, z_left, kind='linear')
        self.left_eye_interpolator_right = interpolate.interp2d(x, y, z_right, kind='linear')
        print("Left Eye scan data Max yaw {} Min yaw {}".format(max(x),min(x)))
        print("Left Eye scan data Max pitch {} Min pitch {}".format(max(y), min(y)))

        '''
        # 3D surface plot to check
        zpl = []
        zpr = []
        xp = []
        yp = []
        i = 0
        for j,k in zip(z_left,z_right):
            if (-0.025 < j < +0.025) and ((-0.025 < k < +0.025)):
                xp.append(x[i])
                yp.append(y[i])
                zpl.append(j)
                zpr.append(k)

            i += 1



        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.scatter(xp, yp, zpl, 'z', c='coral', )
        ax1.scatter(xp, yp, zpr, 'z', c='lightblue', )
        ax1.set_xlabel('Yaw in degrees')
        ax1.set_ylabel('Pitch in degrees')
        ax1.set_zlabel('Left-coral Right-blue Actuator')

        # end of 3D surface plot
        '''



        #******** check
        '''plt.plot(x, z_left, 'r.', y, z_left, 'b^')
        plt.show()
        plt.close()
        
        # 3D plot for the interpolated
        znew_left = []
        for xi,yi in zip(x,y):
            zi = self.left_eye_interpolator_left(xi, yi)[0]
            znew_left.append(zi)
        znew_right = []
        for xi, yi in zip(x, y):
            zi = self.left_eye_interpolator_right(xi, yi)[0]
            znew_right.append(zi)

        zpl = []
        zpr = []
        xp = []
        yp = []
        i = 0
        for j, k in zip(znew_left, znew_right):
            if (-0.05 < j < +0.05) and ((-0.05 < k < +0.05)):
                xp.append(x[i])
                yp.append(y[i])
                zpl.append(j)
                zpr.append(k)
            i += 1

        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.scatter(xp, yp, zpl, 'z', c='coral', )
        #ax1.scatter(xp, yp, zpr, 'z', c='lightblue', )
        ax1.set_xlabel('Yaw in degrees')
        ax1.set_ylabel('Pitch in degrees')
        ax1.set_zlabel('Left Actuator - coral')
        ax1.set_title("Left Eye Left Actuator")

        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        #ax1.scatter(xp, yp, zpl, 'z', c='coral', )
        ax1.scatter(xp, yp, zpr, 'z', c='lightblue', )
        ax1.set_xlabel('Yaw in degrees')
        ax1.set_ylabel('Pitch in degrees')
        ax1.set_zlabel('Right Actuator - Blue')
        ax1.set_title('Left Eye Right Actuator')
        #end of 3D interpolated
        '''

        my_righteye_table = np.array(self.right_eye_scan_data)
        x = my_righteye_table[:, 2]  # all yaw
        y = my_righteye_table[:, 3]  # all pitch
        z_left = my_righteye_table[:, 0]  # all left_actuator positions
        z_right = my_righteye_table[:, 1]  # all right_actuator positions
        self.right_eye_interpolator_left = interpolate.interp2d(x, y, z_left, kind='linear')
        self.right_eye_interpolator_right = interpolate.interp2d(x, y, z_right, kind='linear')

        print("Right Eye scan data Max yaw {} Min yaw {}".format(max(x), min(x)))
        print("Right Eye scan data Max pitch {} Min pitch {}".format(max(y), min(y)))

        '''
        # 3D plot for the interpolated right eye
        znew_left = []
        for xi, yi in zip(x, y):
            zi = self.left_eye_interpolator_left(xi, yi)[0]
            znew_left.append(zi)
        znew_right = []
        for xi, yi in zip(x, y):
            zi = self.left_eye_interpolator_right(xi, yi)[0]
            znew_right.append(zi)

        zpl = []
        zpr = []
        xp = []
        yp = []
        i = 0
        for j, k in zip(znew_left, znew_right):
            if (-0.05 < j < +0.05) and ((-0.05 < k < +0.05)):
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
        ax1.set_title('Right Eye Left Actuator')

        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.scatter(xp, yp, zpr, 'z', c='green', )
        ax1.set_xlabel('Yaw in degrees')
        ax1.set_ylabel('Pitch in degrees')
        ax1.set_zlabel('Right Actuator - green')
        ax1.set_title('Right Eye Right Actuator')
        # end of 3D interpolated
        '''
        # pickle the interpolator functions
        #interpolator_pickle_file = "interp_file"
        interp_functions = [self.left_eye_interpolator_left, self.left_eye_interpolator_right, \
                            self.right_eye_interpolator_left, self.right_eye_interpolator_right]
        with open(self.interpolator_pickle_file,"wb") as f:
            pickle.dump(interp_functions,f)

        return

    def plot_interpolator_functions(self):
        #left eye
        self.read_oreo_yaw_pitch_actuator_data()
        my_lefteye_table = np.array(self.left_eye_scan_data)
        x = my_lefteye_table[:, 2]  # all yaw
        y = my_lefteye_table[:, 3]  # all pitch

        z_left = []
        for xi, yi in zip(x, y):
            zi = self.left_eye_interpolator_left(xi, yi)[0]
            z_left.append(zi)
        z_right = []
        for xi, yi in zip(x, y):
            zi = self.left_eye_interpolator_right(xi, yi)[0]
            z_right.append(zi)

        zpl = []
        zpr = []
        xp = []
        yp = []
        i = 0
        for j, k in zip(z_left, z_right):
            if (-0.05 < j < +0.05) and ((-0.05 < k < +0.05)):
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
        ax1.set_title('Left Eye Left Actuator')

        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.scatter(xp, yp, zpr, 'z', c='blue', )
        ax1.set_xlabel('Yaw in degrees')
        ax1.set_ylabel('Pitch in degrees')
        ax1.set_zlabel('Right Actuator - blue')
        ax1.set_title('Left Eye Right Actuator')
        # end of left eye

        #right eye
        my_righteye_table = np.array(self.right_eye_scan_data)
        x = my_righteye_table[:, 2]  # all yaw
        y = my_righteye_table[:, 3]  # all pitch

        z_left = []
        for xi, yi in zip(x, y):
            zi = self.right_eye_interpolator_left(xi, yi)[0]
            z_left.append(zi)
        z_right = []
        for xi, yi in zip(x, y):
            zi = self.right_eye_interpolator_right(xi, yi)[0]
            z_right.append(zi)

        zpl = []
        zpr = []
        xp = []
        yp = []
        i = 0
        for j, k in zip(z_left, z_right):
            if (-0.05 < j < +0.05) and ((-0.05 < k < +0.05)):
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
        ax1.set_title('Right Eye Left Actuator')

        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.scatter(xp, yp, zpr, 'z', c='green', )
        ax1.set_xlabel('Yaw in degrees')
        ax1.set_ylabel('Pitch in degrees')
        ax1.set_zlabel('Right Actuator - green')
        ax1.set_title('Right Eye Right Actuator')
        # end of 3D interpolated
        return

    def compute_yaw_pitch_for_given_point(self, given_p):
        print("computing yaw, pitch for point x = {}, y = {}, z = {} ".format(given_p[0], given_p[1], given_p[2]))
        the_point = np.array(given_p)

        idx1 = self.jointDict["left_eye_joint"]
        left_eye_origin = self.initPosOrn[idx1][self.POS_IDX]

        lefteye_vector = the_point - left_eye_origin
        lefteye_uvector = lefteye_vector / np.linalg.norm(lefteye_vector)
        y_lefteye, p_lefteye = compute_yaw_pitch_from_vector(lefteye_uvector)

        idx2 = self.jointDict["right_eye_joint"]
        right_eye_origin = self.initPosOrn[idx2][self.POS_IDX]

        righteye_vector = the_point - right_eye_origin
        righteye_uvector = righteye_vector / np.linalg.norm(righteye_vector)
        y_righteye, p_righteye = compute_yaw_pitch_from_vector(righteye_uvector)

        return y_lefteye, p_lefteye, y_righteye, p_righteye

    def get_actuator_positions_for_a_given_yaw_pitch(self,my_angles):
        if (self.left_eye_interpolator_left is None or self.left_eye_interpolator_right is None or \
            self.right_eye_interpolator_left is None or self.right_eye_interpolator_right is None):
            return [0, 0.0,0.0,0.0,0.0] # 0 followed by 4 floats

        left_actuator_pos_lefteye = self.left_eye_interpolator_left(my_angles[0], my_angles[1])[0]
        right_actuator_pos_lefteye = self.left_eye_interpolator_right(my_angles[0], my_angles[1])[0]

        left_actuator_pos_righteye = self.right_eye_interpolator_left(my_angles[2], my_angles[3])[0]
        right_actuator_pos_righteye = self.right_eye_interpolator_right(my_angles[2], my_angles[3])[0]
        return[1, left_actuator_pos_lefteye, right_actuator_pos_lefteye, left_actuator_pos_righteye,
               right_actuator_pos_righteye]

    def look_at_point(self, point_data):
        point_angles = point_data[3]

        left_actuator_pos_lefteye = self.left_eye_interpolator_left(point_angles[0], point_angles[1])
        right_actuator_pos_lefteye = self.left_eye_interpolator_right(point_angles[0], point_angles[1])

        left_actuator_pos_righteye = self.right_eye_interpolator_left(point_angles[2], point_angles[3])
        right_actuator_pos_righteye = self.right_eye_interpolator_right(point_angles[2], point_angles[3])

        pos = [0] * self.actJointNum
        pos[3] = left_actuator_pos_lefteye.tolist()[0]
        pos[4] = right_actuator_pos_lefteye.tolist()[0]
        pos[5] = left_actuator_pos_righteye.tolist()[0]
        pos[6] = right_actuator_pos_righteye.tolist()[0]
        self.ControlActJoints(pos)
        self.actJointPos = pos
        collision = len(p.getContactPoints())
        time.sleep(0.1)
        print("Looking at point {}".format(point_data[0:3]))
        print("Acutuator positions Left Eye (L and R) {} and Right eye (L and R) {}".format(pos[3:5], pos[5:]))
        if collision != 0:
            print("BAM!!! - There is collision")
        else:
            print("No collision")

        my_pose = self.get_yaw_and_pitch_from_orientation()
        print("Left Eye Yaw expected {} - Yaw obtained {}".format(point_angles[0], my_pose[0]))
        print("Left Eye Pitch expected {} - Pitch obtained {}".format(point_angles[1], my_pose[1]))
        print("Right Eye Yaw expected {} - Yaw obtained {}".format(point_angles[2], my_pose[2]))
        print("Right Eye Pitch expected {} - Pitch obtained {}".format(point_angles[3], my_pose[3]))
        print("-----------------------------------------------")

    def get_yaw_and_pitch_from_orientation(self):
        eye_orientation = self.GetLinkOrientationWCS("left_eye_joint")
        # convert to numpy quaternion (w,x,y,z) w is the real part.
        eye_orientation_quat = np.quaternion(eye_orientation[3], eye_orientation[0], eye_orientation[1],
                                             eye_orientation[2])
        my_rot_matrix_left = quaternion.as_rotation_matrix(eye_orientation_quat)
        lefteye_yaw, lefteye_pitch = compute_yaw_pitch_from_vector(my_rot_matrix_left[:, 0])

        eye_orientation = self.GetLinkOrientationWCS("right_eye_joint")
        # convert to numpy quaternion (w,x,y,z) w is the real part.
        eye_orientation_quat = np.quaternion(eye_orientation[3], eye_orientation[0], eye_orientation[1],
                                             eye_orientation[2])
        my_rot_matrix_right = quaternion.as_rotation_matrix(eye_orientation_quat)
        righteye_yaw, righteye_pitch = compute_yaw_pitch_from_vector(my_rot_matrix_right[:, 0])

        return lefteye_yaw, lefteye_pitch, righteye_yaw, righteye_pitch

    def compare_actuator_values(self):
        # Values that created the interpolators
        my_lefteye_table = np.array(self.left_eye_scan_data)
        xl = my_lefteye_table[:, 2]  # all yaw
        yl = my_lefteye_table[:, 3]  # all pitch
        zl_left = my_lefteye_table[:, 0]  # all left_actuator positions
        zl_right = my_lefteye_table[:, 1]  # all right_actuator positions
        my_righteye_table = np.array(self.right_eye_scan_data)
        xr = my_righteye_table[:, 2]  # all yaw
        yr = my_righteye_table[:, 3]  # all pitch
        zr_left = my_righteye_table[:, 0]  # all left_actuator positions
        zr_right = my_righteye_table[:, 1]  # all right_actuator positions

        left_actuator_lefteye = self.left_eye_interpolator_left(xl, yl)
        right_actuator_lefteye = self.left_eye_interpolator_right(xl, yl)

        left_actuator_righteye = self.right_eye_interpolator_left(xr, yr)
        right_actuator_righteye = self.right_eye_interpolator_right(xr, yr)

        i = len(xl)
        j = 0
        while i > j:
            print("Input  Left Eye - Left {} Right {}".format(xl[j], yl[j]))
            print("Output Left Eye - Left {} Right {}".format(left_actuator_lefteye[j], right_actuator_lefteye[j]))
            j += 1

        i = len(xr)
        while i > 0:
            print("Input  Right Eye - Left {} Right {}".format(xr[i], yr[i]))
            print("Output Right Eye - Left {} Right {}".format(left_actuator_righteye, right_actuator_righteye))
            print("---------------------------------------------------")

    # Initialize robot model
    def InitModel(self):
        # Map names to id
        self.numJoints = p.getNumJoints(self.linkage)
        logging.info("There are %d links in file\n", self.numJoints)
        vis_data = p.getVisualShapeData(self.linkage)
        for i in range(self.numJoints):
            # map joint names to id (same mapping for links)
            jointInfo = p.getJointInfo(self.linkage, i)
            self.jointDict[jointInfo[1].decode('UTF-8')] = i

            # record parent and child ids for each joint [child, parent]
            #            self.linkJoints.append([[jointInfo[16]], [i]])
            # ignore the dummy link
            #            if(i > 0) :
            #                self.linkJoints[jointInfo[16]][self.LINK_PARENT_IDX].append(jointInfo[16])

            # measure force/torque at the joints
            p.enableJointForceTorqueSensor(self.linkage, i, enableSensor=True)

            # Store initial positions and orientations
            state = p.getLinkState(self.linkage, jointInfo[0])
            self.initPosOrn.append([state[4], state[5]])

            # debugging
            coll_data = p.getCollisionShapeData(self.linkage, i)
            logging.debug("%s %d", jointInfo[1].decode('UTF-8'), i)
            logging.debug("com frame (posn (global) & orn (global)) %s %s", str(state[0]),
                          str(p.getEulerFromQuaternion(state[1])))
            logging.debug("inertial offset (posn (link frame), orn (link frame)) %s %s", str(state[2]),
                          str(p.getEulerFromQuaternion(state[3])))
            logging.debug("link frame (posn (global), orn (global)) %s %s", str(state[4]),
                          str(p.getEulerFromQuaternion(state[5])))
            logging.debug("type %s", str(jointInfo[2]))
            logging.debug("axis %s", str(jointInfo[13]))
            logging.debug("collision (posn (COM frame), orn(COM frame)) %s %s", str(coll_data[0][5]),
                          str(p.getEulerFromQuaternion(coll_data[0][6])))
            logging.debug("visual (posn (link frame), orn(link frame)) %s %s\n\n", str(vis_data[i][5]),
                          str(p.getEulerFromQuaternion(vis_data[i][6])))

        # Create list of actuated joints
        self.actJointIds = [self.jointDict[act_name] for act_name in self.actJointNames]
        self.actJointNum = len(self.actJointNames)

        # Rajan
        self.prev_pos = [0] * self.actJointNum

        # Create list of spherical joints
        self.spherJointIds = [self.jointDict[spher_name] for spher_name in self.spherJointNames]
        self.spherJointNum = len(self.spherJointNames)

        # Create list of dumb joints
        self.dumbJointIds = [self.jointDict[dumb_name] for dumb_name in self.dumbJointNames]
        self.dumbJointNum = len(self.dumbJointNames)

        # Init struct
        self.linkVelo = [[0, 0, 0], [0, 0, 0]] * p.getNumJoints(self.linkage)

        # Enable collision by default
        self.ToggleCollision(1)

        # Go to home position
        self.HomePos()

        # Constraints
        for i in range(self.numConstraints):
            parent_id = self.jointDict[self.constraintLinks[i][self.CONS_PARENT_IDX]]
            child_id = self.jointDict[self.constraintLinks[i][self.CONS_CHILD_IDX]]
            self.constraintDict[self.constraintLinks[i][self.CONS_NAME_IDX]] = p.createConstraint(self.linkage,
                                                                                                  parent_id,
                                                                                                  self.linkage,
                                                                                                  child_id,
                                                                                                  self.constraintType,
                                                                                                  self.constraintAxis,
                                                                                                  self.constraintParentPos[
                                                                                                      i],
                                                                                                  self.constraintChildPos)
            p.changeConstraint(self.constraintDict[self.constraintLinks[i][self.CONS_NAME_IDX]],
                               maxForce=self.CONSTRAINT_MAX_FORCE)

        #            # Add to linkJoint list (make constraints -ve ids)
        #            self.linkJoints[parent_id][self.LINK_PARENT_IDX].append(parent_id*-1)
        #            self.linkJoints[child_id][self.LINK_CHILD_IDX].append(child_id*-1)

        # Set joint control
        self.SetActJointControlType(self.POSITION_CONTROL)
        self.ResetActJointControl()
        self.ControlDumbJoints()
        self.ControlSpherJoints()

        # Gravity
        p.setGravity(0, 0, self.GRAV_ACCELZ)

        # Simulation type
        self.SetSimType(self.useRealTime)

    # Check if links have collided
    def CheckCollision(self, linkA, linkB):
        points = p.getContactPoints(bodyA=self.linkage, bodyB=self.linkage, linkIndexA=self.jointDict[linkA],
                                    linkIndexB=self.jointDict[linkB])

        if not points:
            logging.debug("No collision points between %s & %s detected\n", linkA, linkB)
            return False
        else:
            logging.debug("Number of collision points between %s & %s detected: %d\n", linkA, linkB, len(points))
            return True
        print()
        print()

    # Check all contact points
    def CheckAllCollisions(self):
        points = p.getContactPoints()
        logging.debug("Contact Points")
        for temp in points:
            logging.debug("a & b = %d & %d", temp[3], temp[4])
        print()
        print()

    # Get list of keys pressed

    def GetKeyEvents(self):
        pressed = []
        keyEvents = p.getKeyboardEvents()
        for char in self.keys:
            key = ord(char)
            if key in keyEvents and keyEvents[key] & p.KEY_WAS_TRIGGERED:
                pressed.append(char)
        return pressed

    # Add key to listen

    def RegKeyEvent(self, userIn):
        if isinstance(userIn, str):
            if userIn not in self.keys:
                self.keys.append(userIn)
        elif isinstance(userIn, list):
            for char in userIn:
                if char not in self.keys:
                    self.keys.append(char)

    # Cleanup stuff
    def Cleanup(self):
        p.disconnect()

    # Reset robot
    def Reset(self):
        p.resetSimulation()

        self.InitModel()

    # Get list of joint names
    def GetJointNames(self):
        return list(self.jointDict.keys())

    # Get list of link names
    def GetLinkNames(self):
        return [name.replace("joint", "link", 1) for name in self.jointDict.keys()]

    # Print constraint reaction forces
    def PrintConstraintDynamics(self):
        logging.debug("Constraint States")

        for cons_id in self.constraintDict:
            logging.debug(p.getConstraintState(self.constraintDict[cons_id]))
        print()
        print()

    # Update link velo
    def UpdateAllLinkAccel(self):
        id_list = list(range(p.getNumJoints(self.linkage)))

        ret = p.getLinkStates(self.linkage, id_list, computeLinkVelocity=1)
        idx = 0
        for state in ret:
            self.linkVelo[idx] = [list(state[6]), list(state[7])]
            idx += 1

        # Print link approx accel at last time step
        def PrintLinkAccel(self):
            logging.info("Link Dynamics")

        id_list = list(range(p.getNumJoints(self.linkage)))
        ret = p.getLinkStates(self.linkage, id_list, computeLinkVelocity=1)
        idx = 0
        for state in ret:
            accel = list(self.linkVelo[idx][self.VELO_IDX])
            ang_accel = list(self.linkVelo[idx][self.ANG_VELO_IDX])
            for i in range(len(accel)):
                accel[i] = (state[6][i] - accel[i]) / self.TIME_STEP
                ang_accel[i] = (state[7][i] - ang_accel[i]) / self.TIME_STEP
            logging.debug("Id=%d accel=%s ang_accel=%s", idx, str(accel), str(ang_accel))
            self.linkVelo[idx] = [list(state[6]), list(state[7])]
            idx += 1
        print()
        print()

    # Compute approx accelerations of the link at last time step

    def GetLinkAccel(self, name):

        # check if given link name (link == joint)
        if "link" in name:
            name.replace("link", "joint", 1)

        # Check link exists
        if name in self.jointDict:
            ret = p.getLinkState(self.linkage, self.jointDict[name], computeLinkVelocity=1)
            accel = self.linkVelo[linkId][self.VELO_IDX]
            ang_accel = self.linkVelo[linkId][self.ANG_VELO_IDX]
            linkId = self.jointDict[name]
            for i in range(len(accel)):
                accel = (accel[i] - ret[6][i]) / self.TIME_STEP
                ang_accel = (ang_accel[i] - ret[7][i]) / self.TIME_STEP
            self.linkVelo[linkId] = [ret[6], ret[7]]
        else:
            logging.error("Unknown name input to GetLinkAccel")
            return []

        return [list(accel), list(ang_accel)]

    # Get the position and orientation
    def GetLinkPosOrn(self, name):

        # check if given link name (link == joint)
        if "link" in name:
            name.replace("link", "joint", 1)
        idx = self.jointDict[name]
        # Check link exists
        if name in self.jointDict:
            state = p.getLinkState(self.linkage, idx, computeForwardKinematics=True)
            pos = list(np.array(state[4]) - np.array(self.initPosOrn[idx][self.POS_IDX]))
            # orn = np.array(state[5]) - self.initPosOrn[idx][self.ORN_IDX]
            orn = p.getDifferenceQuaternion(self.initPosOrn[idx][self.ORN_IDX], state[5])
            # logging.debug("%s pos=%s orn=%s", name, str(pos), str(p.getEulerFromQuaternion(orn)))
        else:
            logging.error("Unknown name input to GetLinkPosOrn")
            return []
        return [pos, orn]

    def GetLinkOrientationWCS(self, name):

        # check if given link name (link == joint)
        if "link" in name:
            name.replace("link", "joint", 1)
        idx = self.jointDict[name]
        # Check link exists
        if name in self.jointDict:
            state = p.getLinkState(self.linkage, idx, computeForwardKinematics=True)
            frame_orientation = list(state[5])
        else:
            logging.error("GetLinkOrientationWCS")
            return []
        return frame_orientation

