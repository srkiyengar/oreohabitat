import habitat_sim
import numpy as np
import quaternion
import sys
import cv2
import oreo



from typing import Any, Dict, List, Union

# pybullet quaternion order x, y, z, w
# numpy quaternion order w, x, y, z
# Whenever a quaternion is specified it is in numpy quaternion convention

import attr

eye_seperation = 0.058
display = "a"
toggle = 0

@attr.s(auto_attribs=True, slots=True)
class SixDOFPose(object):
    r"""Specifies a position with 6 degrees of freedom

    :property position: xyz position
    :property rotation: unit quaternion rotation
    """
    position: np.ndarray = np.zeros(3)
    rotation: Union[np.quaternion, List] = np.quaternion(1, 0, 0, 0)


from habitat_sim.utils.common import (
    quat_from_coeffs,
    quat_from_magnum,
    quat_rotate_vector,
    quat_to_magnum,
)


def create_sensor(orientation=[0.0, 0.0, 0.0], position=[0.0, 0.0, 0.0], sensor_resolution=[512, 512],
                  sensor_uuid="my_sensor", camera_type="C"):
    """
    :param orientation: Axis Angle representation 3 values and describes the rotation with respect to agent
    :param position: in meters x - distance from the ground y - right of the agent, z is up
    :param sensor_resolution: w has to equal to h for foveation (Ryan makes this assumption for shader)
    :param sensor_uuid: The identifier that will be used to refer to the sensor
    :param camera_type: camera_type "C" is RGB Camera and "D" is Depth
    :return: sensor object
    The sensor to the right is the left eye for OREO
    """

    new_sensor = habitat_sim.SensorSpec()
    new_sensor.uuid = sensor_uuid
    if camera_type == "D":
        new_sensor.sensor_type = habitat_sim.SensorType.DEPTH
    else:
        new_sensor.sensor_type = habitat_sim.SensorType.COLOR
    new_sensor.resolution = sensor_resolution

    new_sensor.position = position  # +ive x is to the right, +ive y is UP, -ive z is in front
    new_sensor.orientation = np.array(orientation, dtype=float)

    return new_sensor


def setup_sim_and_sensors():
    # left sensor corresponds to left eye which is on the right side
    left_rgb_sensor = create_sensor(position=[eye_seperation / 2, 0, 0], sensor_uuid="left_rgb_sensor")
    right_rgb_sensor = create_sensor(position=[-eye_seperation / 2, 0, 0], sensor_uuid="right_rgb_sensor")
    depth_sensor = create_sensor(sensor_uuid="depth_sensor", camera_type="D")

    # agent configuration has the sensor_specifications objects as a list
    new_agent_config = habitat_sim.AgentConfiguration()
    new_agent_config.sensor_specifications = [left_rgb_sensor, right_rgb_sensor, depth_sensor]
    # Configuration of the backend of the simulator includes default_agent_id set to 0
    backend_cfg = habitat_sim.SimulatorConfiguration()
    default_agent_id = backend_cfg.default_agent_id
    backend_cfg.foveation_distortion = True

    backend_cfg.scene.id = (
        "../multi_agent/data_files/skokloster-castle.glb"
    )
    '''
    backend_cfg.scene.id = (
        "/Users/rajan/My_Replica/replica_v1/apartment_1/mesh.ply"
     )
    '''
    # Tie the backend of the simulator and a list of agent configurations
    new_Configuration = habitat_sim.Configuration(backend_cfg, [new_agent_config])
    # When Simulator is called the agent configuration becomes Agents (only one agent in our case]
    new_sim = habitat_sim.Simulator(new_Configuration)
    return new_sim, default_agent_id


def get_agent_sensor_position_orientations(my_agent):
    """
    :param my_agent: object obtained from setting sim and agent sensors
    :return: agent orientation a quaternion and sensors orientation is a dict with sensor-id as key
    and the orientation as value.
    get_agent_sensor_position_orientations are both with respect to the habitat frame.
    Internally the relative sensor orientation and translations with respect to agent are stored under _sensor
    but the sensor_states that are accessible are wrt habitat frame only.
    The sensor states (eyes) wrt to the agent state (head/neck) is NOT computed by this function.
    """

    my_agent_state = my_agent.get_state()
    agent_orientation = my_agent_state.rotation
    agent_position = my_agent_state.position
    sensors_position_orientation = my_agent_state.sensor_states
    return agent_orientation, agent_position, sensors_position_orientation


def relative_sensor_rotation(my_agent, sensors_rotation):
    """
    The sensors_rotation is specified with respect to its current frame.
    This function will set the sensors orientation in habitat  w.r.t. to habitat frame.
    The agent is not moving or changing orientation.
    It is important to remember that sensor_states are with respect to habitat frame while the relative rotation
    between agent and sensor is saved within protected _sensor by the set agent sensor API function
    :param my_agent: agent object
    :param sensors_rotation: list of quaternions - rotation of the sensors with respect to agent frame
    :return: nothing
    """

    my_agent_state = my_agent.get_state()
    agent_orn = my_agent_state.rotation
    my_agent_state.sensor_states["left_rgb_sensor"].rotation = agent_orn * sensors_rotation[0]
    my_agent_state.sensor_states["right_rgb_sensor"].rotation = agent_orn * sensors_rotation[1]
    my_agent_state.sensor_states["depth_sensor"].rotation = agent_orn * sensors_rotation[2]
    my_agent.set_state(my_agent_state, infer_sensor_states=False)
    return

def rotate_sensor_wrt_habitat_frame(my_agent, sensors_rotation):
    """
    The sensors_rotation is specified with respect to the habitat frame.
    :param my_agent: agent object
    :param sensors_rotation: list of quaternions - rotation of the sensors with respect to habitat frame
    :return: nothing
    """

    my_agent_state = my_agent.get_state()
    agent_orn = my_agent_state.rotation
    my_agent_state.sensor_states["left_rgb_sensor"].rotation = sensors_rotation[0]
    my_agent_state.sensor_states["right_rgb_sensor"].rotation = sensors_rotation[1]
    my_agent_state.sensor_states["depth_sensor"].rotation = sensors_rotation[2]
    my_agent.set_state(my_agent_state, infer_sensor_states=False)
    return


def verge_sensors(my_angle, my_agent, verge):
    '''

    :param my_angle: The angle by which converging or diverging from the current position
    :param my_agent: The agent object
    :param verge: 'c' it will converge 'd' it will diverge
    :return:
    '''

    if verge == 'c':
        left_sensor = quaternion.from_rotation_vector([0.0, my_angle, 0.0])
        right_sensor = quaternion.from_rotation_vector([0.0, -my_angle, 0.0])
    elif verge == 'd':
        left_sensor = quaternion.from_rotation_vector([0.0, -my_angle, 0.0])
        right_sensor = quaternion.from_rotation_vector([0.0, my_angle, 0.0])
    else:
        return
    depth_sensor = quaternion.from_rotation_vector([0.0, 0.0, 0.0])
    my_agent_state = my_agent.get_state()
    r1_inverse = my_agent_state.rotation.inverse()
    left_sensor = (r1_inverse*my_agent_state.sensor_states["left_rgb_sensor"].rotation)*left_sensor
    right_sensor = (r1_inverse*my_agent_state.sensor_states["right_rgb_sensor"].rotation)*right_sensor
    sensor_rot = [left_sensor, right_sensor, depth_sensor]
    relative_sensor_rotation(my_agent,sensor_rot)
    return


def verge_sensors_to_midpoint_depth(my_agent, my_sim, my_robot):
    scene_depth, res = get_depth(my_sim)
    center_x = int(res.x / 2)
    center_y = int(res.x / 2)
    my_z = -scene_depth[center_x, center_y]

    my_point = np.array([[0.0], [0.0], [my_z]])
    my_point = ((rotatation_matrix_from_Pybullet_to_Habitat().dot(my_point)).flatten()).tolist()
    # move oreo eyes to the point in pybullet to detect collision
    the_angles = my_robot.compute_yaw_pitch_for_given_point(my_point)
    val = my_robot.get_actuator_positions_for_a_given_yaw_pitch(the_angles)
    if val[0] == 1:
        a_pos = val[1:]

        collision, lefteye_orn, righteye_orn = my_robot.move_eyes_to_position_and_return_orn(a_pos)
        if collision == 0:
            depth_sensor = quaternion.from_rotation_vector([0.0, 0.0, 0.0])
            sensor_rot = [lefteye_orn, righteye_orn, depth_sensor]
            rotate_sensor_wrt_habitat_frame(my_agent, sensor_rot)
            return
        else:
            print("verge_sensors_to_midpoint_depth: Collision for depth {}".format(my_z))
            return
    else:
        print("verge_sensors_to_midpoint_depth:Interpolator functions not computed for depth {}".format(my_z))
        return


def verge_sensors_to_point(my_agent, my_robot, my_p):

    my_point = np.array(my_p)
    my_point.shape = (3,1)
    my_point = ((rotatation_matrix_from_Pybullet_to_Habitat().dot(my_point)).flatten()).tolist()
    # move oreo eyes to the point in pybullet to detect collision
    the_angles = my_robot.compute_yaw_pitch_for_given_point(my_point)
    print("Verge_to_point: Target Angles : Left eye yaw = {}, pitch = {}, Right eye yaw = {}, pitch = {}".
          format(the_angles[0],the_angles[1], the_angles[2],the_angles[3]))
    quat = compute_rotation_for_a_given_yaw_pitch(the_angles[0],the_angles[1])
    val = my_robot.get_actuator_positions_for_a_given_yaw_pitch(the_angles)
    if val[0] == 1:
        a_pos = val[1:]
        collision, lefteye_orn, righteye_orn = my_robot.move_eyes_to_position_and_return_orn(a_pos)
        print("VergeToPoint: computed = {} - From Pybullet = {}".format(quat, lefteye_orn))
        if collision == 0:
            depth_sensor = quaternion.from_rotation_vector([0.0, 0.0, 0.0])
            sensor_rot = [lefteye_orn, righteye_orn, depth_sensor]
            yp_left, yp_right = compute_yaw_pitch_from_orientation(lefteye_orn, righteye_orn)
            print("Actuator positions = {}".format(a_pos))
            print ("Verge_to_point: Achieved: Left eye yaw = {}, pitch = {}, Right eye yaw = {}, pitch = {}".format(yp_left[0],
                   yp_left[1], yp_right[0], yp_right[1]))
            # print(sensor_rot)
            rotate_sensor_wrt_habitat_frame(my_agent, sensor_rot)
            return

        else:
            print("verge_sensors_to_point: Collision for point {}".format(my_p))
            return
    else:
        print("verge_sensors_to_point:Interpolator functions not computed for depth {}".format(my_p))
        return

def rotate_agent(my_agent, my_rotation):
    """
    The agent has only a rotation. The sensor orientation and position wrt to habitat frame changes when this
    changes. The sensors themselves remain unchanged in their orientation or position with respect to agent frame.
    The next function move_and_rotate_agent with default move should produce the same result.
    :param my_agent: agent object
    :param my_rotation: quaternion with respect to habitat frame
    :return:
    """

    my_agent_state = my_agent.get_state()
    r1 = my_agent_state.rotation  # rotation from Habitat to Agent wrt Habitat frame
    t1 = my_agent_state.position  # translation from Habitat to Agent wrt Habitata frame
    r1_inverse = my_agent_state.rotation.inverse()  # Inverse of rq

    # only rotations are impacted and not positions
    my_agent_state.rotation = my_rotation  # new rotation of Agent wrt to Habitat

    # Rotation
    my_agent_state.sensor_states["left_rgb_sensor"].rotation = \
        my_rotation * (r1_inverse * my_agent_state.sensor_states["left_rgb_sensor"].rotation)
    my_agent_state.sensor_states["right_rgb_sensor"].rotation = \
        my_rotation * (r1_inverse * my_agent_state.sensor_states["right_rgb_sensor"].rotation)
    my_agent_state.sensor_states["depth_sensor"].rotation = \
        my_rotation * (r1_inverse * my_agent_state.sensor_states["depth_sensor"].rotation)

    my_agent.set_state(my_agent_state, infer_sensor_states=False)
    return


def move_and_rotate_agent(my_agent, my_rotation, move=[0.9539339, 0.1917877, 12.163067]):
    """
    The agent has a translation and rotation. The sensor orientation and position wrt to habitat frame changes when this
    changes. The sensors themselves remain unchanged in their orientation or position with respect to agent frame.
    :param my_agent: agent object
    :param my_rotation: quaternion with respect to habitat frame
    :param move: in habitat frame [x,y,z]
    :return:
    """

    my_agent_state = my_agent.get_state()
    r1 = my_agent_state.rotation  # rotation from Habitat to Agent wrt Habitat frame
    t1 = my_agent_state.position  # translation from Habitat to Agent wrt Habitata frame

    # h1 is current habitat to Agent homogenous transformation (HT), h1new is habitat to Agent with new one
    # h3 - habitat to sensor which is h1 or h1new multiplied and agent to the sensor HT
    # Example - h1new(left_rgb)*h1_inv*h3 where h1_inv*h3 provides sensor pose wrt to agent.
    # Import to remember that get agent_state provides position, orientation wrt to habitant

    h1 = homogenous_transform(quaternion.as_rotation_matrix(r1), list(t1))
    h1_inv = inverse_homogenous_transform(h1)
    h1new = homogenous_transform(quaternion.as_rotation_matrix(my_rotation), move)
    h3_left_rgb = homogenous_transform(quaternion.as_rotation_matrix(
        my_agent_state.sensor_states["left_rgb_sensor"].rotation),
        list(my_agent_state.sensor_states["left_rgb_sensor"].position))
    h3_right_rgb = homogenous_transform(quaternion.as_rotation_matrix(
        my_agent_state.sensor_states["right_rgb_sensor"].rotation),
        list(my_agent_state.sensor_states["right_rgb_sensor"].position))
    h3_depth = homogenous_transform(quaternion.as_rotation_matrix(
        my_agent_state.sensor_states["depth_sensor"].rotation),
        list(my_agent_state.sensor_states["depth_sensor"].position))

    h = h1new.dot(h1_inv)
    newh3_left_rgb = h.dot(h3_left_rgb)
    newh3_right_rgb = h.dot(h3_right_rgb)
    newh3_depth = h.dot(h3_depth)
    # compute new sensor states for each sensor
    my_agent_state.sensor_states["left_rgb_sensor"].rotation = \
        quaternion.from_rotation_matrix(newh3_left_rgb[0:3, 0:3])
    D = newh3_left_rgb[0:3, 3]
    my_agent_state.sensor_states["left_rgb_sensor"].position = \
        D.T
    my_agent_state.sensor_states["right_rgb_sensor"].rotation = \
        quaternion.from_rotation_matrix(newh3_right_rgb[0:3, 0:3])
    D = newh3_right_rgb[0:3, 3]
    my_agent_state.sensor_states["right_rgb_sensor"].position = \
        D.T
    my_agent_state.sensor_states["depth_sensor"].rotation = \
        quaternion.from_rotation_matrix(newh3_depth[0:3, 0:3])
    D = newh3_depth[0:3, 3]
    my_agent_state.sensor_states["depth_sensor"].position = \
        D.T
    # agent state
    my_agent_state.rotation = my_rotation
    my_agent_state.position = move

    my_agent.set_state(my_agent_state, infer_sensor_states=False)
    return


def relative_move_and_rotate_agent(my_agent, rel_rotation, rel_move=[0.0, 0.0, 0.0]):
    """
    The agent move and rotate are expressed relative to its current state.
    The sensor orientation and position wrt to habitat frame changes when this changes.
    The sensors themselves remain unchanged in their orientation or position with respect to agent frame.
    :param my_agent: agent object
    :param rel_rotation: relative rotation (quaternion) with respect to habitat frame from the previous position
    :param rel_move: in habitat frame [x,y,z]
    :return:
    """

    my_agent_state = my_agent.get_state()
    r1 = my_agent_state.rotation  # rotation from Habitat to Agent wrt Habitat frame
    t1 = my_agent_state.position  # translation from Habitat to Agent wrt Habitat frame
    r1_inverse = my_agent_state.rotation.inverse()  # Inverse of rq
    my_rotation = r1 * rel_rotation     # rotation wrt to habitat frame
    # translation wrt to habitat frame
    m = np.reshape(np.array(rel_move, dtype=float),(3,1))
    n = quaternion.as_rotation_matrix(r1)
    p = n.dot(m)
    move = (np.reshape(p,(1,3)) + t1)[0]
    my_agent_state.rotation = my_rotation  # new rotation of Agent wrt to Habitat
    if rel_move == [0.0, 0.0, 0.0]:  # only rotations are impacted and not position
        my_agent_state.sensor_states["left_rgb_sensor"].rotation = \
            my_rotation * (r1_inverse * my_agent_state.sensor_states["left_rgb_sensor"].rotation)
        my_agent_state.sensor_states["right_rgb_sensor"].rotation = \
            my_rotation * (r1_inverse * my_agent_state.sensor_states["right_rgb_sensor"].rotation)
        my_agent_state.sensor_states["depth_sensor"].rotation = \
            my_rotation * (r1_inverse * my_agent_state.sensor_states["depth_sensor"].rotation)
    else:
        # h1 is current habitat to Agent homogenous transformation (HT), h1new is habitat to Agent with new pos,orn.
        # h3 - habitat to sensor which is h1 or h1new multiplied and agent to the sensor HT
        # Example - h1new(left_rgb)*h1_inv*h3 where h1_inv*h3 provides sensor pose wrt to agent.
        # Import to remember that get agent_state provides position, orientation wrt to habitant
        my_agent_state.position = move
        h1 = homogenous_transform(quaternion.as_rotation_matrix(r1), t1.tolist())
        h1_inv = inverse_homogenous_transform(h1)
        h1new = homogenous_transform(quaternion.as_rotation_matrix(my_rotation), move.tolist())
        h3_left_rgb = homogenous_transform(quaternion.as_rotation_matrix(
            my_agent_state.sensor_states["left_rgb_sensor"].rotation),
            list(my_agent_state.sensor_states["left_rgb_sensor"].position))
        h3_right_rgb = homogenous_transform(quaternion.as_rotation_matrix(
            my_agent_state.sensor_states["right_rgb_sensor"].rotation),
            list(my_agent_state.sensor_states["right_rgb_sensor"].position))
        h3_depth = homogenous_transform(quaternion.as_rotation_matrix(
            my_agent_state.sensor_states["depth_sensor"].rotation),
            list(my_agent_state.sensor_states["depth_sensor"].position))

        h = h1new.dot(h1_inv)
        newh3_left_rgb = h.dot(h3_left_rgb)
        newh3_right_rgb = h.dot(h3_right_rgb)
        newh3_depth = h.dot(h3_depth)
        my_agent_state.sensor_states["left_rgb_sensor"].rotation = \
            quaternion.from_rotation_matrix(newh3_left_rgb[0:3, 0:3])
        D = newh3_left_rgb[0:3, 3]
        my_agent_state.sensor_states["left_rgb_sensor"].position = \
            D.T
        my_agent_state.sensor_states["right_rgb_sensor"].rotation = \
            quaternion.from_rotation_matrix(newh3_right_rgb[0:3, 0:3])
        D = newh3_right_rgb[0:3, 3]
        my_agent_state.sensor_states["right_rgb_sensor"].position = \
            D.T
        my_agent_state.sensor_states["depth_sensor"].rotation = \
            quaternion.from_rotation_matrix(newh3_depth[0:3, 0:3])
        D = newh3_depth[0:3, 3]
        my_agent_state.sensor_states["depth_sensor"].position = \
            D.T

    my_agent.set_state(my_agent_state, infer_sensor_states=False)
    return


def compute_yaw_pitch_from_orientation(left_quat, right_quat):
    '''

    :param left_quat: numpy quaternion (w,x,y,z) for left sensor
    :param right_quat: numpy quaternion for right sensor
    :return: tuple - left yaw, pitch, right yaw, pitch in degrees
    '''
    my_rot_matrix_left = quaternion.as_rotation_matrix(left_quat)
    yp1 = oreo.compute_yaw_pitch_from_vector(my_rot_matrix_left[:, 0])
    print("New x-unit vector = {}".format(my_rot_matrix_left[:, 0]))
    # Right eye
    my_rot_matrix_right = quaternion.as_rotation_matrix(right_quat)
    yp2 = oreo.compute_yaw_pitch_from_vector(my_rot_matrix_right[:, 0])
    return yp1, yp2


def compute_rotation_for_a_given_yaw_pitch(given_yaw, given_pitch, given_roll=0.0):
    '''
    This uses the yaw and the pitch to calculate a rotation of the current x-axis to a new direction.
    To perform a roll, another rotation of roll angle around the new-axis needs to be performed.
    From a fixed axis perspective it should be equivalent a sequence of yaw, pitch and roll.
    :param given_yaw:
    :param given_pitch:
    :param given_roll:
    :return: quaternion
    '''

    rz = np.cos(np.radians(given_pitch))
    rxy = np.sin(np.radians(given_pitch))
    ry = rxy*np.sin(np.radians(given_yaw))
    rx = rxy*np.cos(np.radians(given_yaw))
    uvector = np.array([rx,ry,rz])
    v1 = uvector/np.linalg.norm(uvector)

    v2 = np.array([1.0, 0.0, 0.0])
    # my_axis is v2 cross v1
    my_axis = np.cross(v2, v1)
    my_axis = my_axis / np.linalg.norm(my_axis)
    my_angle = np.arccos(np.dot(v1, v2))
    my_axis_angle = my_angle * my_axis
    quat1 = quaternion.from_rotation_vector(my_axis_angle)
    rot_mat = quaternion.as_rotation_matrix(quat1)
    print("x-axis vector = {}".format(rot_mat[:,0]))
    if given_roll == 0.0:
        return quat1
    else:
        roll_axis_angle = np.radians(given_roll)*v1
        quat2 = quaternion.from_rotation_vector(roll_axis_angle)
        return quat1*quat2


'''
Habitat frame versus Pybullet frame
Habitatai coordinate frame (x+ive,y+ive,z -ive) is Pybullet's frame (y-ive, z+ive, x+ive)
Rotation from Habitat to PyBullet R row 0 = [0.-1,0], row 1 = [0,0,1], row 2 = [-1,0,0]
Homogenous transformation can be obtained using the position of the agent.
Rotation (Inverse) from Pybullet to Habitatai row 0 = [0 0 -1], row 1 = [-1 0 0], row 2 = [0,1,0]]
Given a point in Habitat, we do a rotation to pybullet, determine if there is collision and get 
the orientation of the eyes. This orientation has to be rotated back to view the scene in Habitat.
'''

def rotatation_matrix_from_Habitat_to_Pybullet():
    R = np.zeros((3, 3))
    R[0, 1] = -1
    R[1, 2] = 1
    R[2, 0] = -1

    return R


def rotatation_matrix_from_Pybullet_to_Habitat():
    R = np.zeros((3, 3))
    R[0, 2] = -1
    R[1, 0] = -1
    R[2, 1] = 1

    return R


def homogenous_transform(R, vect):
    '''
    :param R: 3x3 matrix
    :param vect: list x,y,z
    :return:Homogenous transformation 4x4 matrix using R and vect
    '''

    H = np.zeros((4, 4))
    H[0:3, 0:3] = R
    frame_displacement = vect + [1]
    D = np.array(frame_displacement)
    D.shape = (1, 4)
    H[:, 3] = D
    return H


def inverse_homogenous_transform(H):
    '''
    :param H: Homogenous Transform Matrix
    :return: Inverse Homegenous Transform Matrix
    '''

    R = H[0:3, 0:3]
    origin = H[:-1, 3]
    origin.shape = (3, 1)

    R = R.T
    origin = -R.dot(origin)
    return homogenous_transform(R, list(origin.flatten()))

def get_depth(oreo_sim):
    depth_sensor = oreo_sim._sensors['depth_sensor']
    depth_sensor.draw_observation()
    return depth_sensor.get_observation(), depth_sensor._sensor_object.framebuffer_size


def get_sensor_observations(oreo_sim):
    '''
    :param oreo_sim: the simulator object
    :return: dict with sensor id as key and it values as ndarray (sensor_resolution (512 x 512), 4) for rgb sensors
    and depth resolution - (512 x 512)
    '''
    for _, sensor in oreo_sim._sensors.items():
        sensor.draw_observation()

    observations = {}
    for sensor_uuid, sensor in oreo_sim._sensors.items():
        observations[sensor_uuid] = sensor.get_observation()
    rgb_left = observations["left_rgb_sensor"]
    rgb_right = observations["right_rgb_sensor"]
    depth = observations["depth_sensor"]

    if len(rgb_left.shape) > 2:
        rgb_left = rgb_left[..., 0:3][..., ::-1]
    if len(rgb_right.shape) > 2:
        rgb_right = rgb_right[..., 0:3][..., ::-1]
    #depth = np.clip(depth, 0, 10)
    depth /= 10.0
    stereo_pair = np.concatenate([rgb_left, rgb_right], axis=1)

    return stereo_pair, depth

def display_image(oreo_sim):
    global toggle

    if toggle == 1:
        cv2.destroyAllWindows()
        toggle = 0

    stereo_image, depth_image = get_sensor_observations(oreo_sim)
    lefteye_image = stereo_image[:,0:512,:]
    righteye_image = stereo_image[:,512:,:]

    if display == 'l':
        cv2.imshow("Left_eye", lefteye_image)
    else:
        cv2.imshow("stereo_pair", stereo_image)
        cv2.imshow("depth", depth_image)

    return


def test_oreo_agent(new_agent, my_sim):
    init_pos = move = [0.9539339, 0.1917877, 12.163067]
    N = 16  # number of moves or rotations
    j = 0
    d = quaternion.from_rotation_vector([0.0, 0.0, 0.0])

    original_agent_state = new_agent.get_state()

    for i in list(range(N)):
        move = [0.1 * i, 0.0, 0.0]
        relative_move_and_rotate_agent(new_agent, d, move)
        stereo_image, depth_image = get_sensor_observations(my_sim)
        cv2.imshow("stereo_pair", stereo_image)
        cv2.imshow("depth", depth_image)
        k = cv2.waitKey(0)
        if k == ord('q'):
            break
        else:
            j += 1
            print("{})Key input {}".format(j, k))

    cv2.destroyAllWindows()

    # Testing move and rotate agent
    j = 0
    for i in list(range(N)):
        d = quaternion.from_rotation_vector([0.0, i * np.pi / N, 0.0])
        move_and_rotate_agent(new_agent, d,
                              move=[init_pos[0] + (0.1 * i), init_pos[1] + 0.5, init_pos[2] + 0.0])  # absolute
        stereo_image, depth_image = get_sensor_observations(my_sim)
        cv2.imshow("stereo_pair", stereo_image)
        cv2.imshow("depth", depth_image)
        k = cv2.waitKey(0)
        if k == ord('q'):
            break
        else:
            j += 1
            print("{})Key input {}".format(j, k))

    cv2.destroyAllWindows()
    j = 0
    # restore original agent state
    new_agent.set_state(original_agent_state)
    # Testing relative move and rotate agent
    d = quaternion.from_rotation_vector([0.0, np.pi / N, 0.0])
    for i in list(range(N)):
        move = [0.1 * i, 0.0, 0.0]
        relative_move_and_rotate_agent(new_agent, d, move)
        stereo_image, depth_image = get_sensor_observations(my_sim)
        cv2.imshow("stereo_pair", stereo_image)
        cv2.imshow("depth", depth_image)
        k = cv2.waitKey(0)
        if k == ord('q'):
            break
        else:
            j += 1
            print("{})Key input {}".format(j, k))

    cv2.destroyAllWindows()

def look_around(my_agent, my_sim, my_robot, type = "a"):
    global display, toggle
    original_state = my_agent.get_state()
    #verge_sensors_to_midpoint_depth(my_agent, my_sim, my_robot)
    if type == 'l':
        display = "l"
    display_image(my_sim)
    delta_move = 0.1
    while (1):
        k = cv2.waitKey(0)
        if k == ord('q'):
            break
        elif k == ord('p'):
            _, _, sensors_pos_orn = get_agent_sensor_position_orientations(my_agent)
            ypl,ypr = compute_yaw_pitch_from_orientation(sensors_pos_orn['left_rgb_sensor'].rotation,
                                                    sensors_pos_orn['right_rgb_sensor'].rotation)
            print("look_around: At: Left eye yaw = {}, pitch = {}, Right eye yaw = {}, pitch = {}".format(
                ypl[0], ypl[1], ypr[0], ypr[1]))
            my_p = [4.0, 3.0, -9.0]
            verge_sensors_to_point(my_agent, my_robot, my_p)
            display_image(my_sim)
        elif k == ord('f'):  # towards the scene
            d = quaternion.from_rotation_vector([0.0, 0.0, 0.0])
            move = [0.0, 0.0, -delta_move]
            relative_move_and_rotate_agent(my_agent, d, move)
            display_image(my_sim)
        elif k == ord('b'):  # away from the scene
            d = quaternion.from_rotation_vector([0.0, 0.0, 0.0])
            move = [0.0, 0.0, delta_move]
            relative_move_and_rotate_agent(my_agent, d, move)
            display_image(my_sim)
        elif k == ord('j'):  # viewing to left side
            d = quaternion.from_rotation_vector([0.0, 0.0, 0.0])
            move = [delta_move, 0.0, 0.0]
            relative_move_and_rotate_agent(my_agent, d, move)
            display_image(my_sim)
        elif k == ord('k'):  # viewing to the right
            d = quaternion.from_rotation_vector([0.0, 0.0, 0.0])
            move = [-delta_move, 0.0, 0.0]
            relative_move_and_rotate_agent(my_agent, d, move)
            display_image(my_sim)
        elif k == ord('r'):  # rotate agent to your (viewer's) right by 15 degrees
            d = quaternion.from_rotation_vector([0.0, -np.pi / 12, 0.0])
            relative_move_and_rotate_agent(my_agent, d)
            display_image(my_sim)
        elif k == ord('l'):  # rotate agent to your (viewer's) left by 15 degrees
            d = quaternion.from_rotation_vector([0.0, np.pi / 12, 0.0])
            relative_move_and_rotate_agent(my_agent, d)
            display_image(my_sim)
        elif k == ord('u'):  # chin up
            d = quaternion.from_rotation_vector([np.pi / 12, 0.0, 0.0])
            relative_move_and_rotate_agent(my_agent, d)
            display_image(my_sim)
        elif k == ord('g'):  # chin down
            d = quaternion.from_rotation_vector([-np.pi / 12, 0.0, 0.0])
            relative_move_and_rotate_agent(my_agent, d)
            display_image(my_sim)
        elif k == ord('c'):  # converge - eyes are moving
            verge_sensors(np.pi/20,my_agent,'c')
            display_image(my_sim)
        elif k == ord('d'):  # diverge - eyes are moving
            verge_sensors(np.pi/20,my_agent,'d')
            display_image(my_sim)
        elif k == ord('n'):
            my_agent.set_state(original_state)
            display_image(my_sim)
        elif k == ord('m'):
            verge_sensors_to_midpoint_depth(my_agent, my_sim, my_robot)
            display_image(my_sim)
        elif k == ord('t'):
            toggle = 1
            if display == "a":
                display = "l"
            else:
                display = "a"



if __name__ == "__main__":

    print("The system version is {}".format(sys.version))
    '''
    R = rotatation_matrix_from_Habitat_to_Pybullet()
    my_vec = [0.2, -0.3, 0.5]
    H = homogenous_transform(R,my_vec)
    I = inverse_homogenous_transform(H)
    s1 = habitat_sim.geo.UP
    s2 = habitat_sim.geo.LEFT
    s3 = habitat_sim.geo.FRONT
    s4 = habitat_sim.geo.RIGHT
    '''
    oreo_robot = oreo.setup_oreo_in_pybullet()
    '''
    # test
    my_point = [0.3, 0.0, 0.0]
    the_angles = oreo_robot.compute_yaw_pitch_for_given_point(my_point)
    val = oreo_robot.get_actuator_positions_for_a_given_yaw_pitch(the_angles)
    if val[0] == 1:
        a_pos = val[1:]
        collision = oreo_robot.move_eyes_to_pos(a_pos)
        if collision == 0:
            orn_lefteye = oreo_robot.GetLinkOrientationWCS("left_eye_joint")
            # convert to numpy quaternion (w,x,y,z) w is the real part.
            orientation_lefteye = np.quaternion(orn_lefteye[3], orn_lefteye[0], orn_lefteye[1], orn_lefteye[2])
            orn_righteye = oreo_robot.GetLinkOrientationWCS("right_eye_joint")
            # convert to numpy quaternion (w,x,y,z) w is the real part.
            orientation_righteye = np.quaternion(orn_righteye[3], orn_righteye[0], orn_righteye[1], orn_righteye[2])
    # end of test
    '''

    the_sim, agent_id = setup_sim_and_sensors()
    o_agent = the_sim.get_agent(agent_id)
    agent_orn, agent_pos, sensors_pose = get_agent_sensor_position_orientations(o_agent)
    print("Initial Agent orientation = {}".format(agent_orn))
    print("Initial Left RGB Sensor orientation = {}".format(sensors_pose["left_rgb_sensor"].rotation))
    print("Initial RGB Sensor orientation = {}".format(sensors_pose["right_rgb_sensor"].rotation))
    print("Initial Sensor orientation = {}".format(sensors_pose["depth_sensor"].rotation))




    # get and save agent state
    # Not done
    #test_oreo_agent(o_agent, the_sim)
    look_around(o_agent, the_sim, oreo_robot, type="l")


