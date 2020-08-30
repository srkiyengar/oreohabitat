import habitat_sim
import numpy as np
import quaternion
import sys
import cv2

from typing import Any, Dict, List, Union

# pybullet quaternion order x, y, z, w
# numpy quaternion order w, x, y, z
# Whenever a quaternion is specified it is in numpy quaternion convention

import attr

eye_seperation = 0.058


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
    #
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
    # backend_cfg.scene.id = (
    #    "/Users/rajan/My_Replica/replica_v1/room_2/mesh.ply"
    # )
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
    get_agent_sensor_position_orientations are both with respect to the habitat frame. Internally the relative sensor orientation
    and translations with respect to agent are stored under _sensor but the sensor_states that are accessible are
    wrt habitat frame only.
    """

    my_agent_state = my_agent.get_state()
    agent_orientation = my_agent_state.rotation
    agent_position = my_agent_state.position
    sensors_position_orientation = my_agent_state.sensor_states
    return agent_orientation, agent_position, sensors_position_orientation


def rotate_sensor_wrt_stationary_agent_frame(my_agent, sensors_rotation):
    """
    The sensors_rotation is specified with respect to the agent frame.
    This function will set the sensors orientation in habitat  w.r.t. to habitat frame.
    The agent is not moving or changing orientation.
    It is important to remember that sensor_states are with respect to habitat frame while the relative rotation
    between agent and sensor is saved within protected _sensor by the set agent sensor API function
    This movement is similar to left and right eye movements of OREO with respect to the frame of the skull.
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
    my_rotation = r1 * rel_rotation
    move = t1 + rel_move

    my_agent_state.rotation = my_rotation  # new rotation of Agent wrt to Habitat
    if rel_move == [0.0, 0.0, 0.0]:  # only rotations are impacted and not position
        my_agent_state.sensor_states["left_rgb_sensor"].rotation = \
            my_rotation * (r1_inverse * my_agent_state.sensor_states["left_rgb_sensor"].rotation)
        my_agent_state.sensor_states["right_rgb_sensor"].rotation = \
            my_rotation * (r1_inverse * my_agent_state.sensor_states["right_rgb_sensor"].rotation)
        my_agent_state.sensor_states["depth_sensor"].rotation = \
            my_rotation * (r1_inverse * my_agent_state.sensor_states["depth_sensor"].rotation)
    else:
        # h1 is current habitat to Agent homogenous transformation (HT), h1new is habitat to Agent with new one
        # h3 - habitat to sensor which is h1 or h1new multiplied and agent to the sensor HT
        # Example - h1new(left_rgb)*h1_inv*h3 where h1_inv*h3 provides sensor pose wrt to agent.
        # Import to remember that get agent_state provides position, orientation wrt to habitant
        my_agent_state.position = move
        h1 = homogenous_transform(quaternion.as_rotation_matrix(r1), list(t1))
        h1_inv = inverse_homogenous_transform(h1)
        h1new = homogenous_transform(quaternion.as_rotation_matrix(my_rotation), list(move))
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


'''
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

    quat = quaternion.from_rotation_matrix(R)  # w, x, y, z numpy quaternion order
    return R, quat


def rotatation_matrix_from_Pybullet_to_Habitat():
    R = np.zeros((3, 3))
    R[0, 2] = -1
    R[1, 0] = -1
    R[2, 1] = 1

    quat = quaternion.from_rotation_matrix(R)  # w, x, y, z numpy quaternion order
    return R, quat


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
    depth = np.clip(depth, 0, 10)
    depth /= 10.0
    stereo_pair = np.concatenate([rgb_left, rgb_right], axis=1)

    return stereo_pair, depth


if __name__ == "__main__":

    print("The system version is {}".format(sys.version))
    s1 = habitat_sim.geo.UP
    s2 = habitat_sim.geo.LEFT
    s3 = habitat_sim.geo.FRONT
    s4 = habitat_sim.geo.RIGHT

    new_sim, agent_id = setup_sim_and_sensors()
    new_agent = new_sim.get_agent(agent_id)
    agent_orn, agent_pos, sensors_pose = get_agent_sensor_position_orientations(new_agent)
    print("Initial Agent orientation = {}".format(agent_orn))
    print("Initial Left RGB Sensor orientation = {}".format(sensors_pose["left_rgb_sensor"].rotation))
    print("Initial RGB Sensor orientation = {}".format(sensors_pose["right_rgb_sensor"].rotation))
    print("Initial Sensor orientation = {}".format(sensors_pose["depth_sensor"].rotation))

    # get and save agent state
    # Not done
    init_pos = move = [0.9539339, 0.1917877, 12.163067]
    N = 16  # number of moves or rotations
    j = 0
    d = quaternion.from_rotation_vector([0.0, 0.0, 0.0])
    for i in list(range(N)):
        move = [0.1*i, 0.0, 0.0]
        relative_move_and_rotate_agent(new_agent, d, move)
        stereo_image, depth_image = get_sensor_observations(new_sim)
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
        stereo_image, depth_image = get_sensor_observations(new_sim)
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
    # Testing relative move and rotate agent
    d = quaternion.from_rotation_vector([0.0, np.pi/N, 0.0])
    for i in list(range(N)):
        move = [0.1 * i, 0.0, 0.0]
        relative_move_and_rotate_agent(new_agent, d, move)
        stereo_image, depth_image = get_sensor_observations(new_sim)
        cv2.imshow("stereo_pair", stereo_image)
        cv2.imshow("depth", depth_image)
        k = cv2.waitKey(0)
        if k == ord('q'):
            break
        else:
            j += 1
            print("{})Key input {}".format(j, k))

    cv2.destroyAllWindows()



