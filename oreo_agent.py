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


def get_agent_sensor_orientations(my_agent):
    """
    :param my_agent: object obtained from setting sim and agent sensors
    :return: agent orientation as a quat and sensor orientation as quats with in a dict with sensor-id as keys
    """

    my_agent_state = my_agent.get_state()
    agent_orientation = my_agent_state.rotation
    sensors_orientation = my_agent_state.sensor_states
    return agent_orientation, sensors_orientation


def rotate_sensor_wrt_stationary_agent_frame(my_agent, sensors_rotation):
    """
    The sensors_rotations are specified with respect to the agent frame.
    This function sets the sensors orientation in habitat  w.r.t. to habitat frame.
    The agent is not moving or changing orientation.
    This is similar to left and right eye movements of OREO with respect to the frame of the skull.
    :param my_agent: agent object
    :param sensors_rotation: list of quaternions - rotation of the sensors with respect to agent frame
    :return: nothing
    """

    my_agent_state = my_agent.get_state()
    agent_orn = my_agent_state.rotation
    my_agent_state.sensor_states["left_rgb_sensor"].rotation = agent_orn*sensors_rotation[0]
    my_agent_state.sensor_states["right_rgb_sensor"].rotation = agent_orn*sensors_rotation[1]
    my_agent_state.sensor_states["depth_sensor"].rotation = agent_orn*sensors_rotation[2]
    my_agent.set_state(my_agent_state, infer_sensor_states=False)
    return


def move_and_rotate_agent(my_agent, my_rotation, move=[0,0,0.0,0.0]):
    """
    The agent has a translation and rotation. The sensor orientation and position wrt to habitat changes due to it.
    sensors does not change their orientation with respect to agent frame
    :param my_agent: agent object
    :param my_rotation: quaternion with respect to habitat frame
    :param move: in habitat frame [x,y,z]
    :return:
    """

    my_agent_state = my_agent.get_state()
    r_inverse = my_agent_state.rotation.inverse()
    my_agent_state.rotation = my_rotation

    my_agent_state.sensor_states["left_rgb_sensor"].rotation = \
        my_rotation*(r_inverse*my_agent_state.sensor_states["left_rgb_sensor"].rotation)
    my_agent_state.sensor_states["right_rgb_sensor"].rotation = \
        my_rotation*(r_inverse*my_agent_state.sensor_states["right_rgb_sensor"].rotation)
    my_agent_state.sensor_states["depth_sensor"].rotation = \
        my_rotation*(r_inverse*my_agent_state.sensor_states["depth_sensor"].rotation)

    if move != [0.0,0.0,0.0]:
        # Add code to determine the absolute position of the sensors w.r.t. habitat frame.
        # compose 3 - H1new*H1_inv*H3, before change of agent pose: H3 - habitat to sensor , H1 is habitat to Agent
        # H1new is after change of agent pose.
        pass
    my_agent.set_state(my_agent_state, infer_sensor_states=False)
    return


def get_sensor_observations(oreo_sim):
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

    return stereo_pair,depth


# Rotation from Pybullet to Habitatai


def rotatation_matrix_from_Pybullet_to_Habitat_frame():
    R = np.zeros((3, 3))
    R[0,2] = 1
    R[1,0] = 1
    R[2,1] = 1

    quat = quaternion.from_rotation_matrix(R)       # w, x, y, z numpy quaternion order
    return R, quat

# Inverse of the rotation from Pybullet to Habitatai


def rotatation_matrix_from_Habitat_to_Pybullet_frame():

    R = np.zeros((3, 3))
    R[0,1] = 1
    R[1,2] = 1
    R[2,0] = 1

    quat = quaternion.from_rotation_matrix(R)       # w, x, y, z numpy quaternion order
    return R, quat


if __name__ == "__main__":
    print("The system version is {}".format(sys.version))

    new_sim, agent_id = setup_sim_and_sensors()
    new_agent = new_sim.get_agent(agent_id)
    agent_orn, sensors_orn = get_agent_sensor_orientations(new_agent)
    print("Initial Agent orientation = {}".format(agent_orn))
    print("Initial Left RGB Sensor orientation = {}".format(sensors_orn["left_rgb_sensor"].rotation))
    print("Initial RGB Sensor orientation = {}".format(sensors_orn["right_rgb_sensor"].rotation))
    print("Initial Sensor orientation = {}".format(sensors_orn["depth_sensor"].rotation))
    j=0
    for i in list(range(9)):
        rot_v = [0.0, i*np.pi/4.0, 0.0]
        d = quaternion.from_rotation_vector([0.0, i*np.pi/4.0, 0.0])
        #rotate_sensor_wrt_stationary_agent_frame(new_agent, [d,d,d])
        move_and_rotate_agent(new_agent,d,move=[1,1,1])
        stereo_image, depth_image = get_sensor_observations(new_sim)
        cv2.imshow("stereo_pair", stereo_image)
        cv2.imshow("depth", depth_image)
        k = cv2.waitKey(0)
        if k == ord('q'):
            break
        else:
            j+=1
            print("{})Key input {}".format(j,k))

    a = "end"


