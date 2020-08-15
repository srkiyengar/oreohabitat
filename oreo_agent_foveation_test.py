import habitat_sim
import numpy as np
import quaternion
import sys
import cv2

from typing import Any, Dict, List, Union

import attr

@attr.s(auto_attribs=True, slots=True)
class SixDOFPose(object):
    r"""Specifies a position with 6 degrees of freedom

    :property position: xyz position
    :property rotation: unit quaternion rotation
    """

    position: np.ndarray = np.zeros(3)
    rotation: Union[np.quaternion, List] = np.quaternion(1, 0, 0, 0)



def rotation_to_new_point(my_coordinates, current_direction):
    '''

    :param my_coordinates: x,y,z coordinate of the point where the camera should point
    Rotate the camera from its current direction to the new direction
    :param current_direction: unit vector corresponding to current direction as a list in the forward direction
    :return: quaternion representing the new rotation
    '''
    new_direction = np.asarray(my_coordinates,dtype=np.float)
    new_direction = new_direction/np.linalg.norm(new_direction)
    orth_vector = np.cross(current_direction,new_direction)

    # vector a x b has a magnitude |a||b|sin(angle) and we want a unit vector
    orth_vector = orth_vector/np.linalg.norm(orth_vector)     #unit vector representing the axis

    rot_angle = np.arccos(np.clip(np.dot(new_direction,current_direction),-1.0,1.0)) #angle of rotation
    my_axis_angle = rot_angle*orth_vector
    my_quat = quaternion.from_rotation_vector(my_axis_angle)
    return my_quat

def forward_vector_from_quaternion(Q):
    '''
    :param Q: quaternion
    :return: 3x1 list which represents the unit vector corresponding to the z-axis of the current pose
    '''
    R = quaternion.as_rotation_matrix(Q)
    L = R[:,2]
    return L

def create_test_sensor(sensor_uuid="my_sensor", camera_type="C"):

    """orientation - Axis Angle representation
       camera_type "C" is RGB Camera and "D" is Depth."""

    new_sensor = habitat_sim.SensorSpec()
    new_sensor.uuid = sensor_uuid
    if camera_type == "D":
        new_sensor.sensor_type = habitat_sim.SensorType.DEPTH
    else:
        new_sensor.sensor_type = habitat_sim.SensorType.COLOR
    new_sensor.resolution = [512, 512]


    return new_sensor
def create_sensor(orientation, position=[0.0,0.0,0.0], sensor_resolution=[512, 512],
                  sensor_uuid="my_sensor", camera_type="C"):

    """orientation - Axis Angle representation 3 values and describes the rotation with respect to agent.
       camera_type "C" is RGB Camera and "D" is Depth."""

    new_sensor = habitat_sim.SensorSpec()
    new_sensor.uuid = sensor_uuid
    if camera_type == "D":
        new_sensor.sensor_type = habitat_sim.SensorType.DEPTH
    else:
        new_sensor.sensor_type = habitat_sim.SensorType.COLOR
    new_sensor.resolution = sensor_resolution

    new_sensor.position = position  # +ive x is to the right, +ive y is UP, -ive z is in front
    new_sensor.orientation = np.array(orientation,dtype=float)

    return new_sensor


if __name__ == "__main__":
    print("The system version is {}".format(sys.version))

    left_rgb_sensor: object = create_sensor([0.0, 0.0, 0.0],sensor_uuid="left_rgb_sensor")
    right_rgb_sensor: object = create_sensor([0.0, 0.0, 0.0], sensor_uuid="right_rgb_sensor")
    depth_sensor: object = create_test_sensor(sensor_uuid="depth_sensor", camera_type="D")

    # agent configuration has the sensor_specifications objects as a list
    my_agent_config = habitat_sim.AgentConfiguration()
    my_agent_config.sensor_specifications = [left_rgb_sensor, right_rgb_sensor, depth_sensor]

    # Configuration of the backend of the simulator includes default_agent_id set to 0
    backend_cfg = habitat_sim.SimulatorConfiguration()
    default_agent_id = backend_cfg.default_agent_id
    backend_cfg.foveation_distortion = False
    backend_cfg.scene.id = (
        "/Users/rajan/My_Replica/replica_v1/room_2/habitat/mesh_semantic.ply"
    )

    # Tie the backend of the simulator and a list of agent configurations
    my_Configuration = habitat_sim.Configuration(backend_cfg, [my_agent_config])
    my_sim = habitat_sim.Simulator(my_Configuration)
    print("Initial Left sensor orientation = {}".format(left_rgb_sensor.orientation))
    my_agent = my_sim.get_agent(default_agent_id)
    my_agent_state = my_agent.get_state()
    print("Agent orientation = {}".format(my_agent_state.rotation))
    print("Agent-Left RGB Sensor orientation = {}".format(my_agent_state.sensor_states['left_rgb_sensor'].rotation))

    my_agent_state.sensor_states['right_rgb_sensor'].rotation = quaternion.from_rotation_vector([0.0, 0.0, 0.0])
    my_agent_state.sensor_states['depth_sensor'].rotation = quaternion.from_rotation_vector([0.0, 0.0, 0.0])


    for i in list(range(9)):
        rot_v = [0.0, i*np.pi/4.0, 0.0]
        print(rot_v)
        d = quaternion.from_rotation_vector([0.0, i*np.pi/4.0, 0.0])
        my_agent_state.rotation = d
        my_agent.set_state(my_agent_state)
        ang = quaternion.as_rotation_vector(my_agent_state.sensor_states['left_rgb_sensor'].rotation)
        print("Agent-Left RGB Sensor orientation = {}".format(ang))
        for _, sensor in my_sim._sensors.items():
            sensor.draw_observation()

        observations = {}
        for sensor_uuid, sensor in my_sim._sensors.items():
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
        cv2.imshow("stereo_pair", stereo_pair)
        k = cv2.waitKey(0)
        if k == ord('q'):
            break
    a = "end"


