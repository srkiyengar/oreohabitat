import habitat_sim
import numpy as np
import quaternion
import math

eye_separation = 0.058
sensor_resolution = [512,512]
scene = "../multi_agent/data_files/skokloster-castle.glb"
# scene = "/Users/rajan/My_Replica/replica_v1/apartment_1/mesh.ply"


def calculate_rotation_to_new_direction(uvector):
    '''
    Computes Axis angle for rotating the -z axis from (0,0,-1) (principle direction) to align with the unit vector.
    :param uvector: numpy array unit vector which is the new direction of the -z axis
    :return:
    '''

    v1 = uvector
    v2 = np.array([0.0, 0.0, -1.0])
    # my_axis is v2 cross v1
    my_axis = np.cross(v2, v1)
    my_axis = my_axis / np.linalg.norm(my_axis)
    my_angle = np.arccos(np.dot(v1, v2))
    my_axis_angle = my_angle * my_axis
    quat = quaternion.from_rotation_vector(my_axis_angle)
    return quat


class agent_oreo(object):
    # constructor
    def __init__(self, depth_camera=False, pos_depth_cam = 'c', foveation=False):

        self. agent_config = habitat_sim.AgentConfiguration()
        # Left sensor - # oreo perspective - staring at -ive z
        self.left_sensor = habitat_sim.SensorSpec()
        self.left_sensor.sensor_type = habitat_sim.SensorType.COLOR
        self.left_sensor.resolution = sensor_resolution
        self.left_sensor.uuid = "left_rgb_sensor"
        self.left_sensor.position = [-eye_separation / 2, 0.0, 0.0]
        self.left_sensor.orientation = np.array([0.0,0.0,0.0], dtype=float)
        left_sensor_hfov = math.radians(int(self.left_sensor.parameters['hfov']))
        self.focal_distance = sensor_resolution[0]/2*math.tan(left_sensor_hfov/2)

        # Right sensor - # oreo perspective - staring at -ive z
        self.right_sensor = habitat_sim.SensorSpec()
        self.right_sensor.sensor_type = habitat_sim.SensorType.COLOR
        self.right_sensor.resolution = sensor_resolution
        self.right_sensor.uuid = "right_rgb_sensor"
        self.right_sensor.position = [eye_separation / 2, 0.0, 0.0]
        self.right_sensor.orientation = np.array([0.0, 0.0, 0.0], dtype=float)
        right_sensor_hfov = math.radians(int(self.right_sensor.parameters['hfov']))
        if right_sensor_hfov != left_sensor_hfov:
            print("Warning - Right and LEft Sensor widths are not identical!")

        # Depth camera - At the origin of the reference coordinate axes (habitat frame)
        if depth_camera:
            self.num_sensors = 3
            self.depth_sensor = habitat_sim.SensorSpec()
            self.depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
            self.depth_sensor.resolution = sensor_resolution
            self.depth_sensor.uuid = "depth_sensor"
            if pos_depth_cam == 'l':
                self.depth_sensor.position = self.left_sensor.position
            elif pos_depth_cam == 'r':
                self.depth_sensor.position = self.right_sensor.position
            else:
                self.depth_sensor.position = [0.0,0.0,0.0]

            self.depth_sensor.orientation = np.array([0.0, 0.0, 0.0], dtype=float)
            self.agent_config.sensor_specifications = [self.right_sensor, self.left_sensor, self.depth_sensor]
        else:
            self.num_sensors = 2
            self.agent_config.sensor_specifications = [self.right_sensor, self.left_sensor]

        self.backend_cfg = habitat_sim.SimulatorConfiguration()

        if foveation:
            self.backend_cfg.foveation_distortion = True

        self.backend_cfg.scene.id = scene

        # Tie the backend of the simulator and the list of agent configurations (only one)
        self.sim_configuration = habitat_sim.Configuration(self.backend_cfg, [self.agent_config])
        self.sim = habitat_sim.Simulator(self.sim_configuration)
        self.agent_id = self.backend_cfg.default_agent_id
        self.agent = self.sim.get_agent(self.agent_id)
        return

    def get_agent_sensor_position_orientations(self):
        """
        :return:
        agent orientation = a quaternion
        agent position = numpy array
        num_sensors = 2 (left and right) or 3 (left, right, depth)
        sensor[num_sensor] orientation quaternions. The sensor_states are with respect to habitat frame.
        Habitat frame is cameras staring at -ive z and the +ive y is UP.

        Internally the relative sensor orientation and translation with respect to agent are stored
        under _sensor but the sensor_states that are accessible are wrt habitat frame.
        The sensor states (eyes) wrt to the agent state (head/neck) is NOT computed by this function.
        """

        agent_state = self.agent.get_state()
        agent_orientation = agent_state.rotation
        agent_position = agent_state.position
        sensors_position_orientation = agent_state.sensor_states
        s1 = agent_state.sensor_states["left_rgb_sensor"].rotation
        s2 = agent_state.sensor_states["right_rgb_sensor"].rotation

        if self.num_sensors == 3:
            s3 = agent_state.sensor_states["depth_sensor"].rotation
            return agent_orientation, agent_position, s1, s2, s3
        else:
            return agent_orientation, agent_position, s1, s2

    def rotate_sensor_wrt_agent(self, sensors_rotation):
        """
        The agent position or orientation is unchanged.
        sensors_rotation is specified with respect to the agent frame.
        The sensor_states are with respect to habitat frame. Note: The relative rotation
        between agent and sensor is saved under protected '_sensor' by the set_agent_state API function
        :param my_agent: agent object
        :param sensors_rotation: list of quaternions - rotation of the sensors with respect to agent frame
        :return: nothing
        """

        agent_state = self.agent.get_state()
        agent_orn = agent_state.rotation
        agent_state.sensor_states["left_rgb_sensor"].rotation = agent_orn * sensors_rotation[0]
        agent_state.sensor_states["right_rgb_sensor"].rotation = agent_orn * sensors_rotation[1]
        if self.num_sensors ==3:
            agent_state.sensor_states["depth_sensor"].rotation = agent_orn * sensors_rotation[2]

        self.agent.set_state(agent_state, infer_sensor_states=False)
        return

    def compute_uvector_for_image_point(self, x_pos, y_pos):
        '''
        The x, y, z values are expressed in pixels. The purpose is to compute unit vector.
        z_pos is given by the distance 'f' from the principle point to the sensor image.
        :param x_pos:
        :param y_pos:
        :return: np array, a unit vector pointing in the direction of the image point
        '''

        #shifting the origin from 0,0 to width/2, height/2
        xval = x_pos - (self.left_sensor.resolution[0]/2)       # width
        yval = y_pos - (self.left_sensor.resolution[1]/2)       # height

        v = np.array[xval, yval, -self.focal_distance]
        unit_vec = v / np.linalg.norm(v)
        return unit_vec

