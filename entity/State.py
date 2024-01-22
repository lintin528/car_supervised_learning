from pydantic import BaseModel
from Supervised.entity.Coordinate import Coordinate 
from Supervised.entity.Velocity import Velocity 
from Supervised.entity.WheelOrientation import WheelOrientation
from Supervised.entity.WheelAngularVel import WheelAngularVel
import json
import math
import numpy as np

class StateType(BaseModel):
    final_target_pos: Coordinate
    car_pos: Coordinate
    car_vel: Velocity  # in ROS2 coordinate system
    car_orientation: float = 0  # radians, around ROS2 z axis, counter-clockwise: 0 - 359
    car_quaternion: list
    wheel_orientation: WheelOrientation  # around car z axis, counter-clockwise: +, clockwise: -, r/s
    car_angular_vel: float  # r/s, in ROS2 around car z axis, yaw++: -, yaw--: +, counter-clockwise: +, clockwise: -, in Unity:  counter-clockwise: -, clockwise: +
    wheel_angular_vel: WheelAngularVel  # around car wheel axis, front: +, back: -
    min_lidar: list = [] # meter
    min_lidar_direction: list = []

    # because orientation is transformed back to Unity coordinate system, here lidar direction alse needs to be transformed back from ROS2 to Unity
    # min_lidar_relative_angle: float # radian, base on car, right(x): 0, front(y): 90,  upper: 180 --->x 0, down: -180 --->x 0
    lidar_no_element_detect: int

class State:
    def __init__(self) -> None:
        self.prev_car_state_training = StateType(
                                                final_target_pos=Coordinate(x=0.0, y=0.0, z=0.0),
                                                car_pos=Coordinate(x=0.0, y=0.0, z=0.0),
                                                car_vel=Velocity(x=0.0, y=0.0, z=0.0),
                                                car_orientation=0.0,
                                                car_quaternion=[],
                                                wheel_orientation=WheelOrientation(left_front=0.0, right_front=0.0),
                                                car_angular_vel=0.0,
                                                wheel_angular_vel=WheelAngularVel(left_back=0.0, 
                                                                                  left_front=0.0, 
                                                                                  right_back=0.0,
                                                                                  right_front=0.0),
                                                min_lidar=[],
                                                min_lidar_direciton=[0.0],
                                                lidar_no_element_detect=0
                                                )

        self.current_car_state_training = self.prev_car_state_training
    
    def __euler_from_quaternion(self, orientation):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x = orientation[0]
        y = orientation[1]
        z = orientation[2]
        w = orientation[3]

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z  # in radians

    def __radToPositiveRad(self, rad):
        # left +, right -, up 0, down 180 => clockwise: 0 - 359
        if rad < 0:
            rad = -rad
        elif rad > 0:
            rad = math.pi * 2 - rad

        return rad
    
    def round_to_decimal_places(self, data_list, decimal_places=3):
        """
        Round the elements of a list to a specified number of decimal places.
        """
        return [round(num, decimal_places) for num in data_list]
    
    def __parse_json(self, data):
        obs = json.loads(data)

        for key, value in obs.items():
            if isinstance(value, str) and value.startswith('(') and value.endswith(')'):
                coordinate_str = value.strip('()')
                coordinates = list(map(float, coordinate_str.split(',')))
                obs[key] = coordinates
        return obs
    
    def get_yaw_from_quaternion(self, z, w):
        return np.degrees(2 * np.arctan2(z, w))

    def get_direction_vector(self, current_position, target_position):
        return np.array(target_position) - np.array(current_position)

    def get_angle_to_target(self, car_yaw, direction_vector):
        target_yaw = np.arctan2(direction_vector[1], direction_vector[0])
        angle_diff = target_yaw - np.radians(car_yaw)
        return (np.degrees(angle_diff)) % 360

    def calculate_angle_point(self, car_quaternion_1, car_quaternion_2, car_pos, target_pos):
        car_pos = [car_pos.x, car_pos.y]
        target_pos = [target_pos.x, target_pos.y]
        car_yaw = self.get_yaw_from_quaternion(car_quaternion_1, car_quaternion_2)
        direction_vector = self.get_direction_vector(car_pos, target_pos)

        angle_to_target = self.get_angle_to_target(car_yaw, direction_vector)
        angle_diff = angle_to_target - 180
        # angle_diff = angle_to_target - 360 if angle_to_target > 180 else angle_to_target
        
        return angle_diff

    def __decode(self, obs):
        data = self.__parse_json(obs)
        
        car_quaternion = [data['ROS2CarQuaternion'][0], data['ROS2CarQuaternion'][1],
                          data['ROS2CarQuaternion'][2], data['ROS2CarQuaternion'][3]]
        
        car_roll_x, car_pitch_y, car_yaw_z = self.__euler_from_quaternion(car_quaternion)
        car_orientation = self.__radToPositiveRad(car_yaw_z)

        wheel_quaternion_left_front = [data['ROS2WheelQuaternionLeftFront'][0],
                                       data['ROS2WheelQuaternionLeftFront'][1],
                                       data['ROS2WheelQuaternionLeftFront'][2],
                                       data['ROS2WheelQuaternionLeftFront'][
                                           3]]  # 48 49 50 51ROS2WheelQuaternionRightBack
        wheel_left_front_roll_x, wheel_left_front_pitch_y, wheel_left_front_yaw_z = self.__euler_from_quaternion(
            wheel_quaternion_left_front)

        wheel_quaternion_right_front = [data['ROS2WheelQuaternionRightFront'][0],
                                        data['ROS2WheelQuaternionRightFront'][1],
                                        data['ROS2WheelQuaternionRightFront'][2],
                                        data['ROS2WheelQuaternionRightFront'][3]]
        wheel_right_front_roll_x, wheel_right_front_pitch_y, wheel_right_front_yaw_z = self.__euler_from_quaternion(
            wheel_quaternion_right_front)
        
        return car_orientation, wheel_left_front_yaw_z, wheel_right_front_yaw_z, car_quaternion, data
        
        
    def update(self, obs):
        car_orientation, wheel_left_front_yaw_z, wheel_right_front_yaw_z, car_quaternion, data = self.__decode(obs)
        
        lidar_data = data.get('ROS2Range', [])
        lidar_no_element_detect = int(bool(lidar_data))

        self.prev_car_state_training = self.current_car_state_training
        self.current_car_state_training = StateType(
            final_target_pos=Coordinate(x=data['ROS2TargetPosition'][0],
                                       y=data['ROS2TargetPosition'][1],
                                       z=0.0),
            car_pos=Coordinate(x=data['ROS2CarPosition'][0],
                              y=data['ROS2CarPosition'][1],
                              z=data['ROS2CarPosition'][1]),
                              
            car_vel=Velocity(x=data['ROS2CarVelocity'][0],
                              y=data['ROS2CarVelocity'][1],
                              z=0.0),
            car_orientation=car_orientation,
            car_quaternion=car_quaternion,
            wheel_orientation=WheelOrientation(left_front=self.__radToPositiveRad(wheel_left_front_yaw_z), \
                                               right_front=self.__radToPositiveRad(wheel_right_front_yaw_z)),

            car_angular_vel=data['ROS2CarAugularVelocity'][2],
            wheel_angular_vel=WheelAngularVel(left_back=data['ROS2WheelAngularVelocityLeftBack'][1],
                                              left_front=data['ROS2WheelAngularVelocityLeftFront'][1],
                                              right_back=data['ROS2WheelAngularVelocityRightBack'][1],
                                              right_front=data['ROS2WheelAngularVelocityRightFront'][1]
                                              ),
            min_lidar=data['ROS2Range'],
            min_lidar_direction=data["ROS2RangePosition"],
            lidar_no_element_detect=lidar_no_element_detect,
        )

    def get_wanted_features(self):
        pattern = []
        if self.current_car_state_training.wheel_angular_vel.left_back > 0 and self.current_car_state_training.wheel_angular_vel.right_back > 0:
            pattern = [1, 0, 0, 0]
        elif self.current_car_state_training.wheel_angular_vel.left_back < 0 and self.current_car_state_training.wheel_angular_vel.right_back > 0:
            pattern = [0, 1, 0, 0]
        elif self.current_car_state_training.wheel_angular_vel.left_back > 0 and self.current_car_state_training.wheel_angular_vel.right_back < 0:
            pattern = [0, 0, 1, 0]
        else :
            pattern = [0, 0, 0, 1]
        wheel_angular_vel_list = [self.current_car_state_training.wheel_angular_vel.left_back, self.current_car_state_training.wheel_angular_vel.right_back]
        car_target_distance = math.sqrt((self.current_car_state_training.car_pos.x - self.current_car_state_training.final_target_pos.x)**2 + 
                                        (self.current_car_state_training.car_pos.y - self.current_car_state_training.final_target_pos.y)**2)
        car_target_distance = self.round_to_decimal_places([car_target_distance])

        car_quaternion = self.round_to_decimal_places(self.current_car_state_training.car_quaternion[2:4])
        angle_diff = self.calculate_angle_point(car_quaternion[0], car_quaternion[1], 
                                                self.current_car_state_training.car_pos, 
                                                self.current_car_state_training.final_target_pos)
        
        angle_diff = self.round_to_decimal_places([angle_diff])
        lidar_18 = []
        minimum = 999.0
        for index in range(180):
            minimum = min(self.current_car_state_training.min_lidar[index], minimum)
            if index % 10 == 0:
                lidar_18.append(minimum)
                minimum = 999.0
        lidar_18 = self.round_to_decimal_places(lidar_18)
        token = str(car_target_distance + angle_diff + lidar_18 + wheel_angular_vel_list + pattern)
        return self.current_car_state_training.lidar_no_element_detect, token
