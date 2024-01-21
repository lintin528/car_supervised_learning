import json
import math
import numpy as np

def parse_json_to_dict(json_str):
    """
    Convert a JSON string to a dictionary, processing special string formats into coordinate lists.
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return {}

    for key, value in data.items():
        if isinstance(value, str) and value.startswith('(') and value.endswith(')'):
            coordinates = list(map(float, value.strip('()').split(',')))  
            data[key] = coordinates
    return data

def get_90_smallest_lidar_values(lidar_data):
    """
    Divide the lidar data into 12 chunks and return the smallest value from each chunk.
    """
    chunk_size = len(lidar_data) // 2
    return [min(lidar_data[i:i + chunk_size]) for i in range(0, len(lidar_data), chunk_size)]

def normalize_lidar_values(lidar_data):
    """
    Normalize the lidar data to a range of 0 to 1.
    """
    min_val, max_val = min(lidar_data), max(lidar_data)
    return [(x - min_val) / (max_val - min_val) if max_val - min_val else 0 for x in lidar_data]

def round_to_decimal_places(data_list, decimal_places=3):
    """
    Round the elements of a list to a specified number of decimal places.
    """
    return [round(num, decimal_places) for num in data_list]

def clamp_number(number, min_val=-10, max_val=10):
    """
    Clamp the number to a specified range.
    """
    return max(min_val, min(max_val, number))

def quaternion_to_car_orientation(x, y, z, w):
    # 左邊0~180 右邊 0~-180
    length = math.sqrt(x*x + y*y + z*z + w*w)
    x /= length
    y /= length
    z /= length
    w /= length
    
    # 計算角度（弧度）
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    
    yaw_degrees = math.degrees(yaw)
    
    return yaw_degrees

def calculate_angle(coord1, coord2):
    # 将列表转换为 NumPy 数组
    vector1 = np.array(coord1)
    vector2 = np.array(coord2)

    # 计算向量的点积
    dot_product = np.dot(vector1, vector2)

    # 计算向量的模
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    # 计算夹角
    angle = np.arccos(dot_product / (norm1 * norm2))

    # 将弧度转换为度
    angle_deg = np.degrees(angle)

    return angle_deg

def transfer_obs(obs):
    """
    Process observation data from Unity and return a token and a flag indicating lidar detection.
    """
    obs = parse_json_to_dict(obs)

    lidar_data = obs.get('ROS2Range', [])
    lidar_no_element_detect = int(bool(lidar_data))

    # if lidar_data:
    #     lidar_data = normalize_lidar_values(lidar_data)
    lidar_data = round_to_decimal_places(lidar_data)

    wheel_angular_vel = [
        obs['ROS2WheelAngularVelocityLeftBack'][1],
        obs['ROS2WheelAngularVelocityRightBack'][1]
    ]
    wheel_angular_vel = round_to_decimal_places(wheel_angular_vel)

    car_quaternion = round_to_decimal_places(obs['ROS2CarQuaternion'][2:4])

    car_orientation = [quaternion_to_car_orientation(0,0,car_quaternion[0], car_quaternion[1])]
    car_orientation = round_to_decimal_places(car_orientation)

    car_pos, target_pos = obs['ROS2CarPosition'], obs['ROS2TargetPosition']
    car_target_distance = math.sqrt((car_pos[0] - target_pos[0])**2 + (car_pos[1] - target_pos[1])**2)
    car_target_distance = round_to_decimal_places([car_target_distance])[0]

    car_target_angle = [calculate_angle([car_pos[0],car_pos[1]], [target_pos[0], target_pos[1]])]
    car_target_angle = round_to_decimal_places(car_target_angle)

    target_flag = 0
    if car_target_distance <=1:
        target_flag = 1
    
    if len(wheel_angular_vel) != 2:
        print("error")

    pattern = 0
    #前進0，左轉1，右轉2
    if wheel_angular_vel[0] > 0 and wheel_angular_vel[1] > 0:
        pattern = 0
    elif wheel_angular_vel[0] < 0 and wheel_angular_vel[1] > 0:
        pattern = 1
    elif wheel_angular_vel[0] > 0 and wheel_angular_vel[1] < 0:
        pattern = 2

    lidar_18 = []
    minimum = 999.0
    for index in range(180):
        minimum = min(lidar_data[index], minimum)
        if index % 10 == 0:
            lidar_18.append(minimum)
            minimum = 999.0
    lidar_18 = round_to_decimal_places(lidar_18)
    
    angle_diff = calculate_angle_point(car_quaternion[0], car_quaternion[1], car_pos, target_pos)
    

    token = str([car_target_distance] + [angle_diff] + lidar_18 + [target_flag] + wheel_angular_vel + [pattern])
    return lidar_no_element_detect, token

def get_yaw_from_quaternion(z, w):
    return np.degrees(2 * np.arctan2(z, w))


def get_direction_vector(current_position, target_position):
    return np.array(target_position) - np.array(current_position)


def get_angle_to_target(car_yaw, direction_vector):
    target_yaw = np.arctan2(direction_vector[1], direction_vector[0])
    angle_diff = target_yaw - np.radians(car_yaw)

    return (np.degrees(angle_diff)) % 360


def calculate_angle_point(car_quaternion_1, car_quaternion_2, car_pos, target_pos):
    car_yaw = get_yaw_from_quaternion(car_quaternion_1, car_quaternion_2)
    direction_vector = get_direction_vector(car_pos, target_pos)
    angle_to_target = get_angle_to_target(car_yaw, direction_vector)
    # angle_diff = np.abs(angle_to_target - 180)
    angle_diff = angle_to_target - 360 if angle_to_target > 180 else angle_to_target
    return angle_diff
