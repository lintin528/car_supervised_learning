from pydantic import BaseModel

class ROS2Point(BaseModel):
    x: float
    y: float
    z: float

# class lidar_direciton(BaseModel):
#     list(float, float, float)

# around ROS2 z axis, left +, right -, up 0, down 180
class WheelOrientation(BaseModel):
    left_front: float=0
    right_front: float=0

# around car wheel axis, front: +, back: -, r/s
class WheelAngularVel(BaseModel):
    left_back: float
    right_back: float

class State(BaseModel):
    targetRelativeToCar: ROS2Point
    car_quaternion: list # unity quaternion z & w
    wheel_angular_vel: WheelAngularVel # around car wheel axis, front: +, back: -
    min_lidar: list # meter

class ControlSignal(BaseModel):
    wheel_vel: float # rad/s
    steering_angle: float # degree, left: -, right: +