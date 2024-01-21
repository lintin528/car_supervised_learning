from pydantic import BaseModel

class ROS2Point(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0