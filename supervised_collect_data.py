import numpy as np
from datetime import datetime
import os
from Supervised.entity.State import State
from rclpy.node import Node
from std_msgs.msg import String
import csv


class supervised_collect_data_node(Node):
    def __init__(self):
        super().__init__("Supervised_data_collection_node")
        self.get_logger().info("Node start")#ros2Ai #unity2Ros
        self.state = State()
        self.subscriber_fromUnity_thu_ROSbridge_ = self.create_subscription(
            String, 
            "Unity_2_AI", 
            self.callback_from_Unity, 
            10
        )
        self.subscriber_fromUnity_thu_ROSbridge_stopFlag = self.create_subscription(
            String, 
            "Unity_2_AI_stop_flag", 
            self.callback_from_Unity_stop_flag, 
            10
        )
        self.tokens = list() 

    def callback_from_Unity(self, msg):
        Unitystate = msg.data
        self.state.update(Unitystate)
        state_detect, token = self.state.get_wanted_features()
        if state_detect == 1:
             self.tokens.append(token)
        else:
            print("Unity lidar no signal.....")

    def callback_from_Unity_stop_flag(self, msg):
        print("saving data....")
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        csv_directory = os.path.join('.', 'dataset')
        csv_file_path = os.path.join(csv_directory, f'lstm_training_{timestamp}.csv')

        os.makedirs(csv_directory, exist_ok=True)
        
        with open(csv_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['token'])
            for item in self.tokens:
                csv_writer.writerow([item])
        self.tokens = list()
        print("finish saving !")        
