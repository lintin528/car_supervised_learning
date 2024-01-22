import numpy as np
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
import torch
import torch.nn as nn
import torch.nn.functional as F
from entity.State import State

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class supervised_inference_node(Node):
    def __init__(self):
        super().__init__("ROS2_Node")
        self.get_logger().info("supervised learning inference")#ros2Ai #unity2Ros
        self.state = State()
        self.subscriber_fromUnity_thu_ROSbridge_ = self.create_subscription(
            String, 
            "/Unity_2_AI", 
            self.callback_from_Unity, 
            10
        )

        self.publisher_AINode_2_unity_thu_ROSbridge = self.create_publisher(
            Float32MultiArray, 
            '/AI_2_Unity', 
            10
        )

        self.input_size = 22 # input dimension 15
        self.hidden_size = 64
        self.num_layers = 1
        self.fc_hidden_size = 128
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True).to(device)
        self.fc1 = nn.Linear(self.hidden_size, 4).to(device) 

        # self.token = []

        self.model_path = './Supervised_Model/best_model.pth'
        self.model_weights = torch.load(self.model_path, map_location=device)
        self.lstm.load_state_dict(self.model_weights)

        self.lstm.eval()
        

    def publish_predict_data_2_unity(self, data):
        self.data = Float32MultiArray()
        self.data.data = data
        self.publisher_AINode_2_unity_thu_ROSbridge.publish(self.data)
        print(f"predictrd action: {data}")

    def unity_data_collect(self, unityState):
        token = list()
        self.state_detect, token = self.state.get_wanted_features()
        if self.state_detect == 1:
            #  收到資料後將他丟入LSTM
            self.lstm.to(device).eval()
            with torch.inference_mode():
                test_input = [eval(token)]
                test_input_tensor = torch.tensor(test_input, dtype=torch.float32)
                test_input_tensor = test_input_tensor[:, :-4]
                test_input_tensor = test_input_tensor.unsqueeze(0).to(device)
                h0 = torch.zeros(self.num_layers, test_input_tensor.size(0), self.hidden_size).to(device)
                c0 = torch.zeros(self.num_layers, test_input_tensor.size(0), self.hidden_size).to(device)
                lstm_output, _ = self.lstm(test_input_tensor, (h0, c0))

                lstm_output_last = lstm_output[:, -1, :]
                predicted_output = self.fc1(lstm_output_last)

                probabilities = F.softmax(predicted_output, dim=-1)
                print("Softmax probabilities:", probabilities)

                predicted_class = torch.argmax(predicted_output)
                print("Predicted class (Argmax):", predicted_class.item())
                
                action = list()
                action.append(float(predicted_class))
                self.publish_predict_data_2_unity(action)
                
    def callback_from_Unity(self, msg):
        Unitystate = msg.data
        self.state.update(Unitystate)
        self.unity_data_collect(msg.data)


# def spin_pros(node):
#     exe = rclpy.executors.SingleThreadedExecutor()
#     exe.add_node(node)
#     exe.spin()
#     rclpy.shutdown()
#     sys.exit(0)

# def print_usage():
#     print("modes:")
#     print(" 0 -- supervised learning data colletion.")
#     print(" 1 -- supervised learning inference.")
#     print(" 2 -- rule-based control.")
#     print(" 3 -- reinforced learning inference.")

# def main(mode):
#     if mode == "1":
#         rclpy.init()
#         node = supervised_inference_node()
#         spin_pros(node) 

# if __name__ == '__main__':
#     print_usage()
#     mode = input("Enter mode: ")
#     main(mode)
