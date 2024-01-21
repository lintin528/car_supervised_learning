import sys
import rclpy
from supervised_collect_data import supervised_collect_data_node
from supervised_inference import supervised_inference_node


def spin_pros(node):
    exe = rclpy.executors.SingleThreadedExecutor()
    exe.add_node(node)
    exe.spin()
    rclpy.shutdown()
    sys.exit(0)

def print_usage():
    print("modes:")
    print(" 1 -- supervised learning data colletion.")
    print(" 2 -- supervised learning inference.")
    print(" 3 -- rule-based control.")
    print(" 4 -- reinforced learning inference.")

def main(mode):
    rclpy.init()
    if mode == "1":
        node = supervised_collect_data_node() 
    elif mode == "2":
        node = supervised_inference_node()
    else:
        print("please type the mode nums listed upside.")
    spin_pros(node)  

if __name__ == '__main__':
    print_usage()
    mode = input("Enter mode: ")
    main(mode)