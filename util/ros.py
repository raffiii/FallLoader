import rosbag
import cv2
import torch


def load_frames_from_rosbag(rosbag_file, topics):
    bag = rosbag.Bag(rosbag_file)
    bridge = cv2.bridge.CvBridge()
    frames = []

    for topic, msg, t in bag.read_messages(topics=topics):
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        frame_tensor = torch.from_numpy(frame)
        frames.append(frame_tensor)

    bag.close()
    return torch.stack(frames)


# Usage example
def main():
    rosbag_file = "/path/to/your/rosbag.bag"
    topic = "/camera_topic"
    frames_tensor = load_frames_from_rosbag(rosbag_file, topic)
    print(frames_tensor.shape)  # Output: torch.Size([N, H, W, C])


if __name__ == "__main__":
    main()
