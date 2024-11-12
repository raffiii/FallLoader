import torch
from PIL import Image
import numpy as np


def load_video_as_tensor(image_list, use_numpy=False):
    frames = []
    for image_path in image_list:
        if use_numpy:
            frame = read_bin_file_rgb(
                image_path
            )  # Load frame as numpy array using read_bin_file_rgb function
        else:
            frame = Image.open(image_path)  # Read image using PIL
        frames.append(frame)

    video_tensor = torch.stack(frames)  # Stack frames to create video tensor
    return video_tensor


def read_bin_file_rgb(file_path):
    """
    From cvhci-fall/scripts/datasets/OCCU_EDF/create_rgb_video.py
    """
    with open(file_path, "rb") as f:
        data = np.fromfile(f, dtype=np.uint16)

    ind = (data[1] << 16).astype(np.uint32) | data[0].astype(np.uint32)
    rows = data[3]
    cols = data[2]

    img_data = data[4:]
    img_data = img_data.view(np.uint8)
    img = np.reshape(img_data, (3, rows, cols))
    img = img.transpose((2, 1, 0))
    return ind, rows, cols, img


# Example usage
def main():
    image_list = [
        f"/home/rflbr/projects/Studium/MA/test_data/EDF/EDF/peter/peter/view1/rgb/00000{i}.bin"
        for i in range(1, 10)
    ]
    video_tensor = load_video_as_tensor(image_list)
    print(video_tensor.shape)  # Output: torch.Size([3, 3, H, W])


if __name__ == "__main__":
    main()
