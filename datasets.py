import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
from typing import List
import os.path
from dataclasses import dataclass


class _FallDataset(Dataset):

    def __init__(self):
        self.image_paths = []
        self.labels = []

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]

        return image, label


@dataclass
class FallSample:
    frames: Image  # TODO check type
    dataset: int
    activity: int
    fall_frame: int | None
    recovery_frame: int | None

    dataset_map = [
        "MultiCameraFall",
        "OCCU",
        "EDF",
        "OOPS",
        "Le2i",
        "CAUCAFall",
        "UR_Fall",
        "MUVIM",
        "FDPS_v2",
    ]
    activities = ["FallForward", "FallLateral", "FallBackward", "ADL"]


class MultiCameraView(_FallDataset):
    def __init__(self, root_dir: str, format="mp4"):
        """
        Args:
            root_dir: Root directory of MCV (unmodified, archive extracted)
        """
        super().__init__()
        self.root_dir = root_dir
        self.image_paths = [
            os.path.join(
                root_dir,
                "Multiple_Cameras_Fall/dataset",
                f"chute{fall_n+1}/cam{cam_n+1}.{format}",
            )
            for fall_n in range(24)  # 24 scenarios
            for cam_n in range(8)  # 8 cams
        ]
        self.labels = ["fall"] * (22 * 8) + ["adl"] * (2 * 8)


class Oops(_FallDataset):
    def __init__(self, root_dir: str, scenes: int | List[str] | None = None):
        super().__init__()
        self.root_dir = root_dir

        annotations_dir = os.path.join(self.root_dir, "OOPS/oops_dataset/annotations")
        videos_dir = os.path.join(self.root_dir, "OOPS/oops_dataset/oops_video")

        train_names, val_names = [], []
        # Get Filtered files
        with open(
            os.path.join(
                annotations_dir, "OOPS/oops_dataset/annotations/train_filtered.txt"
            )
        ) as f:
            train_names = f.readlines()
        with open(
            os.path.join(
                annotations_dir, "OOPS/oops_dataset/annotations/val_filtered.txt"
            )
        ) as f:
            val_names = f.readlines()
        files = [(name, "train") for name in train_names] + [
            (name, "val") for name in val_names
        ]
        files = [(name.replace("\n", ""), split) for name, split in files]

        # Get labels
        with open(os.path.join(annotations_dir, "transition_times.json")) as f:
            labels = json.load(f)
        for filename, split in files:
            if filename in labels:
                self.image_paths.append(os.path.join(videos_dir, split, filename))
                # select Median if at least 2 annotators found unintentional action
                self.labels.append(
                    None
                    if labels[filename]["n_notfound"] > 1
                    else sorted(labels[filename]["t"])[1]
                )
