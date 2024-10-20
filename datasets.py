import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import json, csv
from typing import List
import os.path
import torchvision.io as io
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


class FallDataset(Dataset):
    def __init__(self):
        self.samples = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx].load()


@dataclass
class FallSampleData:
    idx: int
    path: str
    dataset: int
    activity: int
    fall_frame: int | None
    recovery_frame: int | None
    # Begin and end frame of the fall/ADL subsequence of the full video
    begin_freme: int | None
    end_frame: int | None
    train_split: bool

    def load(self):
        video_tensor, _, _ = io.read_video(
            self.path,
            start_pts=self.start_frame,
            end_pts=self.end_frame,
            pts_unit="frames",
        )
        return FallSample(
            self.idx,
            video_tensor,
            self.dataset,
            self.activity,
            self.fall_frame,
            self.recovery_frame,
            self.train_split,
        )


@dataclass
class FallSample:
    idx: int
    frames: torch.Tensor  # TODO check type
    dataset: int
    activity: int
    fall_frame: int | None
    recovery_frame: int | None
    train_split: bool

    dataset_map = {
        1: "UR_Fall",
        2: "CAUCAFall",
        5: "EDF",
        7: "MultiCameraFall",
        8: "OCCU",
        9: "OOPS",
        # "Le2i",
        # "MUVIM",
        # "FDPS_v2",
    }
    activities = {
        0: "No Fall",
        2: "FallForward",
        2: "FallBackward",
        3: "FallLateral",
    }

    @classmethod
    def load_sample(sample: FallSampleData):
        video_tensor, _, _ = io.read_video(
            sample.path,
            start_pts=sample.start_frame,
            end_pts=sample.end_frame,
            pts_unit="frames",
        )
        return FallSample(
            video_tensor,
            sample.dataset,
            sample.activity,
            sample.fall_frame,
            sample.recovery_frame,
        )


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


class OCCU(_FallDataset):
    def __init__(self):
        super().__init__()


class SuperSet(FallDataset):
    def __init__(self, base_path, metadata_file):
        super().__init__()
        rows = []
        with open(os.path.join(base_path, metadata_file)) as f:
            csvreader = csv.reader(f)
            rows = [r for r in csvreader]
        if len(rows) == 0:
            raise Exception("No metadata available")
        self.samples = [
            FallSampleData(
                idx, path, dataset_id, label, None, None, start, end, train_split
            )
            for idx, path, dataset_id, _subject, start, end, label, _cls, _subjects, train_split in rows
        ]
