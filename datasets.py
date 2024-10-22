import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import json, csv
from typing import List
import os.path
import torchvision.io as io
from dataclasses import dataclass


class FallDataset(Dataset):
    def __init__(self):
        self.samples = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx, mode="action"):
        loaded_sample: FallSample = self.samples[idx].load()
        if mode.lower() in ["fall", "action", "activity", "label"]:
            return loaded_sample.frames, loaded_sample.activity
        if mode.lower() in ["dataset", "domain"]:
            return loaded_sample.frames, loaded_sample.dataset
        if mode.lower() in [
            "fall_start",
            "start_fall",
            "start",
            "time",
            "starttime",
            "start_time",
        ]:
            return loaded_sample.frames, loaded_sample.fall_frame
        if mode.lower() in ["full", "all"]:
            return loaded_sample
        return loaded_sample


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
