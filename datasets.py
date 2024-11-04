import os
import torch
from torch.utils.data import Dataset
import csv
from typing import List
import os.path
from dataclasses import dataclass
from videoutils import load_video_segment


class FallDataset(Dataset):
    def __init__(self, query_mode="action"):
        self.samples: List[FallSampleData] = []
        self.query = lambda x: x
        self.set_query_mode(query_mode)

    def __len__(self):
        return len(self.samples)

    def set_query_mode(self, mode="action"):
        def action(sample: FallSample):
            return sample.frames, sample.activity

        def domain(sample: FallSample):
            return sample.frames, sample.dataset

        def time(sample: FallSample):
            return sample.frames, sample.fall_frame

        def default(sample: FallSample):
            return sample

        if mode.lower() in ["fall", "action", "activity", "label"]:
            self.query = action
        elif mode.lower() in ["dataset", "domain"]:
            self.query = domain
        elif mode.lower() in [
            "fall_start",
            "start_fall",
            "start",
            "time",
            "starttime",
            "start_time",
        ]:
            self.query = time
        elif mode.lower() in ["full", "all"]:
            self.query = default
        else:
            self.query = default

    def __getitem__(self, idx):
        loaded_sample: FallSample = self.samples[idx].load(target_fps=30)
        return self.query(loaded_sample)


@dataclass
class FallSampleData:
    idx: int
    path: str
    dataset: int
    activity: int
    fall_frame: int | None
    recovery_frame: int | None
    # Begin and end frame of the fall/ADL subsequence of the full video
    begin_time_ms: int | None
    end_time_ms: int | None
    train_split: bool

    def load(self, target_fps):
        np_video = load_video_segment(
            self.path,
            self.begin_time_ms / 1000.0,
            self.end_time_ms / 1000.0,
            target_fps,
        )
        video_tensor = torch.from_numpy(np_video)
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


class SuperSet(FallDataset):
    def __init__(self, base_path, metadata_file, samples=None, query_mode="action"):
        super().__init__(query_mode=query_mode)
        self.base_path, self.metadata_file = base_path, metadata_file
        if samples is not None:
            self.samples = samples
            return
        rows = []
        with open(os.path.join(base_path, metadata_file)) as f:
            csvreader = csv.reader(f)
            rows = [r for r in csvreader]
        if len(rows) == 0:
            raise Exception("No metadata available")
        self.samples = [
            FallSampleData(
                int(idx),
                os.path.join(base_path, path),
                int(dataset_id),
                int(label),
                None,
                None,
                float(start),
                float(end),
                train_split,
            )
            for idx, path, dataset_id, subject, start, end, label, cls, subjects, train_split in rows[
                1:
            ]
        ]

    def filter(self, filter):
        filtered_samples = [
            sample for i, sample in enumerate(self.samples) if filter(sample, i)
        ]
        return SuperSet(None, None, samples=filtered_samples)


def main():
    base_path = "/home/rflbr/projects/Studium/MA/test_data"

    def path_exist_filter(sample: FallSampleData, idx):
        return os.path.exists(sample.path)

    def take_first_n(n):
        def filter(sample, idx):
            return idx < n

        return filter

    dataset = SuperSet(base_path, "relative_superset.csv").filter(
        path_exist_filter
    )  # .filter(take_first_n(4))
    print(len(dataset.samples))

    def collate(batch):
        return tuple(zip(*batch))
        # return (torch.stack([video for video, _ in batch]),torch.stack([label for _, label in batch]))

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=1, collate_fn=collate
    )
    for data in data_loader:
        videos, labels = data
        print(len(videos), len(labels))
        print(videos[0].shape, labels[0])


main()
