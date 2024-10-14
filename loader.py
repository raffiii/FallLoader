import skvideo.io
import os.path
import os
import json
from typing import List, Union, Tuple

class Scene:
    def __init__(self, filename: str, frames: Tuple[int | None, int | None] = (None,None), fall_frame: int | None = None):
        self.filename, frames, fall_frame = filename, frames, fall_frame

class FallLoader:
    def __init__(self, datasets_root, scenes, cams):
        self.datasets_root=datasets_root
        self.scenes = scenes
        self.cams = cams

    def identifiers(self, cams=None,scenes=None) -> List[Scene]:
        raise NotImplementedError

    def iterate_all(self):
        for file, (frame_start, frame_end) in self.identifiers():
            videodata = skvideo.io.vread(file)
            yield videodata[frame_start:frame_end]
    
class MultiViewLoader(FallLoader):

    def __init__(self, datasets_root):
        super.__init__(self, datasets_root, 24,8)

    def identifiers(self, cams=None, scenes=None, format="mp4"):
        scenes = self.scenes if scenes is None else scenes
        scenes = range(1,scenes+1) if type(scenes) == int else scenes
        cams = self.cams if cams is None else cams
        cams = range(1,cams+1) if type(cams) == int else cams
        files = [f"Multiple_Cameras_Fall/dataset/chute{fall_n}/cam{cam_n}.{format}" 
                    for fall_n in scenes 
                    for cam_n in cams
                    ]
        # select all frames in each file
        return [Scene(f) for f in files]


class OopsLoader(FallLoader):
    def __init__(self, datasets_root):
        super.__init__(self, datasets_root, 0,1)
        self.annotations_dir = f"{self.datasets_root}/OOPS/oops_dataset/annotations"
        self.videos_dir = f"{self.datasets_root}/OOPS/oops_dataset/oops_video"
        with open(f"{self.annotations_dir}/filtered_vids.txt") as f:
            self.scenes = len(f.readlines())

    def identifiers(self, cams=None, scenes=None) -> List[Tuple[str | Tuple[int | None, int | None]]]:
        scenes = self.scenes if scenes is None else scenes
        scenes = range(1,scenes+1) if type(scenes) == int else scenes
        train_names, val_names = [], []
        # Get Filtered files
        with open(f"{self.annotations_dir}/OOPS/oops_dataset/annotations/train_filtered.txt") as f:
            train_names = f.readlines()
        with open(f"{self.annotations_dir}/OOPS/oops_dataset/annotations/val_filtered.txt") as f:
            val_names = f.readlines()
        files = [(name, "train") for name in train_names] + [(name, "val") for name in val_names]

        # Get labels
        with open(f"{self.annotations_dir}/transition_times.json") as f:
            labels = json.load(f)
        train_names = [f"{self.videos_dir}/train/{name.replace("\n","")}" for name in train_names]
        val_names = [f"{self.videos_dir}/val/{name.replace("\n","")}" for name in val_names]
        labels = {}
        return [
            Scene(
                f"{self.videos_dir}/{split}/{file}", 
                fall_frame=None if labels[file]["n_notfound"] > 1 else sorted(labels[file]["t"])[1]
            ) 
            for file, split in files 
            if file in labels
            ]