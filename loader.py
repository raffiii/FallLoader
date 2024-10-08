import skvideo.io
import os.path
import os
from typing import List, Union, Tuple

class FallLoader:
    def __init__(self, datasets_root, falls, cams):
        self.datasets_root=datasets_root
        self.falls = falls
        self.cams = cams

    def identifiers(self, cams=None,falls=None) -> List[Tuple[str,Tuple[Union[None,int],Union[None,int]]]]:
        # Should return [(filename , (start_frame, end_frame))]
        # start_frame, end_frame should be None for full video
        raise NotImplementedError

    def iterate_all(self):
        for file, (frame_start, frame_end) in self.identifiers():
            videodata = skvideo.io.vread(file)
            yield videodata[frame_start:frame_end]
    
class MultiViewLoader(FallLoader):

    def __init__(self, datasets_root):
        super.__init__(self, datasets_root, 24,8)

    def identifiers(self, cams=None, falls=None, format="mp4"):
        falls = self.falls if falls is None else falls
        falls = range(1,falls+1) if type(falls) == int else falls
        cams = self.cams if cams is None else cams
        cams = range(1,cams+1) if type(cams) == int else cams
        files = [f"Multiple_Cameras_Fall/dataset/chute{fall_n}/cam{cam_n}.{format}" 
                    for fall_n in falls 
                    for cam_n in cams
                    ]
        # select all frames in each file
        files = [(f,(None,None)) for f in files]
