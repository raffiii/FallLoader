import av
import av.container
import av.frame
from fractions import Fraction
import numpy as np
import cv2


def packet_iterator(
    start_time: float | Fraction | None, end_time: float | Fraction | None
):

    def iterate_frames(container: av.container.InputContainer):
        # Iterate over packets in the container
        previous_packet = None
        for packet in container.demux(video=0):
            if packet.pts < start_time * packet.time_base:
                previous_packet = packet
                continue
            if previous_packet is not None:
                for frame in previous_packet.decode():
                    if end_time is not None and frame.time >= end_time:
                        return
                    if start_time is None or frame.time >= start_time:
                        yield frame.to_ndarray(format="rgb24")
                previous_packet = None
            for frame in packet.decode():
                if end_time is not None and frame.time >= end_time:
                    return
                # Capture frames at the specified frame interval to adjust to target FPS
                if start_time is None or frame.time >= start_time:
                    yield frame.to_ndarray(format="rgb24")

    return iterate_frames


def frame_iterator(start_time, end_time):
    def iterate_frames(container):
        for frame in container.decode(video=0):
            if end_time is not None and frame.time >= end_time:
                return
            if start_time is None or frame.time >= start_time:
                yield frame.to_ndarray(format="rgb24")

    return iterate_frames


def resample_skip_iterator(target_rate):
    target_interval = 1 / Fraction(target_rate)

    def resample(frames, native_frame_interval):
        if native_frame_interval == target_interval:
            yield from frames
            return
        out_time = Fraction(0)
        source_time = Fraction(0)
        for frame in frames:
            if out_time <= source_time:
                while out_time < source_time + native_frame_interval:
                    yield frame
                    out_time += target_interval
            source_time += native_frame_interval

    return resample


def load_video(input_path, frame_extractor, frame_processor):
    container = av.open(input_path)
    native_fps = container.streams.video[0].average_rate
    native_frame_interval = 1 / native_fps
    source_frames = frame_extractor(container)
    target_frames = frame_processor(source_frames, native_frame_interval)
    frames = [frame for frame in target_frames]
    container.close()
    return np.stack(frames) if frames else np.array([])


def load_video_segment(input_path, start_time, end_time, target_fps):
    frame_extractor = frame_iterator(start_time, end_time)
    frame_processor = resample_skip_iterator(target_fps)
    return load_video(input_path, frame_extractor, frame_processor)


def resize_frames(width, height):
    def resize(frames):
        resized_frames = []
        for frame in frames:
            resized_frame = cv2.resize(frame, (width, height))
            resized_frames.append(resized_frame)
        return resized_frames

    return resize


def sample_frames(num_frames, subsample_rate):

    def sample(frames):
        indices = sample_frame_indices(num_frames, subsample_rate, len(frames))
        return np.stack([frames[i] for i in indices])

    return sample


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    """
    Sample frame indices for a video clip. The indices are sampled to form a clip of `clip_len` frames sampling
    one out of every `frame_sample_rate` frames from a segment of length `seg_len`.
    (inspired by https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/VideoMAE/Quick_inference_with_VideoMAE.ipynb#scrollTo=6LBAV-7u3cI6)

    Args:
        clip_len (int): Number of frames in the clip.
        frame_sample_rate (int): Sampling rate for frames in the clip.
        seg_len (int): Length of the segment to sample frames from.
    """
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, max(converted_len + 1, seg_len))
    str_idx = end_idx - converted_len
    index = (
        np.linspace(str_idx, end_idx, num=clip_len, endpoint=False)
        .clip(str_idx, seg_len)
        .astype(np.int64)
    )
    return index
