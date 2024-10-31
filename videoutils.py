import av
import itertools
import numpy as np


def resample_video_streaming(frames, original_fps, new_fps):
    """
    Resample the video frames from original FPS to new FPS
    source: https://github.com/pytorch/vision/issues/3016
    """
    step = float(original_fps) / new_fps

    if step.is_integer():
        return itertools.islice(frames, 0, None, int(step))
    else:

        def _gen():
            output_frames = 0

            for input_frames, frame in enumerate(frames):
                while output_frames < input_frames:
                    yield frame
                    output_frames += step

        return _gen()


def load_video_segment(input_path, start_time, end_time, target_fps):
    # Open the video file
    container = av.open(input_path)

    # Calculate frame skip interval based on the video's native framerate and target framerate
    native_fps = container.streams.video[0].average_rate
    frame_interval = int(native_fps / target_fps)

    # Initialize variables
    frames = []
    current_time = start_time

    # Seek to the starting point
    container.seek(int(start_time * av.time_base), stream=container.streams.video[0])

    # Load frames until reaching end_time
    for packet in container.demux(video=0):
        for frame in packet.decode():
            if frame.time >= end_time:
                return np.stack(frames) if frames else np.array([])

            # Capture frames at the specified frame interval to adjust to target FPS
            if frame.time >= current_time and (frame.index % frame_interval == 0):
                frames.append(frame.to_ndarray(format="rgb24"))
                current_time += 1.0 / target_fps

    return np.stack(frames) if frames else np.array([])


def extract_video_segment(video_path, start_time, end_time):
    """
    Extract a segment from a video using pyav and return as torch tensors
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    start_pts = int(start_time * stream.time_base.den / stream.time_base.num)
    end_pts = int(end_time * stream.time_base.den / stream.time_base.num)

    container.seek(start_pts, any_frame=False, backward=True, stream=stream)

    frames = []
    for frame in container.decode(stream):
        if frame.pts >= end_pts:
            break
        frames.append(torch.from_numpy(frame.to_rgb().to_ndarray()))

    container.close()

    return frames