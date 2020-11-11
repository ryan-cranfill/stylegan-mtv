import json
import math
import pickle
import librosa
import tempfile
import numpy as np
from tqdm import tqdm
from pathlib import Path

from src.settings import LATENT_DIR
from .base import BaseOfflineProcessor
from src.utils import warn


def ramp_to_edges(timestamp: float, beginning: float, end: float, k: float = 3):
    timestamp_centered = timestamp - beginning
    seg_duration = (end - beginning)
    ratio = timestamp_centered / seg_duration
    if ratio == 0:
        ratio = 0.0000001
    elif ratio == 1:
        ratio = 0.9999999
    return 1 / (1 + (ratio / (1 - ratio)) ** -k)


def checkpoints_from_config_file(config_file):
    print('reading configuration from', config_file)
    checkpoints = []

    with open(config_file) as f:
        config = json.load(f)

    for step in config['steps']:
        vec = np.load(str(LATENT_DIR / step['name']))
        checkpoints.append((step['time'], vec))
        print(step)

    return checkpoints


class InterpolationOfflineProcessor(BaseOfflineProcessor):
    def __init__(self, model_name='cats', fps=5, random_seed=False, frame_chunk_size=500):
        super().__init__(model_name, fps, random_seed, frame_chunk_size)

    def make_checkpoints(self, duration, n_points=3, likes_file=None, config_file=None, likes_dir=None):
        """
        Returns a list of tuples with timestamps of when to hit which random point
        """
        if config_file:
            return checkpoints_from_config_file(config_file)

        rng = np.random.default_rng()

        if likes_file:
            with open(likes_file, 'rb') as f:
                likes = pickle.load(f)
            points = rng.choice(likes, n_points)
        elif likes_dir:
            likes_path = Path(likes_dir)
            vecs = [np.load(str(p)) for p in likes_path.glob('*.npy')]
            points = rng.choice(vecs, n_points)
        else:
            points = self.get_random_points(n_points)

        checkpoints = []
        for i in range(n_points):
            timestamp = duration * (i / (n_points - 1))
            checkpoints.append((timestamp, points[i]))

        return checkpoints

    def interp_between_checkpoints(self, timestamp: float, beginning: tuple, end: tuple, is_dlatent=False, ramp=False):
        beginning_ts, beginning_vec = beginning
        end_ts, end_vec = end
        timestamp_centered = timestamp - beginning_ts
        if ramp:
            ratio = ramp_to_edges(timestamp, beginning_ts, end_ts)
        else:
            ratio = timestamp_centered / (end_ts - beginning_ts)
        if is_dlatent:
            return ((1 - ratio) * beginning_vec) + (end_vec * ratio)
        return (((1 - ratio) * beginning_vec) + (end_vec * ratio)).reshape(1, -1)

    def get_images(self, sound_data, total_frames, duration, n_points, likes_file=None):
        images = {}

        checkpoints = self.make_checkpoints(duration, n_points, likes_file)

        beginning, end = checkpoints[0], checkpoints[1]
        checkpoint_idx = 0

        chunks = np.array_split(sound_data, total_frames)
        for i, frame in tqdm(enumerate(chunks), total=total_frames):
            timestamp = i / self.fps
            if timestamp > end[0]:
                checkpoint_idx += 1
                beginning, end = checkpoints[checkpoint_idx], checkpoints[checkpoint_idx + 1]

            latent_vec = self.interp_between_checkpoints(timestamp, beginning, end)

            images[i] = self.model.run_image(latent_vec, as_bytes=False)

            if i > 0 and i % self.frame_chunk_size == 0 and images:
                self.write_chunk_to_temp(images)
                del images
                images = {}

        self.write_chunk_to_temp(images)

        return images

    def process_file(self, input_path: str, output_path: str, start=0, duration=None, sr=None,
                     write=True, n_points=3, likes_file=None):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        if n_points < 2:
            warn(f'WARN: n_points must be >=2, setting to 2 (received {n_points})')
            n_points = 2

        sound_data, sample_rate = librosa.load(input_path, sr=sr, offset=start, duration=duration)
        duration = sound_data.shape[0] / sample_rate
        total_frames = math.ceil(duration * self.fps)
        print('********* My duration:', duration, n_points, likes_file)

        self.get_images(sound_data, total_frames, duration, n_points, likes_file)

        return self.create_video(duration, input_path, output_path, write=write, start=start)


if __name__ == '__main__':
    pass
