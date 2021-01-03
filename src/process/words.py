import cv2
import math
import pickle
import librosa
import tempfile
import numpy as np
from tqdm import tqdm
from pathlib import Path

from src.settings import LATENT_DIR
from .base import BaseOfflineProcessor
from src.utils import warn, filter_latent_reps_by_images
from src.model.word_vec import WordVectorizer


def ramp_to_edges(timestamp: float, beginning: float, end: float, interp_time: float = .9):
    # interp time - what pct should i spend doing interpolation between points
    assert 0 <= interp_time <= 1
    static_pct = 1 - interp_time
    static_h = static_pct / 2
    # interp_start, interp_end = (1 + static_h) * beginning, (1 - static_h) * end

    timestamp_centered = timestamp - beginning
    seg_duration = (end - beginning)
    interp_duration = seg_duration * interp_time
    interp_start, interp_end = static_h * seg_duration, seg_duration - (static_h * seg_duration)
    if timestamp_centered < interp_start:
        return 0
    elif timestamp_centered > interp_end:
        return 1
    else:
        return (timestamp_centered - interp_start) / interp_duration


class WordsOfflineProcessor(BaseOfflineProcessor):
    def __init__(self, model_name='cats', fps=5, random_seed=False, frame_chunk_size=500):
        super().__init__(model_name, fps, random_seed, frame_chunk_size)
        self.word_vectorizer = WordVectorizer(output_shape=self.model.input_shape)

    def make_checkpoints(self, text, duration):
        """
        Returns a list of tuples with timestamps of when to hit which point
        """
        points = self.word_vectorizer.get_word_vecs_from_text(text)
        n_points = len(points)

        checkpoints = []
        for i in range(n_points):
            timestamp = duration * (i / (n_points - 1))
            word, vec = points[i]
            checkpoints.append((timestamp, word, vec))

        return checkpoints

    def interp_between_checkpoints(self, timestamp: float, beginning: tuple, end: tuple, interp_time=0.):
        beginning_ts, beginning_word, beginning_vec = beginning
        end_ts, end_word, end_vec = end
        timestamp_centered = timestamp - beginning_ts
        if 1 > interp_time > 0:
            # ratio = ramp_to_edges(timestamp, beginning_ts, end_ts, k=25)
            ratio = ramp_to_edges(timestamp, beginning_ts, end_ts, interp_time=interp_time)
        else:
            ratio = timestamp_centered / (end_ts - beginning_ts)
        return (((1 - ratio) * beginning_vec) + (end_vec * ratio)).reshape(1, -1)

    # def get_images(self, sound_data, total_frames, duration, n_points, likes_file=None):
    def get_images(self, text:str, duration=30, overlay_word=True, interp_time=0.):
        images = {}

        checkpoints = self.make_checkpoints(text, duration)

        beginning, end = checkpoints[0], checkpoints[1]
        checkpoint_idx = 0

        total_frames = duration * self.fps
        for i in tqdm(range(total_frames)):
            timestamp = i / self.fps
            if timestamp > end[0]:
                checkpoint_idx += 1
                beginning, end = checkpoints[checkpoint_idx], checkpoints[checkpoint_idx + 1]

            latent_vec = self.interp_between_checkpoints(timestamp, beginning, end, interp_time)
            img = self.model.run_image(latent_vec, as_bytes=False)

            if overlay_word:
                border_height = int(self.model.output_shape[0] * .2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10, self.model.output_shape[0] + border_height - int(border_height * .333))
                bottomMiddle = (int(self.model.output_shape[0] * .25),
                                self.model.output_shape[0] + border_height - int(border_height * .333))
                # fontScale = 1
                fontScale = .5
                fontColor = (255, 255, 255)
                lineType = 1

                beginning_diff = abs(timestamp - beginning[0])
                end_diff = abs(timestamp - end[0])

                img = cv2.copyMakeBorder(img, 0, 20, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                cv2.putText(img,
                            # f'{beginning[1]} - {end[1]}',
                            beginning[1] if beginning_diff < end_diff else end[1],
                            # bottomLeftCornerOfText,
                            bottomMiddle,
                            font,
                            fontScale,
                            fontColor,
                            lineType)

            images[i] = img

            if i > 0 and i % self.frame_chunk_size == 0 and images:
                self.write_chunk_to_temp(images)
                del images
                images = {}

        self.write_chunk_to_temp(images)

        return images

    def process_file(self, output_path: str, text: str, duration=30, write=True, interp_time=0.):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        self.get_images(text, duration, interp_time=interp_time)

        return self.create_video(duration, None, output_path, write=write, sound=False)


if __name__ == '__main__':
    pass
