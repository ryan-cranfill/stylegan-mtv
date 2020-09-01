import librosa
import tempfile
import numpy as np
from tqdm import tqdm
from pathlib import Path

from src.model.stylegan import BATCH_SIZE
from .spectro import SpectrogramOfflineProcessor
from .interpolation import InterpolationOfflineProcessor


DEFAULT_SPECTRO_PARAMS = dict(
    # n_fft=2048,
    n_fft=8192,
    hop_length=512,
    n_mels=128,
    fmin=20,
    fmax=20000
)


class SpectrogramInterpolationOfflineProcessor(SpectrogramOfflineProcessor, InterpolationOfflineProcessor):
    def __init__(self, model_name='cats', fps=5, random_seed=False, frame_chunk_size=500):
        super().__init__(model_name, fps, random_seed, frame_chunk_size)


    # def interp_between_checkpoints(self, timestamp: float, beginning: tuple, end: tuple, magnitude=1.):
    #     beginning_ts, beginning_vec = beginning
    #     end_ts, end_vec = end
    #     timestamp_centered = timestamp - beginning_ts
    #     ratio = timestamp_centered / (end_ts - beginning_ts)
    #     return (((1 - ratio) * beginning_vec) + (end_vec * ratio * magnitude)).reshape(1, -1)

    def get_images(self, sound_data, sample_rate, spectrogram_params: dict, window_size=1, displacement_factor=0.1,
                   # n_points=3, likes_file=None, use_mag=False):
                   n_points=3, likes_file=None):
        spectrogram = self.sound_to_mel_spectrogram(sound_data, sample_rate, spectrogram_params)
        duration = sound_data.shape[0] / sample_rate
        checkpoints = self.make_checkpoints(duration, n_points, likes_file)

        beginning, end = checkpoints[0], checkpoints[1]
        checkpoint_idx = 0

        # todo: investigate batch image synthesis
        images = {}
        batch, nums = [], []
        for i, frame in tqdm(enumerate(spectrogram), total=len(spectrogram)):
            spectrogram_vec = self.get_spectrogram_vec(spectrogram, i, window_size, displacement_factor)

            timestamp = i / self.fps
            if timestamp > end[0]:
                checkpoint_idx += 1
                beginning, end = checkpoints[checkpoint_idx], checkpoints[checkpoint_idx + 1]

            magnitude = np.linalg.norm(frame) / 2000
            if magnitude < 0.1:
                magnitude = 0.1

            interp_vec = self.interp_between_checkpoints(timestamp, beginning, end)
            # if use_mag:
            #     magnitude = np.linalg.norm(frame) / 2000
            #     # print(magnitude)
            #     interp_vec = self.interp_between_checkpoints(timestamp, beginning, end, magnitude)
            # else:
            #     interp_vec = self.interp_between_checkpoints(timestamp, beginning, end)

            # latent_vec = spectrogram_vec + (interp_vec * magnitude)
            latent_vec = spectrogram_vec * interp_vec
            # latent_vec = spectrogram_vec * interp_vec * magnitude

            images[i] = self.model.run_image(latent_vec, as_bytes=False)

            # latent_vec = spectrogram_vec * interp_vec * magnitude
            # batch.append(latent_vec)
            # nums.append(i)
            #
            # if len(batch) == BATCH_SIZE:
            #     latents = np.vstack(batch)
            #     # print(latents.shape, latent_vec.shape)
            #     imgs = self.model.run_images(latents)
            #     for idx, img in zip(nums, imgs):
            #         images[idx] = img
            #         batch, nums = [], []

            if i > 0 and i % self.frame_chunk_size == 0 and images:
                self.write_chunk_to_temp(images)
                del images
                images = {}

        self.write_chunk_to_temp(images)

        return images

    def process_file(self, input_path: str, output_path: str, start=0, duration=None, sr=None,
                     write=True, window_size=1, displacement_factor=0.1, spectrogram_params=None, n_points=3,
                     likes_file=None):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        if spectrogram_params is None:
            spectrogram_params = DEFAULT_SPECTRO_PARAMS
        sound_data, sample_rate = librosa.load(input_path, sr=sr, offset=start, duration=duration)

        self.get_images(sound_data, sample_rate, spectrogram_params, window_size, displacement_factor, n_points, likes_file)

        duration = sound_data.shape[0] / sample_rate
        return self.create_video(duration, input_path, output_path, write=write, start=start)


if __name__ == '__main__':
    pass
