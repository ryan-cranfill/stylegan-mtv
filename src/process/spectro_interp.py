import librosa
import tempfile
import numpy as np
from tqdm import tqdm
from pathlib import Path

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

    def get_images(self, sound_data, sample_rate, spectrogram_params: dict, window_size=1, displacement_factor=0.1,
                   n_points=3, likes_file=None):
        spectrogram = self.sound_to_mel_spectrogram(sound_data, sample_rate, spectrogram_params)
        duration = sound_data.shape[0] / sample_rate
        checkpoints = self.make_checkpoints(duration, n_points, likes_file)

        beginning, end = checkpoints[0], checkpoints[1]
        checkpoint_idx = 0

        images = {}
        for i, frame in tqdm(enumerate(spectrogram), total=len(spectrogram)):
            spectrogram_vec = self.get_spectrogram_vec(spectrogram, i, window_size, displacement_factor)

            timestamp = i / self.fps
            if timestamp > end[0]:
                checkpoint_idx += 1
                beginning, end = checkpoints[checkpoint_idx], checkpoints[checkpoint_idx + 1]

            interp_vec = self.interp_between_checkpoints(timestamp, beginning, end)

            latent_vec = spectrogram_vec + interp_vec

            images[i] = self.model.run_image(latent_vec, as_bytes=False)

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
