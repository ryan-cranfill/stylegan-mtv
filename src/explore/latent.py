import cv2
import pickle
import numpy as np
from pathlib import Path

from src.model import StyleGANModel
from src.utils import find_stylegan_models, warn
from src.settings import LIKES_FILE, DISLIKES_FILE, LATENT_DIR

AVAILABLE_STYLEGAN_MODELS = find_stylegan_models()


def make_likes_and_dislikes_paths(prefix=None):
    if prefix:
        likes_file = LATENT_DIR / f'{prefix}_likes.pkl'
    else:
        likes_file = LIKES_FILE

    if prefix:
        dislikes_file = LATENT_DIR / f'{prefix}_dislikes.pkl'
    else:
        dislikes_file = DISLIKES_FILE

    return likes_file, dislikes_file


def load_likes_and_dislikes(likes_file, dislikes_file):
    if not likes_file.exists():
        likes = []
        warn('no likes file found, creating a new one')
    else:
        with open(likes_file, 'rb') as f:
            likes = pickle.load(f)

    if not dislikes_file.exists():
        dislikes = []
        warn('no dislikes file found, creating a new one')
    else:
        with open(dislikes_file, 'rb') as f:
            dislikes = pickle.load(f)

    return likes, dislikes


def save_likes_and_dislikes(likes, dislikes, likes_file=LIKES_FILE, dislikes_file=DISLIKES_FILE):
    with open(likes_file, 'wb') as f:
        pickle.dump(likes, f)

    with open(dislikes_file, 'wb') as f:
        pickle.dump(dislikes, f)


class LatentSpaceExplorer:
    def __init__(self, model_name='wikiart-sg2', random_seed=False, save=False, prefix=None):
        if model_name in AVAILABLE_STYLEGAN_MODELS:
            self.model_name = model_name
        else:
            fallback_model = list(AVAILABLE_STYLEGAN_MODELS.keys())[0]
            warn(f'Model {model_name} not available, falling back to {fallback_model}')
            self.model_name = fallback_model

        self.model = StyleGANModel(AVAILABLE_STYLEGAN_MODELS[self.model_name], random_seed=random_seed,
                                   reduced_memory=False)
        self.save = save

        self.likes_file, self.dislikes_file = make_likes_and_dislikes_paths(prefix)

        self.likes_dir = LATENT_DIR / self.likes_file.stem
        self.likes_dir.mkdir(exist_ok=True)

        self.likes, self.dislikes = load_likes_and_dislikes(self.likes_file, self.dislikes_file)

    def get_random_points(self, n=1):
        return np.random.randn(n, self.model.input_shape)

    def generate_and_show_image(self):
        latent_vec = self.get_random_points()
        img = self.model.run_image(latent_vec, as_bytes=False)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow('[ for like, ] for dislike', img)
        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            return False
        else:
            if key == ord('['):
                self.like(latent_vec, img)
            elif key == ord(']'):
                self.dislikes.append(latent_vec)
            return key

    def like(self, latent_vec, img):
        self.likes.append(latent_vec)
        if self.save:
            filename = f'img_{len(self.likes) - 1}.jpg'
            fp = self.likes_dir / filename
            cv2.imwrite(str(fp), img)

    def explore(self):
        return_code = True
        while return_code:
            return_code = self.generate_and_show_image()

        save_likes_and_dislikes(self.likes, self.dislikes, self.likes_file, self.dislikes_file)
