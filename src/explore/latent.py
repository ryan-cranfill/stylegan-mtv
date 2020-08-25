import cv2
import pickle
import numpy as np

from src.model import StyleGANModel
from src.utils import find_stylegan_models, warn
from src.settings import LIKES_FILE, DISLIKES_FILE

AVAILABLE_STYLEGAN_MODELS = find_stylegan_models()


def load_likes_and_dislikes():
    if not LIKES_FILE.exists():
        likes = []
        warn('no likes file found, creating a new one')
    else:
        with open(LIKES_FILE, 'rb') as f:
            likes = pickle.load(f)

    if not DISLIKES_FILE.exists():
        dislikes = []
        warn('no dislikes file found, creating a new one')
    else:
        with open(DISLIKES_FILE, 'rb') as f:
            dislikes = pickle.load(f)

    return likes, dislikes


def save_likes_and_dislikes(likes, dislikes):
    with open(LIKES_FILE, 'wb') as f:
        pickle.dump(likes, f)

    with open(DISLIKES_FILE, 'wb') as f:
        pickle.dump(dislikes, f)


class LatentSpaceExplorer:
    def __init__(self, model_name='wikiart', random_seed=False):
        if model_name in AVAILABLE_STYLEGAN_MODELS:
            self.model_name = model_name
        else:
            fallback_model = list(AVAILABLE_STYLEGAN_MODELS.keys())[0]
            warn(f'Model {model_name} not available, falling back to {fallback_model}')
            self.model_name = fallback_model

        self.model = StyleGANModel(AVAILABLE_STYLEGAN_MODELS[self.model_name], random_seed=random_seed,
                                   reduced_memory=False)

        self.likes, self.dislikes = load_likes_and_dislikes()

    def get_random_points(self, n=1):
        return np.random.randn(n, self.model.input_shape)

    def generate_and_show_image(self):
        latent_vec = self.get_random_points()
        img = self.model.run_image(latent_vec, as_bytes=False)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow('[ for like, ] for dislike', img)
        key = cv2.waitKey(0)
        if key == 27:
            return False
        else:
            if key == ord('['):
                self.likes.append(latent_vec)
            elif key == ord(']'):
                self.dislikes.append(latent_vec)
            return key

    def explore(self):
        return_code = True
        while return_code:
            return_code = self.generate_and_show_image()

        save_likes_and_dislikes(self.likes, self.dislikes)
