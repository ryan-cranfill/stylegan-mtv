import sys
import numpy as np
from pathlib import Path

from .settings import MODEL_DIR, LATENT_DIR, LIB_DIR

IMAGE_TYPES = ['.png', '.jpg', '.jpeg']


def warn(*values):
    print(*values, file=sys.stderr)


def add_src_to_sys_path():
    print('********', LIB_DIR)
    sys.path.append(str(LIB_DIR))
    print(sys.path)


def make_model_map(model_files):
    model_map = {}
    for p in model_files:
        split = p.stem.split('_')
        if len(split[0]) > 20:  # dumb hack to drop out the url MD5 hash:
            print('splitting this filepath:', p.stem)
            name = '_'.join(split[1:])
        else:
            name = p.stem
        model_map[name] = str(p)
    return model_map


def find_stylegan_models():
    model_files = MODEL_DIR.glob('*.pkl')
    return make_model_map(model_files)


def find_latent_representations():
    reps = LATENT_DIR.glob('*.npy')
    return {i.stem: str(i) for i in reps}


def load_latent_reps():
    latent_reps_paths = find_latent_representations()
    reps = {name: np.expand_dims(np.load(path), axis=0)  # is this expand_dims necessary?
            for name, path in latent_reps_paths.items()}
    return reps


def filter_stems_for_path(path: (Path, str), types: (list, tuple)):
    return {p.resolve().stem for p in Path(path).glob("*.*") if p.suffix in types}


def get_image_stems(path):
    return filter_stems_for_path(path, IMAGE_TYPES)


def get_numpy_stems(path):
    return filter_stems_for_path(path, ['.npy'])


def filter_latent_reps_by_images(latent_reps_dir, images_dir):
    image_stems = get_image_stems(images_dir)
    keeps = [i for i in list(Path(latent_reps_dir).glob('*.npy')) if i.stem in image_stems]
    return keeps


if __name__ == '__main__':
    warn('this is a test')
    warn('this is another', 'test', 'haha cool')
    print(find_stylegan_models())
