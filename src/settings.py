from pathlib import Path

SETTINGS_FILE = Path(__file__).absolute()
LIB_DIR = SETTINGS_FILE.parent
ROOT_DIR = LIB_DIR.parent
OUTPUT_DIR = ROOT_DIR / 'output'
INPUT_DIR = ROOT_DIR / 'input'
MODEL_DIR = ROOT_DIR / 'models'
LATENT_DIR = ROOT_DIR / 'latent_representations'
TEST_SOUND_SHORT = LIB_DIR / 'sound/test.wav'
DNNLIB_PATH = LIB_DIR / 'dnnlib'
USE_RANDOM_SEED = True
LIKES_FILE = LATENT_DIR / 'likes.pkl'
DISLIKES_FILE = LATENT_DIR / 'dislikes.pkl'

ONLINE_MODEL_PATHS_TF = {
    # 'albums': 'https://drive.google.com/file/d/1KfzkR-4Gpc8MMpI0hygCz2X9Yg-9-hPv/view?usp=sharing',
    'albums': 'https://drive.google.com/uc?id=1KfzkR-4Gpc8MMpI0hygCz2X9Yg-9-hPv',
    'cats': 'https://drive.google.com/uc?id=1ir8vo3OH0O-WBdagUDrE4t_MeXXHOL6l',
    'paintings': 'https://drive.google.com/uc?id=18bPwwhg6_LSdjzEFGo9Nd9ntjzuX5wzB',
    'wikiart': 'https://drive.google.com/uc?id=1VRxueZjsKCv0JkVfRA5TJzyF7hGSGocJ'
}


if __name__ == '__main__':
    print(SETTINGS_FILE)
    print(LIB_DIR)
    print(MODEL_DIR)

