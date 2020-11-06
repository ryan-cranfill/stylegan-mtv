# Stylegan Model Class
import cv2
import pickle
import numpy as np
import tensorflow as tf

from src.dnnlib import tflib
from src.utils import find_stylegan_models, load_latent_reps

AVAILABLE_MODELS = find_stylegan_models()
LOAD_ONLY = ['wikiart']
BATCH_SIZE = 1


class StyleGANModel:
    img_fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    gpu_options = tf.GPUOptions(
        allow_growth=True,
        visible_device_list='0',
    )

    def __init__(self, model_path: str, random_seed=False, reduced_memory=True):
        if random_seed:
            print('i am using random seed', random_seed)
            if isinstance(random_seed, int):
                np.random.seed(random_seed)
            else:
                np.random.seed(420)

        if reduced_memory:
            print('using reduced GPU memory')
            self.gpu_options.per_process_gpu_memory_fraction = 0.85
        else:
            print('i will use all the GPU memory i want and you cant stop me')

        self.model_path = model_path
        self.model, self.graph,  self.sess, self.input_shape, self.output_shape = self.load_model(model_path)
        self.device = 'gpu'
        self.base_dlatent = None

    def load_model(self, model_path=None):
        if not model_path:
            model_path = self.model_path

        print('loading model from', model_path, '....')

        graph = tf.get_default_graph()

        with graph.as_default():
            # Initialize TensorFlow session.
            config = tf.ConfigProto(gpu_options=self.gpu_options)
            config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
            sess = tf.Session(config=config).__enter__()

            # Import trained network
            with open(model_path, 'rb') as file:
                G, D, Gs = pickle.load(file)
                # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
                # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
                # Gs = Long-term average of the generator. Yields higher-quality results than instantaneous snapshot.

        input_shape = Gs.input_shape[1]
        output_shape = Gs.output_shape[2:]
        print('model loaded, input shape:', input_shape, 'output shape:', output_shape)
        return Gs, graph, sess, input_shape, output_shape

    def run_image(self, latent_vecs: np.ndarray, as_bytes=True, dlatent=None, latent_strength=0.25):
        with self.sess.as_default():
            with self.graph.as_default():
                src_dlatents = self.model.components.mapping.run(latent_vecs, None)  # [seed, layer, component]

                if dlatent is not None:
                    src_dlatents = (src_dlatents * latent_strength) + dlatent

                images = self.model.components.synthesis.run(src_dlatents, randomize_noise=False,
                                                             truncation_psi=0.5, output_transform=self.img_fmt)

        img = images[0]

        final_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if as_bytes:
            out_img = cv2.imencode('.jpg', final_img, self.encode_param)[1].tobytes()
        else:
            out_img = final_img

        return out_img

    def run_images(self, latent_vecs: np.ndarray, use_base_dlatent=False):
        with self.sess.as_default():
            with self.graph.as_default():
                src_dlatents = self.model.components.mapping.run(latent_vecs, None)  # [seed, layer, component]

                if use_base_dlatent and self.base_dlatent is not None:
                    src_dlatents = (src_dlatents * 0.24) + self.base_dlatent

                images = self.model.components.synthesis.run(src_dlatents, randomize_noise=False,
                                                             truncation_psi=0.5, output_transform=self.img_fmt, minibatch_size=8)

        # final_imgs = images
        final_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
        return final_imgs

    def set_base_dlatent(self, latent_rep):
        self.base_dlatent = latent_rep
