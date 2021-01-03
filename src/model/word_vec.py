import spacy
import numpy as np


class WordVectorizer:
    def __init__(self, model='en_core_web_lg', output_shape=512):
        print(f'loading model {model}....')
        self.nlp = spacy.load(model)
        print('model loaded')
        self.output_shape = output_shape

    def get_word_vecs_from_text(self, text: str):
        tokens = self.nlp(text)

        vecs = []
        for token in tokens:
            if token.has_vector:
                vec = token.vector
                vec_len = vec.shape[0]
                if vec_len != self.output_shape:
                    # todo: check if amount remaining is < vec_len
                    remaining = self.output_shape - vec_len
                    vec = np.concatenate((vec, vec[:remaining]))

                vecs.append((token.text, vec.reshape(1, -1)))

        return vecs


if __name__ == '__main__':
    wv = WordVectorizer()
    text = 'cool aright great kick it'
    vecs = wv.get_word_vecs_from_text(text)
    print(vecs)
