from tensorflow import keras
import json
import numpy as np
from my_preprocess import SEQUENCE_LENGTH, MAPPING_PATH


class MelodyGenerator:

    def __init__(self, model_path="model.h5"):

        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ["/"] * SEQUENCE_LENGTH


    def generate_melody(self, seed, mum_steps, max_sequence_length, temperature):

        # create seed with start symbols
        seed = seed.split()
        melody = seed
        seed = self._start_symbols +seed

        # map seed to int
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(mum_steps):

            # limit the seed to max_sequence_length
            seed = seed[-max_sequence_length:]

            # one-hot encode the seed
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))
            # (1, max_sequence_length, num of symbols in the vocabulary)
            onehot_seed = onehot_seed[np.newaxis, ...]

            # make a prediction
            probabilities = self.model.prediction(onehot_seed)[0]
