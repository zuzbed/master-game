from itertools import product
import numpy as np
import os
import pickle
from string import ascii_uppercase

from config import Config as cfg


class Data:
    def __init__(self):
        self.guess_options = self._load_guess_options()
        self.max_guess_nbr = len(self.guess_options)
        self.responses = self._load_responses_array()
        self.feedback_options = np.unique(self.responses).tolist()
        self.max_feedback_nbr = len(self.feedback_options)
        self.max_feedback_value = max(self.feedback_options)

    def _load_guess_options(self):
        files = [f for f in os.listdir()]
        if f'nn_options_c{cfg.colors}_p{cfg.places}.pickle' in files:
            options = pickle.load(open(f'nn_options_c{cfg.colors}_p{cfg.places}.pickle', 'rb'))
        else:
            opt = product(ascii_uppercase[:cfg.colors], repeat=cfg.places)
            options = []
            for o in opt:
                new = ''.join(o)
                options.append(new)
            pickle.dump(options, open(f'nn_options_c{cfg.colors}_p{cfg.places}.pickle', 'wb'))
        return options

    def _load_responses_array(self):
        files = [f for f in os.listdir()]
        if f'nn_responses_c{cfg.colors}_p{cfg.places}.npy' in files:
            responses_array = np.load(f'nn_responses_c{cfg.colors}_p{cfg.places}.npy')
        else:
            responses_array = self._prepare_responses_array()
            np.save(f'nn_responses_c{cfg.colors}_p{cfg.places}.npy', responses_array)
        return responses_array

    def _prepare_responses_array(self):
        # [code, guess]
        return np.array([[self._give_feedback(code, guess) for guess in self.guess_options] for code in self.guess_options])

    def _give_feedback(self, in_code, guess):
        blacks = [0] * len(in_code)
        whites = [0] * len(in_code)
        idxs = []
        code = list(in_code)
        for idx, c in enumerate(in_code):
            if c == guess[idx]:
                blacks[idx] = 1
                idxs.append(idx)
                code[idx] = 'ZZ'

        for idx, c in enumerate(code):
            if blacks[idx] == 0:
                if guess[idx] in code[0: idx:] + code[idx + 1::]:
                    if guess[idx] in code[0: idx:]:
                        fit = code[0: idx:].index(guess[idx])
                        if fit not in idxs:
                            idxs.append(code[0: idx:].index(guess[idx]))
                            whites[idx] = 1
                    elif guess[idx] in code[idx + 1::]:
                        fit = idx + code[idx::].index(guess[idx])
                        if fit not in idxs:
                            idxs.append(idx + code[idx::].index(guess[idx]))
                            whites[idx] = 1
        return sum(blacks) * 10 + sum(whites)


if __name__ == '__main__':
    d = Data()
