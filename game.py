import numpy as np
import tensorflow as tf

from config import Config as cfg
from data_preparation import Data


class Game:

    def __init__(self, policy, code_idx):
        self.data = Data()
        self.policy = policy
        self.code_idx = code_idx
        self.start_guess_idx = np.random.randint(self.data.max_guess_nbr)

    def _code_from_index(self, idx):
        return self.data.guess_options[idx]

    def _feedback_from_index(self, idx):
        return self.data.feedback_options[idx]

    def _index_from_guess(self, guess):
        return self.data.guess_options.index(guess)

    def _index_from_feedback(self, feedback):
        return self.data.feedback_options.index(feedback)

    def print_full_history(self, history):
        print()
        for guess_idx, feedback_idx in history:
            print(f'{self._code_from_index(guess_idx)} - {self._feedback_from_index(feedback_idx)}')
        print()

    def _select_next_action(self, action_distribution):
        next_guess_idx = tf.random.categorical(action_distribution[0,:,:], 1)[0][0]
        return next_guess_idx

    def play(self):
        start_feedback = self.data.responses[self.start_guess_idx, self.code_idx]
        history = [(self.start_guess_idx, self._index_from_feedback(start_feedback))]
        for _ in range(cfg.max_turns - 1):
            guess_idx = self._select_next_action(self.policy(history))
            feedback = self.data.responses[guess_idx, self.code_idx]
            history.append((guess_idx, self._index_from_feedback(feedback)))
            if feedback == self.data.max_feedback_value:
                # print(f'YOU WON! guesses: {len(history)}')
                break
        # self.print_full_history(history)
        return history
