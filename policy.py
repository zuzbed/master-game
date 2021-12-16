import itertools
from tensorflow.keras.layers import Dense, Embedding, LSTMCell, StackedRNNCells
import tensorflow as tf

from config import Config as cfg
from data_preparation import Data


class Policy:
    def __init__(self):
        self.data = Data()
        self.guess_embedding = Embedding(self.data.max_guess_nbr + 1, cfg.guess_embedding_out)
        self.feedback_embedding = Embedding(self.data.max_feedback_nbr + 1, cfg.feedback_embedding_out)
        stacked_lstm = StackedRNNCells([LSTMCell(cfg.lstm_size) for _ in range(cfg.lstm_nbr)])
        self.lstm = tf.keras.layers.RNN(stacked_lstm, return_sequences=True, return_state=True)
        self.dense = Dense(self.data.max_guess_nbr)

    @property
    def variables(self):
        return [*self.guess_embedding.variables,
                *self.feedback_embedding.variables,
                *self.lstm.variables,
                *self.dense.variables]

    @property
    def named_variables(self):
        return dict(zip(map(str, itertools.count()), self.variables))

    def __call__(self, game_state, with_softmax=True):
        state = [[tf.zeros((1, cfg.lstm_size), dtype=tf.float32), tf.zeros((1, cfg.lstm_size), dtype=tf.float32)] for _ in range(cfg.lstm_nbr)]

        for guess, feedback in game_state:
            guess_tensor = tf.reshape(tf.convert_to_tensor(guess), (1,))
            feedback_tensor = tf.reshape(tf.convert_to_tensor(feedback), (1,))
            guess_embedded = self.guess_embedding(guess_tensor)
            feedback_embedded = self.feedback_embedding(feedback_tensor)

            combined = tf.concat([guess_embedded, feedback_embedded], axis=-1)
            new_shape = (1, 1, cfg.lstm_size) #(batch_size, max_time, n_input)

            whole_seq_output, final_memory_state, final_carry_state = self.lstm(tf.reshape(combined, new_shape), state)
            state = [final_memory_state, final_carry_state]

        logits = self.dense(whole_seq_output)
        if with_softmax:
            return tf.nn.softmax(logits)
        return logits


if __name__ == '__main__':
    p = Policy()
    g2 = p([(0, 10), (12, 2)]) # (guess_idx, feedback_idx)
