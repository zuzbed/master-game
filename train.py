import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from config import Config as cfg
from data_preparation import Data
from game import Game
from policy import Policy


def train(episodes_nbr, save_every):
    data = Data()
    policy = Policy()
    checkpoint = tf.train.Checkpoint(policy.named_variables)
    manager = tf.train.CheckpointManager(checkpoint, 'training_checkpoints_v18x', max_to_keep=10)
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    codes, last_feedback, turns = [], [], []
    for i in tqdm(range(episodes_nbr)):
        secret_code = np.random.randint(data.max_guess_nbr-1)
        game = Game(policy, secret_code)
        history = game.play()

        codes.append(data.guess_options[secret_code])
        last_feedback.append(history[-1][1])
        turns.append(len(history))

        G = -1
        optimizer = tf.keras.optimizers.SGD(learning_rate=cfg.reinforce_alpha * G)

        for j in reversed(range(1, len(history))):
            history_so_far = history[:j]
            next_action, _ = history[j]

            with tf.GradientTape(persistent=True) as g:
                action_logits = policy(history_so_far, with_softmax=False)
                loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(tf.convert_to_tensor(value=[next_action]),
                                                                                 data.max_guess_nbr), logits=action_logits)
            grads = g.gradient(loss, policy.variables)
            optimizer.apply_gradients(zip(grads, policy.variables))
            G -= 1
            optimizer._learning_rate = G * cfg.reinforce_alpha
            optimizer._learning_rate_tensor = None

        if i % save_every == 0 or i == episodes_nbr-1:
            save_path = manager.save()

            df = pd.DataFrame({'code': codes,
                               'last_feedback': last_feedback,
                               'feedback': [data.feedback_options[f] for f in last_feedback],
                               'turns': turns})
            df.to_csv(f'results_v18x/neural_network_training_results_lstm512_{i}.csv', sep=';')


if __name__ == '__main__':
    train(episodes_nbr=11000, save_every=50)
