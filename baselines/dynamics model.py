import numpy as np
import tensorflow as tf
from tensorflow.contrib.keras import backend
from baselines.common import ImprovedDataset
from baselines import bench
from baselines import nn_tools
import csv

LOG_DIR = '/Users/beni/PycharmProjects/baselines/baselines/logs/dynamics'
EPOCHS = 1
BATCH_SIZE = 256

def load_logs(dir):
    with open(dir) as file:
        reader = csv.reader(file)
        state, ac, rew, done = [], [], [], []
        for row in reader:
            state.append([float(s) for s in row[0][1:-1].split()])
            ac.append([float(s) for s in row[1][1:-1].split()])
            rew.append(float(row[2]))
            done.append(int(row[3] == 'True'))
    return state, ac, rew, done


def main():

    state_action_pairs, next_states, rewards, terminals = [], [], [], []
    for i in range(8):
        logdir = '/Users/beni/PycharmProjects/baselines/baselines/logs/RoboschoolWalker2d-v1/transitions{}.csv'.format(i)
        state, ac, rew, done = load_logs(logdir)
        for t in range(len(rew) - 1):
            state_action_pairs.append(state[t] + ac[t])
            next_states.append(state[t+1])
            rewards.append(rew[t])
            label = [0, 0]
            label[done[t]] = 1
            terminals.append(label)

    X = np.array(state_action_pairs)
    T = np.array(terminals)
    d = ImprovedDataset((X, T))
    d.prep_data(0.85)

    with tf.Session() as sess:
        backend.set_session(sess)
        done_classifier = nn_tools.NeuralNet(sess, d, LOG_DIR, regress=False, net_type='fc')
        done_classifier.train(EPOCHS, BATCH_SIZE, prep=False)
        done_classifier.test(plot=True)


if __name__ == "__main__":
    main()
