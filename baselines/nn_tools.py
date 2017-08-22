import tensorflow as tf
from tensorflow.contrib.keras import layers, backend, initializers, activations, regularizers
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from baselines.common import Batcher
import os.path as osp

class NeuralNet():
    def __init__(self, session, data, logdir, net_type='cnn', regress=True, shift=False, reg=0.01, dropout=0.01, net_size=10,
                 fc_layers=1, conv_layers=3, half_fc=False, save=False):

        self.logs = logdir
        if tf.gfile.Exists(logdir):
            tf.gfile.DeleteRecursively(logdir)
        tf.gfile.MakeDirs(logdir)

        self.type = net_type
        self.regression = regress
        self.data = data
        self.sess = session
        self.onehot = True
        self.shift = shift
        self.reg = reg
        self.dropout = dropout
        self.network_size = net_size
        print('Initializing network')
        self.fc_layers = fc_layers
        self.conv_layers = conv_layers
        self.half_fc = half_fc
        self.init_network()
        print('Initialized')
        self.save = save
        self.saver = tf.train.Saver()

    def init_network(self):
        self.X = tf.placeholder(tf.float32, shape=self.data.x_shape)
        self.T = tf.placeholder(tf.float32, shape=self.data.t_shape)

        if self.type is 'fc':
            self._simple_fc()
        else:
            self._convnet()

        if self.regression:
            self.loss = tf.reduce_mean(tf.squared_difference(self.Y, self.T))
            self.accuracy = pearson_r(self.Y, self.T)
        else:
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.T, logits=self.Y))
            self.accuracy = tf.metrics.accuracy(tf.arg_max(self.T, 1), tf.arg_max(self.Y, 1))
        tf.summary.scalar('loss', self.loss)

        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
        self.summary = tf.summary.merge_all()
        self.sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

    def train(self, epochs, batch_size, split=0.85, prep=True):
        train_writer = tf.summary.FileWriter(osp.join(self.logs, 'train'), self.sess.graph)
        test_writer = tf.summary.FileWriter(osp.join(self.logs, 'test'))
        self.batch_size = batch_size
        if prep:
            self.data.prep_data(train_split=split, scale_to=1)
        print('Training...')
        i = 0
        for epoch in range(epochs):
            epoch_flag = False
            while not epoch_flag:
                x_batch, t_batch, epoch_flag = self.data.next_batch(batch_size)
                feed_dict = {self.X: x_batch, self.T: t_batch, backend.learning_phase(): 1}
                _, curr_loss, curr_acc, summ = self.sess.run([self.train_op, self.loss, self.accuracy, self.summary],
                                                             feed_dict=feed_dict)
                train_writer.add_summary(summ, i)
                i += 1
            print('Epoch: {}. Loss = {}. Accuracy = {}'.format(epoch, curr_loss, curr_acc))
            self.test(summary_info=(test_writer, i))

        if self.data.no_training_samples < 50000:
            y_train, self.training_loss, self.training_acc, _ = self.predict(self.data.x_train, T=self.data.t_train)
        else:  # sample training set - THIS CHANGES T_TRAIN
            self.data.t_train = self.data.t_train[:50000]
            y_train, self.training_loss, self.training_acc, _ = self.predict(self.data.x_train[:50000],
                                                                             T=self.data.t_train)
        if self.shift:
            self._set_shift(self.data.t_train, y_train)

        print('shape', y_train.shape)
        self.data.update_y(y_train, 'train')

        print('Training finished. Final Accuracy = {}'.format(self.training_acc))
        train_writer.close()
        test_writer.close()
        # example = [[0, 0, 3, 0, 0, 3, 2, 3, 0, 0, 3, 0, 3, 1, 1, 2, 0, 3, 1, 3, 0, 1, 3, 3, 3, 2, 2, 3, 3, 2]]
        # feed_dict = {self.X: example, backend.learning_phase(): 0}
        # Y, X1, X2 = self.sess.run([self.Y, self.X1, self.X2], feed_dict=feed_dict)
        # print(Y)
        # print(X1)
        # print(X2)

        if self.save:
            self.saver.save(self.sess, 'saved networks/cnn1')

    def predict(self, X, T=None):
        acc, loss = [], []
        summ = None
        pred = None
        batcher = Batcher(X, self.batch_size, T=T)
        if T is not None:
            while batcher.epoch_flag is False:
                x_batch, t_batch = batcher.next_batch()
                feed_dict = {self.X: x_batch, self.T: t_batch, backend.learning_phase(): 0}
                p, l, a, summ = self.sess.run([self.Y, self.loss, self.accuracy, self.summary], feed_dict=feed_dict)
                acc.append(a)
                loss.append(l)
                pred = np.vstack((pred, p)) if pred is not None else p
        else:
            while batcher.epoch_flag is False:
                x_batch = batcher.next_batch()
                feed_dict = {self.X: x_batch, backend.learning_phase(): 0}
                p = self.sess.run(self.Y, feed_dict=feed_dict)
                pred = np.vstack((pred, p)) if pred is not None else p
                # Haven't included shifted here yet

        if self.shift:
            pred = self._shift(pred)

        return (pred, np.mean(loss), np.mean(acc), summ) if T is not None else pred

    def test(self, test_set=None, plot=False, summary_info=None):
        if not test_set:
            T = self.data.t_test
            Y, loss, acc, summ = self.predict(self.data.x_test, T=T)
            if summary_info is not None:
                summary_info[0].add_summary(summ, summary_info[1])
            self.data.update_y(Y, 'test')
        else:
            # test_set.prep_data(train_split=0)
            T = test_set.t_data
            Y, loss, acc, _ = self.predict(test_set.x_data, T=T)

        print('Test data (test dataset): Accuracy = {}'.format(acc))
        if self.regression:
            accuracy = r2_score(T, Y)
            print("R^2 score: {}".format(accuracy))
        else:
            accuracy = acc
        if plot:
            self.fig, self.ax = plt.subplots(2, figsize=(20, 10))
            self.ax[0] = plot_accuracy(self.fig, self.ax[0], self.data.t_train, self.data.y_train,
                                       regress=self.regression, title='Train performance')
            self.ax[1] = plot_accuracy(self.fig, self.ax[1], T, Y, regress=self.regression, title='Test performance')
            plt.tight_layout()
            plt.show()

        return loss, accuracy

    def load_trained_network(self):
        self.saver.restore(self.sess, 'saved networks/cnn1')

    def _shift(self, Y):
        shifted = []
        for y in Y:
            shift = (y - self.b) / self.m
            shifted.append(shift)
        return shifted

    def _set_shift(self, T_train, Y_train):
        T_train = np.reshape(T_train, -1)
        self.m, self.b = np.polyfit(T_train, Y_train, 1)

    def _convnet(self):
        b_init = initializers.Constant(0.1)
        reg = regularizers.l2(self.reg)
        dropout = self.dropout
        N = self.network_size

        X = layers.Conv1D(N * 4, 2, activation='relu', padding='same', kernel_regularizer=reg)(self.X)
        # X = layers.BatchNormalization()(X)
        X = layers.Conv1D(N * 2, 4, activation='relu', padding='same', kernel_regularizer=reg)(X)
        X = layers.Conv1D(N, 8, activation='relu', padding='same', kernel_regularizer=reg)(X)
        X = layers.Flatten()(X)
        X = layers.Dense(N * 16, activation='relu', bias_initializer=b_init, kernel_regularizer=reg)(X)
        X = layers.Dropout(dropout)(X)
        self.Y = layers.Dense(self.data.t_shape[1])(X)

    def _simple_fc(self):
        b_init = initializers.Constant(0.1)
        reg = regularizers.l2(0.01)
        N = self.network_size
        dropout = 0.1

        # X_flat = layers.Flatten()(self.X)
        X = layers.Dense(2 * N * 16, activation='relu', bias_initializer=b_init, kernel_regularizer=reg)(self.X)
        X = layers.Dropout(dropout)(X)
        X = layers.Dense(2 * N * 8, activation='relu', bias_initializer=b_init, kernel_regularizer=reg)(X)
        X = layers.Dropout(dropout)(X)
        self.Y = layers.Dense(self.data.t_shape[1])(X)

def pearson_r(x, y):
    m_x, s_x = tf.nn.moments(x, [0, 1])
    m_y, s_y = tf.nn.moments(y, [0, 1])
    cov = tf.reduce_mean((x-m_x)*(y-m_y))
    pearson = cov/(tf.sqrt(s_x) * tf.sqrt(s_y) + 1e-9)
    return pearson

def plot_accuracy(fig, ax, T, Y, regress=True, title='Regression plot'):
    # plots regression/confusion matrix
    if regress:
        ax.scatter(T, Y)
        ax.set_xlabel('True values')
        ax.set_ylabel('Predicted values')
        ax.set_xlim([min(T), max(T)])
        ax.set_ylim([min(Y), max(Y)])
        x = np.linspace(0, 10, 20)
        ax.plot(x, x, "r--")        # x=y
        T = np.reshape(T, -1)
        m, b = np.polyfit(T, Y, 1)
        ax.plot(x, m*x + b, 'g-')
    else:
        if Y.shape[1] > 1:
            Y = [np.argmax(y) for y in Y]
        if T.shape[1] > 1:
            T = [np.argmax(t) for t in T]
        cm = confusion_matrix(y_true=T, y_pred=Y)
        plot_confusion_matrix(cm, fig, ax, ['miss', 'hit'])

    ax.set_title(title)

    return ax

def plot_confusion_matrix(cm, fig, ax, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    fig.colorbar(mappable=im, ax=ax)
    tick_marks = range(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
