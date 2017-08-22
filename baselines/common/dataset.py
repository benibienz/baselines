import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import minmax_scale

class Dataset(object):
    def __init__(self, data_map, deterministic=False, shuffle=True):
        self.data_map = data_map
        self.deterministic = deterministic
        self.enable_shuffle = shuffle
        self.n = next(iter(data_map.values())).shape[0]
        self._next_id = 0
        self.shuffle()

    def shuffle(self):
        if self.deterministic:
            return
        perm = np.arange(self.n)
        np.random.shuffle(perm)

        for key in self.data_map:
            self.data_map[key] = self.data_map[key][perm]

        self._next_id = 0

    def next_batch(self, batch_size):
        if self._next_id >= self.n and self.enable_shuffle:
            self.shuffle()

        cur_id = self._next_id
        cur_batch_size = min(batch_size, self.n - self._next_id)
        self._next_id += cur_batch_size

        data_map = dict()
        for key in self.data_map:
            data_map[key] = self.data_map[key][cur_id:cur_id+cur_batch_size]
        return data_map

    def iterate_once(self, batch_size):
        if self.enable_shuffle: self.shuffle()

        while self._next_id <= self.n - batch_size:
            yield self.next_batch(batch_size)
        self._next_id = 0

    def subset(self, num_elements, deterministic=True):
        data_map = dict()
        for key in self.data_map:
            data_map[key] = self.data_map[key][:num_elements]
        return Dataset(data_map, deterministic)

class ImprovedDataset():
    def __init__(self, data=None, preserve_random_idx=True):
        self.batch_idx = 0
        self.random_idx = None
        self.preserve_random_idx = preserve_random_idx
        self.y_train = None
        self.y_test = None
        self.prepped = False
        self.df = pd.DataFrame(columns=['split', 'X', 'T', 'Y'])
        self.encode_dict = None  # overwrite with encode dict
        self.dims = {'X': 0, 'T': 0}

        if data is not None:
            self.set_data(data[0], data[1])

    def set_data(self, X, T):
        # takes in tuple of lists (X, T) or sets from df columns specified by string
        # may not be useful for multi dimensional X
        if isinstance(T, str):
            X = self.df[X].values
            T = self.df[T].values
        self.x_data = X if len(X.shape) > 1 else X[None].T
        self.t_data = T if len(T.shape) > 1 else T[None].T

        s = []
        for c in range(self.x_data.shape[1]):
            s.append(pd.Series(self.x_data[:, c], name='X{}'.format(c)))
            self.dims['X'] += 1
        for c in range(self.t_data.shape[1]):
            s.append(pd.Series(self.t_data[:, c], name='T{}'.format(c)))
            self.dims['T'] += 1

        self.df = pd.concat(s, axis=1)
        self.df['split'] = np.nan
        self._update_shapes()

    def prep_data(self, train_split, scale_to=None, randomize=True, bins=False, subsample=None, stratify=False,
                  drop_cols=True, onehot=True):
        # drop cols = False for debugging and looking at data
        # shuffles data, scales and splits training + test sets - can also bin into classification data
        # must take in column arrays

        if randomize:
            self._subsample(subsample, drop_cols=drop_cols, stratify=stratify)

        self.t_data = self._df_to_array('T')
        self.x_data = self._df_to_array('X')

        if scale_to is not None:
            self.t_data = minmax_scale(self.t_data, feature_range=(0, scale_to))
            if not drop_cols:
                self.df['T scaled'] = self.t_data
            else:
                self.df['T'] = self.t_data
        if bins:
            self.t_data = self._bin(self.t_data, bins=bins)
            if not drop_cols:
                self.df['T binned'] = [np.argmax(t) for t in self.t_data]
            else:
                self.df['T'] = [np.argmax(t) for t in self.t_data]

        # if len(self.t_data.shape) == 1:
        #     self.t_data = self.t_data[None].T
        # if len(self.x_data.shape) == 1:
        #     self.x_data = self.t_data[None].T

        if train_split:
            train_amount = round(self.df.shape[0] * train_split)
            self.df.loc[:train_amount, 'split'] = 'train'
            self.df.loc[train_amount:, 'split'] = 'test'
            self.t_train = self.t_data[:train_amount, :]
            self.x_train = self.x_data[:train_amount, :]
            self.t_test = self.t_data[train_amount:, :]
            self.x_test = self.x_data[train_amount:, :]
            self.no_training_samples = len(self.t_train)

        self.prepped = True
        self._update_shapes()

    def next_batch(self, size):
        epoch_flag = False

        if self.batch_idx < self.no_training_samples - size:
            x_batch = self.x_train[self.batch_idx:self.batch_idx + size, :]
            t_batch = self.t_train[self.batch_idx:self.batch_idx + size, :]
            self.batch_idx += size
        else:
            x_batch = self.x_train[self.batch_idx:, :]
            t_batch = self.t_train[self.batch_idx:, :]
            self.batch_idx = 0
            epoch_flag = True

        return x_batch, t_batch, epoch_flag

    def plot_hists(self):
        fig, ax = plt.subplots(3)

        ax[0].hist(self.df['T'].values, 50, alpha=0.75)
        ax[0].set_title('All Data')
        if self.prepped:
            ax[1].hist(self.df.loc[self.df['split'] == 'train', 'T'], 50, alpha=0.75)
            ax[1].set_title('Train')
            ax[2].hist(self.df.loc[self.df['split'] == 'test', 'T'], 50, alpha=0.75)
            ax[2].set_title('Test')
        plt.show()

    def encode(self, x, onehot=True):
        # encodes strings using a dict
        # will do nothing if no encode dict
        if self.encode_dict is not None:
            if onehot:
                encoded = np.zeros((len(x), len(self.encode_dict)), dtype=np.int8)
                for i in range(len(x)):
                    encoded[i, self.encode_dict[x[i]]] = 1
            else:
                encoded = [self.encode_dict[i] for i in x]
        else:
            encoded = x
        return encoded

    def update_y(self, Y, split):
        if split is 'train':
            self.y_train = Y
        elif split is 'test':
            self.y_test = Y
        if Y.shape[1] > 1:
            Y = [np.argmax(y) for y in Y]
        self.df.loc[self.df['split'] == split, 'Y'] = Y

    def _data_check(self, drop=True):
        duplicates = self.df.duplicated(subset='X')
        if drop:
            self.df = self.df.drop_duplicates(subset='X')
        else:
            self.df['duplicate?'] = duplicates
        print('Duplicate count:\n', duplicates.value_counts())


    def _subsample(self, size, stratify=False, drop_cols=True, bins=2):
        # randomizes and subsamples data
        n = size if size is not None else self.df.shape[0]
        if stratify:
            binned = self._bin(self.t_data, bins=bins)
            self.df['T binned'] = [np.argmax(t) for t in binned]
            strat_frames = []
            sanple_n = int(n / bins)
            n = sanple_n*bins
            for i in range(bins):
                strat_frames.append(
                    self.df.loc[self.df['T binned'] == i].sample(n=sanple_n, replace=True).reset_index(drop=True))
            self.df = pd.concat(strat_frames)

        self.df = self.df.sample(n=n).reset_index(drop=drop_cols)

    def _df_to_array(self, dat_type):
        arr = np.empty((self.df.shape[0], self.dims[dat_type]))
        for i in range(self.dims[dat_type]):
            arr[:, i] = self.df[dat_type + str(i)]
        return arr

    def _update_shapes(self):
        self.x_shape = [None] + list(self.x_data.shape)[1:]
        self.t_shape = [None] + list(self.t_data.shape)[1:]

    def _update_df(self):
        # depends on data type
        pass

    def _bin(self, t_data, bins=10):
        binned = []
        _, bin_edges = np.histogram(t_data, bins=bins)
        for t in t_data:
            onehot = np.zeros(bins, dtype=np.int8)
            for i in range(1, bins + 1):
                if t < bin_edges[i]:
                    onehot[i-1] = 1
                    break
                elif i == bins:
                    onehot[-1] = 1
            binned.append(onehot)
        return np.array(binned)

class Batcher():
    def __init__(self, X, size, T=None):
        self.X = X
        self.T = T
        self.size = size
        self.batch_idx = 0
        self.epoch_flag = False

    def next_batch(self):
        if self.batch_idx < len(self.X) - self.size:
            x_batch = self.X[self.batch_idx:self.batch_idx + self.size, :]
            t_batch = self.T[self.batch_idx:self.batch_idx + self.size, :] if self.T is not None else None
            self.batch_idx += self.size
        else:
            x_batch = self.X[self.batch_idx:, :]
            t_batch = self.T[self.batch_idx:, :] if self.T is not None else None
            self.batch_idx = 0
            self.epoch_flag = True
        return x_batch if self.T is None else (x_batch, t_batch)


def iterbatches(arrays, *, num_batches=None, batch_size=None, shuffle=True, include_final_partial_batch=True):
    assert (num_batches is None) != (batch_size is None), 'Provide num_batches or batch_size, but not both'
    arrays = tuple(map(np.asarray, arrays))
    n = arrays[0].shape[0]
    assert all(a.shape[0] == n for a in arrays[1:])
    inds = np.arange(n)
    if shuffle: np.random.shuffle(inds)
    sections = np.arange(0, n, batch_size)[1:] if num_batches is None else num_batches
    for batch_inds in np.array_split(inds, sections):
        if include_final_partial_batch or len(batch_inds) == batch_size:
            yield tuple(a[batch_inds] for a in arrays)

if __name__ == '__main__':
    data = (np.array([[1,2], [3,4]]), np.array([[5,6], [7,8]]))
    d = ImprovedDataset(data)
    d.prep_data(0.85)
    print(d.df.head())