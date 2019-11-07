import h5py
import numpy as np

# Original Labels are within 0 to 9. But proper label mapping is required as there are 8 classes.

class DataSampler(object):
    def __init__(self):
        self.shape = [55, 55, 3]
        self.X_total, self.y_total = self._load_data()
        self.total_size = self.X_total.shape[0]
        self.train_size = int(0.9*self.total_size)
        self.X_train, self.X_test = self.X_total[:self.train_size], self.X_total[self.train_size:]
        self.y_train, self.y_test = self.y_total[:self.train_size], self.y_total[self.train_size:]

    def _load_data(self):
        data_path = './data/ytf/ytf.h5'
        with h5py.File(data_path, 'r') as f:
            x = np.asarray(f.get('data'), dtype='float64')
            y = np.asarray(f.get('labels'), dtype='int32')
        x = x.transpose((0,2,3,1))
        x = x.reshape((x.shape[0], -1))
        x = x/127.5 - 1.0
        y[y==41] = 0
        return x, y

       
    def train(self, batch_size, label = False):
        indx = np.random.randint(low = 0, high = self.total_size, size = batch_size)

        if label:
            return self.X_total[indx, :], self.y_total[indx].flatten()
        else:
            return self.X_total[indx, :]

    def validation(self):
        return self.X_total, self.y_total

    def test(self):
        return self.X_test, self.y_test
    
    def data2img(self, data):
        return np.reshape(data, [data.shape[0]] + self.shape)

    def load_all(self):
        return self.X_total, self.y_total

