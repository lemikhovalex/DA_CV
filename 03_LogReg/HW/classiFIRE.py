import numpy as np
import pandas as pd
import math

class classiFIRE():
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.w = None
        self.__x = None
        self.__y = None

    def predict_proba(self, x: np.ndarray) -> np.ndarray:

        x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
        out = None
        if type(x) == pd.DataFrame:
            out = np.matmul(x, self.w.reshape(-1, 1))
        elif type(x) == np.ndarray:
            out = np.matmul(x, self.w.reshape(-1, 1))
        else:
            return
        out = np.apply_along_axis(func1d=lambda z: 1 / (1 + np.exp(z)),
                                  axis=1,
                                  arr=out)
        return np.array([out.reshape(-1), 1 - out.reshape(-1)]).T

    def log_loss_grad(self, indexes):
        _x = pd.DataFrame()
        _x['-y_i*xw'] = (-1) * np.multiply(np.matmul(self.__x[indexes], self.w), self.__y[indexes]).reshape(-1, )
        _x['scalar'] = _x['-y_i*xw'].apply(lambda z: 1. / (1 + np.exp(z)) * np.exp(z))
        _x['scalar'] = np.multiply(_x['scalar'].values, self.__y[indexes].reshape(-1) * (-1))
        _x.drop(columns=['-y_i*xw'], inplace=True)
        out = np.multiply(self.__x[indexes], _x['scalar'].values.reshape(-1, 1))
        return out.sum(axis=0)

    def fit(self, x, y, batch_size=1, learning_rate=1e-2, tol=1e-2, n_iter=np.inf) -> None:
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
        self.__x = x
        self.__y = y
        if batch_size <= 1:
            batch_size *= x.shape[0]
        batch_size = math.floor(batch_size)
        # y must be {-1, 1}
        if self.w is None:
            self.w = np.zeros(x.shape[1])
        w_prev = self.w
        _l = 0
        step = w_prev
        while ((np.linalg.norm(step) > tol) | (_l == 0)) & (_l < n_iter):
            w_prev = self.w
            sample_index = np.random.choice(x.shape[0], batch_size)
            step = learning_rate * self.log_loss_grad(sample_index)
            self.w -= step
            _l += 1
        self.__x = None
