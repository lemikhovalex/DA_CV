import numpy as np
import pandas as pd


class classiFIRE():
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.w = None

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        self.w = self.w.reshape(-1, 1)
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
        out = None
        if type(x) == pd.DataFrame:
            out = np.matmul(x, self.w)
        elif type(x) == np.ndarray:
            out = np.matmul(x, self.w)
        else:
            return
        out = np.apply_along_axis(func1d=lambda z: 1 / (1 + np.exp(z)),
                                  axis=1,
                                  arr=out)
        return np.array([out.reshape(-1), 1 - out.reshape(-1)]).T

    def log_loss_grad(self, x: pd.DataFrame, y):
        _x = pd.DataFrame()
        _x['-y_i*xw'] = (-1) * np.multiply(np.matmul(x, self.w), y).reshape(-1, )
        _x['scal'] = _x['-y_i*xw'].apply(lambda z: 1. / (1 + np.exp(z)) * np.exp(z))
        _x['scal'] = np.multiply(_x['scal'].values, y.reshape(-1) * (-1))
        _x.drop(columns=['-y_i*xw'], inplace=True)
        out = np.multiply(x, _x['scal'].values.reshape(-1, 1))
        return out.sum(axis=0)

    def fit(self, x, y, batch_size=1, learning_rate=1e-2, tol=1e-2) -> None:
        x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
        if batch_size <= 1:
            batch_size *= x.shape[0]
        else:
            batch_size *= x.shape[0]
        batch_size = int(batch_size)
        # y must be {-1, 1}
        self.w = np.zeros(x.shape[1])
        w_prev = self.w
        _l = 0
        step = w_prev
        while np.linalg.norm(step) > tol or _l == 0:
            w_prev = self.w
            sample_indx = np.random.choice(x.shape[0], batch_size)
            step = learning_rate * self.log_loss_grad(x[sample_indx], y[sample_indx])
            self.w -= step
            _l += 1
