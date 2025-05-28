import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Initialize global scaler
scaler = MinMaxScaler(feature_range=(0, 1))

def scale_data(data):
    """
    Fit and transform the data using MinMaxScaler.
    Should be used only during training phase.
    """
    return scaler.fit_transform(data)

def transform_data(data):
    """
    Only transform new input data (use after scaler is fit).
    """
    return scaler.transform(data)

def inverse_scale(scaled_data):
    """
    Convert scaled data back to original scale.
    """
    return scaler.inverse_transform(scaled_data)

def split_sequence(sequence, n_steps):
    """
    Split a univariate sequence into samples for LSTM/GRU.
    """
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
  