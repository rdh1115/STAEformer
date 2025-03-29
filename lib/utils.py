import numpy as np
import pandas as pd
import torch
import pickle
import random
import os
import json
from sklearn.model_selection import train_test_split


class Scaler:
    def transform(self, data):
        raise NotImplementedError

    def inverse_transform(self, data, args=None):
        raise NotImplementedError

    def to_device(self, device):
        raise NotImplementedError


class StandardScaler(Scaler):
    """
    z-score norm the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data, device='cpu', args=None):
        data_shape = data.shape
        D = data_shape[-1]

        return (data * self.std) + self.mean

    def __str__(self):
        return f"StandardScaler(mean={self.mean}, std={self.std})"

    def to_device(self, device):
        for attr in ['mean', 'std']:
            attr_val = getattr(self, attr)
            if isinstance(attr_val, np.ndarray):
                setattr(self, attr, torch.tensor(attr_val, dtype=torch.float, device=device))
            elif isinstance(attr_val, torch.Tensor):
                setattr(self, attr, attr_val.to(device))
            elif isinstance(attr_val, float) or isinstance(attr_val, int):
                setattr(self, attr, torch.tensor(attr_val, device=device))
            else:
                raise NotImplementedError('scaler attributes should be torch.Tensor or np.ndarray or float/int')
        return self


def normalize(train_x, val_x, test_x, train_y, val_y, test_y):
    # all inputs have shape [num_points, num_time_points, n_nodes, node_dim]
    mean, std = train_x[..., 0].mean(), train_x[..., 0].std()
    # only normalize the sensor data, not the time_in_day data
    scaler = StandardScaler(np.array(mean), np.array(std))

    train_x[..., 0] = scaler.transform(train_x[..., 0])
    val_x[..., 0] = scaler.transform(val_x[..., 0])
    test_x[..., 0] = scaler.transform(test_x[..., 0])
    return train_x, val_x, test_x, train_y, val_y, test_y, scaler


def masked_mae_loss(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


class MaskedMAELoss:
    def _get_name(self):
        return self.__class__.__name__

    def __call__(self, preds, labels, null_val=0.0):
        return masked_mae_loss(preds, labels, null_val)


def print_log(*values, log=None, end="\n"):
    print(*values, end=end)
    if log:
        if isinstance(log, str):
            log = open(log, "a")
        print(*values, file=log, end=end)
        log.flush()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print("Unable to load data ", pickle_file, ":", e)
        raise
    return pickle_data


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def set_cpu_num(cpu_num: int):
    os.environ["OMP_NUM_THREADS"] = str(cpu_num)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
    os.environ["MKL_NUM_THREADS"] = str(cpu_num)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
    torch.set_num_threads(cpu_num)


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return f"Shape: {obj.shape}"
        elif isinstance(obj, torch.device):
            return str(obj)
        else:
            return super(CustomJSONEncoder, self).default(obj)


def vrange(starts, stops):
    """Create ranges of integers for multiple start/stop

    Parameters:
        starts (1-D array_like): starts for each range
        stops (1-D array_like): stops for each range (same shape as starts)
        
        Lengths of each range should be equal.

    Returns:
        numpy.ndarray: 2d array for each range
        
    For example:

        >>> starts = [1, 2, 3, 4]
        >>> stops  = [4, 5, 6, 7]
        >>> vrange(starts, stops)
        array([[1, 2, 3],
               [2, 3, 4],
               [3, 4, 5],
               [4, 5, 6]])

    Ref: https://codereview.stackexchange.com/questions/83018/vectorized-numpy-version-of-arange-with-multiple-start-stop
    """
    stops = np.asarray(stops)
    l = stops - starts  # Lengths of each range. Should be equal, e.g. [12, 12, 12, ...]
    assert l.min() == l.max(), "Lengths of each range should be equal."
    indices = np.repeat(stops - l.cumsum(), l) + np.arange(l.sum())
    return indices.reshape(-1, l[0])


def print_model_params(model):
    param_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("%-40s\t%-30s\t%-30s" % (name, list(param.shape), param.numel()))
            param_count += param.numel()
    print("%-40s\t%-30s" % ("Total trainable params", param_count))


def fill_drops(data):
    if len(data.shape) == 3:
        T, V, D = data.shape
    elif len(data.shape) == 4:
        N, T, V, D = data.shape
    else:
        raise NotImplementedError('expected data shape to have 3 or 4 dimensions')

    # Replace sudden drops with historical averages if enabled
    drops = 0
    for v in range(V):  # Iterate over nodes
        for d in range(D):  # Iterate over features
            t = 1
            while t < T:
                if data[t, v, d] == 0 and data[t - 1, v, d] != 0:  # Check for sudden drops
                    start_t = t
                    while t < T and data[t, v, d] == 0:
                        t += 1
                    historical_avg = np.mean(data[:start_t, v, d][data[:start_t, v, d] != 0])
                    noise = np.random.normal(loc=0, scale=0.05 * historical_avg, size=(t - start_t))
                    data[start_t:t, v, d] = historical_avg + noise

                    drops += 1
                else:
                    t += 1
    print(f'Replaced {drops} sudden drops with historical averages')
    return data


def generate_regression_task(
        data, n_hist, n_pred,
        add_time_in_day=True, add_day_in_week=False,
        replace_drops=False,
):
    """
    Generate features and targets for regression tasks from a DataFrame or NumPy array.

    :param data: DataFrame (shape [T, V]) or NumPy array (shape [T, V, D]) of sensor data
    :param n_hist: Number of observed time points
    :param n_pred: Time points to be predicted
    :param add_time_in_day: Whether to add time-in-day information (only for DataFrame inputs)
    :param add_day_in_week: Whether to add day-in-week information (only for DataFrame inputs)
    :param replace_drops: Whether to replace sudden drops (values dropping to 0) with historical averages
    :return: Features and targets as NumPy arrays
    """
    features, targets = [], []

    # Check if input is a DataFrame or NumPy array
    is_dataframe = isinstance(data, pd.DataFrame)

    if is_dataframe:
        df = data
        T, V = df.shape
        data_np = np.expand_dims(df.values, axis=-1)  # Convert DataFrame to NumPy array
        data_list = [data_np]

        # Handle time-based features if index is datetime
        if not df.index.values.dtype == np.dtype("<M8[ns]"):
            add_time_in_day = False
            add_day_in_week = False
        if add_time_in_day:
            time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
            time_in_day = np.tile(time_ind, [1, V, 1]).transpose((2, 1, 0))
            data_list.append(time_in_day)
        if add_day_in_week:
            day_in_week = np.zeros(shape=(T, V, 7))
            day_in_week[np.arange(T), :, df.index.dayofweek] = 1
            data_list.append(day_in_week)

        # Combine all features into a single NumPy array
        data_np = np.concatenate(data_list, axis=-1)
    else:
        data_np = data

    T, V, D = data_np.shape
    if replace_drops:
        data_np = fill_drops(data_np)

    # Create indices for slicing data into features and targets
    indices = [
        (i, i + (n_hist + n_pred))
        for i in range(T - (n_hist + n_pred) + 1)
    ]

    for i, j in indices:
        features.append(data_np[i: i + n_hist, ...])
        targets.append(data_np[i + n_hist: j, ...])

    # Convert features and targets to arrays
    features = np.stack(features, axis=0)
    targets = np.stack(targets, axis=0)

    return features, targets


def generate_split(X, y, split_ratio, norm):
    X, X_fill = X
    y, y_fill = y
    num_data = X.shape[0]
    assert num_data == y.shape[0]

    test_split, valid_split = split_ratio
    test_split, valid_split = test_split / 100, valid_split / 100
    print(
        f"creating train/valid/test datasets, ratio: "
        f"{1.0 - test_split - valid_split:.1f}/{valid_split:.1f}/{test_split:.1f}"
    )
    valid_split = valid_split / (1.0 - test_split)

    # no shuffle for traffic data to avoid train data leakage in test data since time slices overlap
    shuffle = False
    train_valid_idx, test_idx = train_test_split(
        np.arange(num_data),
        test_size=test_split,
        shuffle=shuffle,
    )
    train_idx, valid_idx = train_test_split(
        train_valid_idx,
        test_size=valid_split,
        shuffle=shuffle,
    )
    train_x, val_x, test_x = X_fill[train_idx], X[valid_idx], X[test_idx]
    train_y, val_y, test_y = y_fill[train_idx], y[valid_idx], y[test_idx]
    if norm:
        return normalize(train_x, val_x, test_x, train_y, val_y, test_y), train_idx, valid_idx, test_idx
    else:
        return (train_x, val_x, test_x, train_y, val_y, test_y, None), train_idx, valid_idx, test_idx
