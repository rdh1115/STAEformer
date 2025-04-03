from pathlib import Path

import pandas as pd
import torch
import numpy as np
import os
from .utils import print_log, StandardScaler, vrange, generate_regression_task, generate_split


# ! X shape: (B, T, N, C)

def get_raw_data(dataset_path, split_ratio, n_hist, n_pred, norm):
    assert norm, 'Traffic data should be normalized for better performance'

    X_s, y_s = list(), list()
    scaler = None
    for split in ['train', 'val', 'test']:
        data_path = list(Path(dataset_path).glob(f'{split}*_hist{n_hist}_pred{n_pred}.npz'))

        if data_path:
            print(f'Loading {split} data from {data_path[0]}')
            data = np.load(data_path[0])
            X_s.append(data['x'])
            y_s.append(data['y'])
            if split == 'train':
                scaler = StandardScaler(mean=data['mean'], std=data['std'])
        else:
            print(f"preprocessed data not found at {dataset_path}, generating new data")
            h5_path = list(Path(dataset_path).glob('*.h5'))
            add_time_in_day, add_time_in_week = True, True
            if h5_path:
                print(f'Loading data from {h5_path[0]}')
                df = pd.read_hdf(h5_path[0])
                features, targets = generate_regression_task(
                    df, n_hist, n_pred,
                    add_time_in_day=add_time_in_day,
                    add_day_in_week=add_time_in_week,
                )
                features_fill, targets_fill = generate_regression_task(
                    df, n_hist, n_pred,
                    add_time_in_day=add_time_in_day,
                    add_day_in_week=add_time_in_week,
                    replace_drops=True,
                )

                (
                    (train_x, val_x, test_x,
                     train_y, val_y, test_y, scaler),
                    train_idx, val_idx, test_idx,
                ) = generate_split(
                    (features, features_fill),
                    (targets, targets_fill),
                    split_ratio,
                    norm
                )
            else:
                data_npz_path = os.path.join(dataset_path, "data.npz")
                print(f'Loading data from {data_npz_path}')
                # process X and get node features
                X = np.load(data_npz_path)["data"].astype(np.float32)
                features, targets = generate_regression_task(
                    X, n_hist, n_pred
                )
                features_fill, targets_fill = generate_regression_task(
                    X, n_hist, n_pred, replace_drops=True
                )

                (
                    (train_x, val_x, test_x,
                     train_y, val_y, test_y, scaler),
                    train_idx, val_idx, test_idx,
                ) = generate_split(
                    (features, features_fill),
                    (targets, targets_fill),
                    split_ratio,
                    norm
                )

            suffix = ''
            if add_time_in_day is not None:
                if add_time_in_day:
                    suffix += '_day'
                if add_time_in_week:
                    suffix += '_week'
            suffix += f'_hist{n_hist}_pred{n_pred}'
            np.savez_compressed(
                os.path.join(dataset_path, f'train{suffix}.npz'),
                x=train_x,
                y=train_y,
                idx=train_idx,
                mean=scaler.mean,
                std=scaler.std
            )
            np.savez_compressed(
                os.path.join(dataset_path, f'val{suffix}.npz'),
                x=val_x,
                y=val_y,
                idx=val_idx
            )
            np.savez_compressed(
                os.path.join(dataset_path, f'test{suffix}.npz'),
                x=test_x,
                y=test_y,
                idx=test_idx
            )
            return train_x, val_x, test_x, train_y, val_y, test_y, scaler

    return X_s + y_s + [scaler]


def get_dataloaders_from_index_data(
        data_dir,
        n_hist=12,
        n_pred=12,
        norm=True,
        split_ratio=(20, 10),
        tod=True,
        dow=False,
        dom=False,
        batch_size=64,
        log=None
):
    if 'PEMSBAY' or 'METRLA' in data_dir:
        split_ratio = (20, 10)

    x_train, x_val, x_test, y_train, y_val, y_test, scaler = get_raw_data(
        data_dir,
        split_ratio,
        n_hist,
        n_pred,
        norm
    )
    y_train, y_val, y_test = y_train[..., :1], y_val[..., :1], y_test[..., :1]
    print_log(f"Trainset:\tx-{x_train.shape}\ty-{y_train.shape}", log=log)
    print_log(f"Valset:  \tx-{x_val.shape}  \ty-{y_val.shape}", log=log)
    print_log(f"Testset:\tx-{x_test.shape}\ty-{y_test.shape}", log=log)

    trainset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_train), torch.FloatTensor(y_train)
    )
    valset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_val), torch.FloatTensor(y_val)
    )
    testset = torch.utils.data.TensorDataset(
        torch.FloatTensor(x_test), torch.FloatTensor(y_test)
    )

    trainset_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    valset_loader = torch.utils.data.DataLoader(
        valset, batch_size=batch_size, shuffle=False
    )
    testset_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )

    return trainset_loader, valset_loader, testset_loader, scaler
