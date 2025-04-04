import unittest
from datetime import datetime

import torch

from model import train
from model.STAEformer import STAEformer
from lib.data_prepare import get_dataloaders_from_index_data
import os
import yaml


def load_model(model_fp, model):
    with open(model_fp, 'rb') as f:
        checkpoint = torch.load(f, map_location='cpu')

    if "model" in checkpoint.keys():
        checkpoint_model = checkpoint["model"]
    else:
        checkpoint_model = checkpoint

    msg = model.load_state_dict(
        checkpoint_model,
    )
    print(msg)
    return model


class MyTestCase(unittest.TestCase):
    def test_test(self):
        DEVICE = torch.device("cpu")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = "PEMS03"
        dataset = dataset.upper()
        data_path = f"../data/{dataset}"
        model_name = STAEformer.__name__
        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        log_path = f"../logs/"
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log = os.path.join(log_path, f"{model_name}-{dataset}-{now}.log")
        log = open(log, "a")
        log.seek(0)
        log.truncate()

        with open(f"/Users/markbai/Documents/GitHub/STAEformer/model/STAEformer.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        cfg = cfg[dataset]
        model = STAEformer(**cfg["model_args"])
        model = load_model(
            '/Users/markbai/Documents/GitHub/STAEformer/STAEformer-PEMS03.pt',
            model
        )
        (
            _,
            _,
            testset_loader,
            SCALER,
        ) = get_dataloaders_from_index_data(
            data_path,
            tod=cfg.get("time_of_day"),
            dow=cfg.get("day_of_week"),
            batch_size=32,
            log=log,
        )
        train.test_model(model, testset_loader, log=log, device=DEVICE, scaler=SCALER)


if __name__ == '__main__':
    unittest.main()
