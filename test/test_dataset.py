import unittest
from lib.data_prepare import get_dataloaders_from_index_data

class MyTestCase(unittest.TestCase):
    def test_dataset(self):
        get_dataloaders_from_index_data(
            data_dir='../data/PEMS04',
            n_hist=12,
            n_pred=12,
            norm=True,
            split_ratio=(20, 20),
            tod=True,
            dow=True,
            dom=False,
            batch_size=64,
            log=None
        )


if __name__ == '__main__':
    unittest.main()
