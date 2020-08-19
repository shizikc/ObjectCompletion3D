import unittest

from src.dataset.data_utils import plot_pc
from src.dataset.shapeDiff import ShapeDiffDataset

train_path = 'C:/Users/sharon/Documents/Research/data/dataset2019/shapenet/chair/'
obj_id = '03001627'


class TestShapeNet(unittest.TestCase):
    def test_shapenet(self):
        shapenet = ShapeDiffDataset(train_path, obj_id)
        x_partial, hist, edges, x_diff = shapenet[0]
        self.assertEqual(hist.shape, (20, 20, 20))
        self.assertEqual(x_partial.shape, (1740, 3))
        self.assertEqual(hist.flatten().sum(), 1.0)


if __name__ == '__main__':
    unittest.main()
