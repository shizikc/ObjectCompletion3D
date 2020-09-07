# evaluate


import torch

from src.pytorch.visualization import plot_pc_mayavi
from src.dataset.shapeDiff import ShapeDiffDataset
from src.pytorch.train import model_path, get_model, bins, batch_size, train_path

model, _ = get_model()
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

train_dataset = ShapeDiffDataset(train_path, bins, dev="cpu", seed=0)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)

if __name__ == '__main__':
    x_partial, x_diff, hist = next(iter(train_loader))
    pred = model(x_partial.transpose(2, 1), pred_pc=True)
    # pred_round = torch.relu(pred[0] - threshold)

    # # uniform sample from bounding box
    b_gt = hist.flatten() > 0.
    b_pred = pred[1] > 0.01
    c = model.voxel_centers[b_gt].detach().numpy()
    t = model.voxel_centers[b_pred].detach().numpy()

    # t = model.centers.view(b_gt.shape[0], 3)[b_pred].detach().numpy()
    # t = model.centers.view(b_gt.shape[0], 3)[b_gt].detach().numpy()

    plot_pc_mayavi([c, t, x_diff], colors=((1., 1., 1.), (0., 0., 1.), (1., 0., 0.))) #, pred[0].detach().numpy()

    plot_pc_mayavi([pred[0].detach().numpy(), x_diff], colors=((1., 1., 1.), (0., 0., 1.)))
