# evaluate


import torch

from src.dataset.data_utils import plot_pc
from src.dataset.shapeDiff import ShapeDiffDataset
from src.pytorch.train import model_path, get_model

model, _ = get_model()
model.load_state_dict(torch.jit.load(model_path, map_location='cpu'))
model.eval()

train_path = "C:\\Users\\sharon\\Documents\\Research\\data\\dataset2019\\shapenet\\"

val_dataset = ShapeDiffDataset(train_path, "04256520", val=True)
val_loader = torch.utils.data.DataLoader(val_dataset, 1, shuffle=True)

if __name__ == '__main__':

    for i, (x_partial, hist, edges, x_diff) in enumerate(val_loader):
        pred = model(x_partial)
        # pred_round = torch.relu(pred[0] - threshold)

        # uniform sample from bounding box
        plot_pc([x_partial, pred], colors=("black", "red"))
        plot_pc([x_partial[0], x_diff[0]], colors=("black", "red"))

