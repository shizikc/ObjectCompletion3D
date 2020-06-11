import argparse
import logging

import torch
import torch.optim as opt
import numpy as np
from mayavi import mlab
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


from src.dataset.data_utils import plot_pc, plot_hist3d
from src.dataset.shapenet import ShapeDiffDataset
from src.pytorch.model import HDM

# 'C:/Users/sharon/Documents/Research/data/dataset2019/shapenet/chair/train/partial_sub_group/03001627/'
logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--model_path', default='C:/Users/sharon/Documents/Research/ObjectDetection3D/model/model.pt',
                    help='Log dir [default: model]')
parser.add_argument('--train_path', default='C:/Users/sharon/Documents/Research/data/dataset2019/shapenet/chair/train'
                                            '/partial_sub_group/03001627/',
                    help='input data dir, should contain all needed folders [default: data]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 100]')
parser.add_argument('--bins', type=int, default=10, help='resolution of main cube [default: 10]')
parser.add_argument('--train', type=int, default=0, help='1 if training, 0 otherwise [default: 1]')
parser.add_argument('--eval', type=int, default=1, help='1 if evaluating, 0 otherwise [default:0]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
args = parser.parse_args()

# Model Life-Cycle
##################
dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
# Prepare the Data
if args.train:
    train_dataset = ShapeDiffDataset(args.train_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True)

# if args.eval:
val_path = args.train_path.replace('train', 'val')
val_dataset = ShapeDiffDataset(val_path)
val_loader = torch.utils.data.DataLoader(val_dataset, 1, shuffle=True)
# Binary segmentation function
criterion = torch.nn.BCELoss()

writer = SummaryWriter(args.log_dir)

r = lambda: np.random.rand()


# Define the Model

def get_model():
    hdm_model = HDM(10 ** 3).double()
    return hdm_model, opt.Adam(hdm_model.parameters(), lr=0.0001, betas=(0.9, 0.999))


def loss_batch(model, part_in, gt_hist, loss_func, opt=None):
    """

    :param gt_hist: in (batch_size, KxKxK, 1)
    :param part_in:
    :param model:
    :param loss_func: should expect pred, and dist
    :param opt:
    :return:
    """
    p = model(part_in)  # in (batch_size, KxKxK, 1)
    loss = loss_func(p, gt_hist)  # scalar

    if opt is not None:
        # training
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), part_in.shape[0]


# Train the Model
def fit(epochs, model, loss_func, op, train_dl, valid_dl):
    min_loss = 10000000
    for epoch in range(epochs):
        model.train()
        losses, nums = zip(
            *[loss_batch(model, x, h, loss_func, op) for x, h, e in train_dl]
        )
        train_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        if epoch % 20 == 0:
            logging.info("Epoch : % 3d, Training error : % 5.5f" % (epoch, train_loss))

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, x, h, loss_func) for x, h, e in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        # print to console
        if epoch % 20 == 0:
            logging.info("Epoch : % 3d, Validation error : % 5.5f" % (epoch, val_loss))

        writer.add_scalar('epoch_training_loss', train_loss, epoch)
        writer.add_scalar('epoch_validating_loss', val_loss, epoch)

        if val_loss < min_loss:
            min_loss = val_loss
            min_model = model

        # temporary save model
        if epoch % 50 == 0:
            torch.save(min_model.state_dict(), args.model_path)

    # save model
    torch.save(min_model.state_dict(), args.model_path)


# Make Predictions

if __name__ == '__main__':

    if args.train:
        # run model
        model, opt = get_model()

        # train model
        fit(args.max_epoch, model, criterion, opt, train_loader, val_loader)

    # evaluate
    if args.eval:
        model, _ = get_model()
        model.load_state_dict(torch.load(args.model_path, map_location=dev))
        model.eval()

        t = 0.0150
        rng = np.arange(-1, 1, 2 / args.bins, dtype=float)

        mesh_edge = np.meshgrid(rng, rng, rng)
        xv, yv, zv = mesh_edge[0].flatten(), mesh_edge[1].flatten(), mesh_edge[2].flatten()

        for x, h, e in val_loader:
            pred = model(x)
            # cube indicator prediction
            pred_ind = F.relu(pred - t); h_ind = F.relu(h); acc_ind = (pred_ind == h_ind).float()
            print(pred_ind.shape)
            logging.info("Indicator Accuracy % f", acc_ind[0].mean())

            # continues uniform distribution
            err = torch.abs(pred-h)
            logging.info("Continues Accuracy % f", 1-err[0].mean())
            break

            # plot partial and prediction
            # xv*h_ind, yv*h_ind, zv*h_ind
            plot_pc(x)

            # f = mlab.figure()
            # plot_hist3d(h[0].reshape(10, 10, 10))
            # plot_hist3d(pred[0].detach().reshape(10, 10, 10), color=(r(), r(), r()))
            # mlab.points3d(x, color=(r(), r(), r()))
            # mlab.show()
