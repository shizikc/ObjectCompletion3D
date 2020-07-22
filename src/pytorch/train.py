import argparse
import logging

import torch
import torch.optim as opt
import numpy as np
from mayavi import mlab
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from src.dataset.data_utils import plot_pc, plot_hist3d, set_fig, plot_mesh
from src.dataset.shapenet import ShapeDiffDataset
from .vae import VariationalAutoEncoder, VAELoss

# 'C:/Users/sharon/Documents/Research/data/dataset2019/shapenet/chair/train/partial_sub_group/03001627/'
logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--model_path', default='C:/Users/sharon/Documents/Research/ObjectDetection3D/model/model_20.pt',
                    help='Log dir [default: model]')
parser.add_argument('--train_path', default='C:/Users/sharon/Documents/Research/data/dataset2019/shapenet/chair/train'
                                            '/partial_sub_group/03001627/',
                    help='input data dir, should contain all needed folders [default: data]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 100]')
parser.add_argument('--bins', type=int, default=20, help='resolution of main cube [default: 10]')
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

criterion = VAELoss()

writer = SummaryWriter(args.log_dir)

r = lambda: np.random.rand()


# Define the Model
def get_model():
    vae = VariationalAutoEncoder(num_cubes=10 ** 3, threshold=0.001).double()
    return vae, opt.Adam(vae.parameters(), lr=0.0001, betas=(0.9, 0.999))


def loss_batch(model, input, prob_target, x_diff_target, loss_func, opt=None):
    """

    :param model:
    :param input:
    :param prob_pred:
    :param prob_target:
    :param x_diff_pred:
    :param x_diff_target:
    :param loss_func:
    :param opt:
    :return:
    """
    x_diff_pred, prob_pred, mu_out, sigma_out = model(input)
    #
    loss = loss_func(prob_pred, prob_target, x_diff_pred, x_diff_target)  # scalar

    if opt is not None:
        # training
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), input.shape[0]


# Train the Model
def fit(epochs, model, loss_func, op, train_dl, valid_dl):

    min_loss = 10000000

    for epoch in range(epochs):
        model.train()
        # model, input, prob_target, x_diff_target, loss_func, opt=None
        # x_partial, hist, edges, x_diff
        losses, nums = zip(
            *[loss_batch(model, x, h.flatten(), d, loss_func, op) for x, h, e, d in train_dl]
        )
        train_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        if epoch % 5 == 0:
            logging.info("Epoch : % 3d, Training error : % 5.5f" % (epoch, train_loss))

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, x, h.flatten(), d, loss_func) for x, h, e, d in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        # print to console
        if epoch % 5 == 0:
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

        t = 0.001
        ivl = 2 / args.bins

        for x_partial, hist, edges, x_diff in val_loader:
            # print(x.shape) torch.Size([1, 1740, 3])
            # print(h.shape) torch.Size([1, 10, 10, 10])
            # print(d.shape) torch.Size([1, 325, 3])
            # print(edges) # torch.Size([1, 3, 11])
            pred = model(x_partial)  # torch.Size([1, 10**3])
            pred_round = torch.relu(pred[0] - t)

            # pred_round
            # uniform sample from bounding box
            # cube indicator prediction
            pred_ind = ((pred[0] - t) > 0)  # torch.Size([1000])
            print("positive pred:", pred_ind.int().sum())

            h_ind = (hist[0] > 0).flatten()  # torch.Size([1000])
            d = h_ind.int().sum()
            print("positive gt:", d)

            ## Accuracy mesurment
            cond1 = torch.tensor(h_ind == True)
            cond2 = torch.tensor(pred_ind == h_ind)
            acc_ind = cond1 & cond2 # torch.Size([1000])

            logging.info("Indicator True Positive % f", acc_ind.float().sum()/d )
            logging.info("Indicator Accuracy % f", acc_ind.float().sum())

            # continues uniform distribution
            err = torch.abs(pred - hist[0].flatten())  # torch.Size([1, 1000])

            logging.info("Continues Accuracy % f", 1 - err[0].mean())

            mesh = np.meshgrid(edges[0][0][0:args.bins], edges[0][1][0:args.bins], edges[0][2][0:args.bins])
            h_ind = h_ind.reshape(args.bins, args.bins, args.bins)
            pred_ind = pred_ind.reshape(args.bins, args.bins, args.bins)

            ax = set_fig(edges[0])

            # plot_mesh(ax, mesh[1][idx_h], mesh[0][idx_h], mesh[2][idx_h], ivl=ivl, col="red") #gt box

            plot_mesh(ax, mesh[0][pred_ind], mesh[1][pred_ind], mesh[2][pred_ind], ivl=ivl, col="red")  # gt box
            ax.scatter(x_partial[0][:, 0], x_partial[0][:, 1], x_partial[0][:, 2], s=4, color="grey")  # gt diff
            ax.scatter(x_diff[0][:, 0], x_diff[0][:, 1], x_diff[0][:, 2], s=4, color="black")  # gt diff

            # ax.scatter(mesh[0][pred_ind], mesh[1][pred_ind], mesh[2][pred_ind], s=7, color="red")  # gt diff
            # ax.scatter( mesh[1][h_ind], mesh[0][h_ind], mesh[2][h_ind], s=4, col or="red")  # gt diff
            break
