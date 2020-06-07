import argparse
import logging

import torch
import torch.optim as opt
import numpy as np
from src.dataset.data_utils import plot_pc, plot_hist3d
from src.dataset.shapenet import ShapeDiffDataset
from src.pytorch.model import HDM

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--model_path', default='model', help='Log dir [default: model]')
# 'C:/Users/sharon/Documents/Research/data/dataset2019/shapenet/chair/train/partial_sub_group/03001627/'
parser.add_argument('--train_path', default='data',
                    help='input data dir, should contain all needed folders [default: data]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [256/512/1024/2048] [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--train', type=int, default=1, help='1 if training, 0 otherwise [default: 1]')
parser.add_argument('--eval', type=int, default=1, help='1 if evaluating, 0 otherwise [default:0]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--num_of_samples', type=int, default=100, help='number of samples to train [default: 10]')
parser.add_argument('--infer_threshold', type=float, default=0.9, help='inference probability [default: 0.9]')
parser.add_argument('--sampling_method', default='exact', help='exact or uniform [default: exact]')
parser.add_argument('--model', default='FilterModel', help='which model to train [default: FilterModel]')
args = parser.parse_args()

# Model Life-Cycle
##################

# Prepare the Data
if args.train:
    print(args.train_path)
    train_dataset = ShapeDiffDataset(args.train_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True)

if args.eval:
    val_path = args.train_path.replace('train', 'val')
    val_dataset = ShapeDiffDataset(val_path)
    val_loader = torch.utils.data.DataLoader(val_dataset, args.batch_size, shuffle=True)


# Define the Model

def get_model():
    model = HDM()
    return model, opt.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))


def loss_batch(model, part_in, gt_hist, loss_func, opt=None):
    """

    :param gt_hist:
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

    return loss.item()


# Train the Model

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    min_loss = 10000000
    for epoch in range(epochs):
        model.train()

        losses, nums = zip(
            *[loss_batch(model, loss_func, x, h, epoch, opt) for x, h in train_dl]
        )
        train_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        if epoch % 20 == 0:
            logging.info("Epoch : % 3d, Training error : % 5.5f" % (epoch, train_loss))

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, x, h) for x, h in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        # print to console
        if epoch % 20 == 0:
            logging.info("Epoch : % 3d, Validation error : % 5.5f" % (epoch, val_loss))

        if val_loss < min_loss:
            min_loss = val_loss
            min_model = model

        # temporary save model
        if epoch % 100 == 0:
            torch.save(min_model.state_dict(), args.model_path)

    # save model
    torch.save(min_model.state_dict(), args.model_path)


# Evaluate the Model

# Make Predictions


if __name__ == '__main__':
    for x_partial, hist in train_loader:
        plot_pc(x_partial)
        plot_hist3d(hist)
        break
