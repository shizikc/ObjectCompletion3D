import argparse
import logging

import torch
import torch.optim as opt
import numpy as np
from src.dataset.data_utils import plot_pc, plot_hist3d
from src.dataset.shapenet import ShapeDiffDataset
from src.pytorch.model import HDM

# 'C:/Users/sharon/Documents/Research/data/dataset2019/shapenet/chair/train/partial_sub_group/03001627/'

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--model_path', default='model', help='Log dir [default: model]')
parser.add_argument('--train_path', default='C:/Users/sharon/Documents/Research/data/dataset2019/shapenet/chair/train'
                                            '/partial_sub_group/03001627/',
                    help='input data dir, should contain all needed folders [default: data]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [256/512/1024/2048] [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=4, help='Epoch to run [default: 250]')
parser.add_argument('--train', type=int, default=1, help='1 if training, 0 otherwise [default: 1]')
parser.add_argument('--eval', type=int, default=1, help='1 if evaluating, 0 otherwise [default:0]')
parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 32]')
parser.add_argument('--num_of_samples', type=int, default=100, help='number of samples to train [default: 10]')
args = parser.parse_args()

# Model Life-Cycle
##################

# Prepare the Data
if args.train:
    train_dataset = ShapeDiffDataset(args.train_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True)

# if args.eval:
val_path = args.train_path.replace('train', 'val')
val_dataset = ShapeDiffDataset(val_path)
val_loader = torch.utils.data.DataLoader(val_dataset, args.batch_size, shuffle=True)

loss_fun = torch.nn.CrossEntropyLoss


# Define the Model

def get_model():
    model = HDM(10 ** 3).double()
    return model, opt.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))


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

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    min_loss = 10000000
    for epoch in range(epochs):
        model.train()
        losses, nums = zip(
            *[loss_batch(model, x, h, loss_func, opt) for x, h in train_dl]
        )
        train_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        if epoch % 20 == 0:
            logging.info("Epoch : % 3d, Training error : % 5.5f" % (epoch, train_loss))

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, x, h, loss_func) for x, h in valid_dl]
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
    if args.train:
        # run model
        model, opt = get_model()

        # train model
        fit(args.max_epoch, model, loss_fun, opt, train_loader, val_loader)
