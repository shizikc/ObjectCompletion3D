import argparse
import logging

import torch
import torch.optim as opt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from src.dataset.shapenet import ShapeDiffDataset
from src.pytorch.vae import VariationalAutoEncoder, VAELoss
from src.dataset.data_utils import plot_pc

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--model_path', default='C:/Users/sharon/Documents/Research/ObjectCompletion3D/model/model.pt')
                    # default='/home/coopers/models/model.pt')  #
parser.add_argument('--train_path', default='C:\\Users\\sharon\\Documents\\Research\\data\\dataset2019\\shapenet\\chair\\')
                    # default='/home/coopers/data/chair/')  #
parser.add_argument('--max_epoch', type=int, default=1, help='Epoch to run [default: 100]')
parser.add_argument('--bins', type=int, default=20 ** 3, help='resolution of main cube [default: 10]')
parser.add_argument('--train', type=int, default=1, help='1 if training, 0 otherwise [default: 1]')
parser.add_argument('--eval', type=int, default=1, help='1 if evaluating, 0 otherwise [default:0]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
parser.add_argument('--object_id', default='03001627', help='object id = sub folder name [default: 03001627 (chair)]')
parser.add_argument('--threshold', default=0.01, help='cube probability threshold')
args = parser.parse_args()

# Model Life-Cycle
##################

dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

logging.info("Available device: " + str(dev))

# Prepare the Data
if args.train:
    train_dataset = ShapeDiffDataset(args.train_path, args.object_id)
    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True)

if args.eval:
    val_dataset = ShapeDiffDataset(args.train_path, args.object_id, val=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, 1, shuffle=True)

criterion = VAELoss()


#writer = SummaryWriter(args.log_dir)

r = lambda: np.random.rand()


# Define the Model
def get_model():
    vae = VariationalAutoEncoder(num_cubes=args.bins, threshold=args.threshold, dev=dev).double().to(dev)
    return vae.to(dev), opt.Adam(vae.parameters(), lr=0.0001, betas=(0.9, 0.999))


def loss_batch(model, input, prob_target, x_diff_target, loss_func, opt=None, idx=1):
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

    if idx % 1000 == 1:
        logging.info("Finished " + str(idx) + " batches.")
        # test for vanishing gradient
        logging.info("sum of non changed centers: " + str(torch.sum(mu_out == model.last_mu)))

    model.last_mu = mu_out

    loss = loss_func(prob_pred, prob_target.reshape((prob_pred.shape[0], prob_target.shape[0])),
                     x_diff_pred, x_diff_target, mu_out, model.lower_bound, model.upper_bound)  # scalar

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

        losses, nums = zip(
            *[loss_batch(model, x.transpose(2, 1), h.flatten(), d, loss_func, op, i) for i, (x, h, e, d) in
              enumerate(train_dl)]
        )
        train_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        logging.info("Epoch : % 3d, Training error : % 5.5f" % (epoch, train_loss))

        model.eval()

        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, x.transpose(2, 1), h.flatten(), d, loss_func) for x, h, e, d in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        # print to console
        logging.info("Epoch : % 3d, Validation error : % 5.5f" % (epoch, val_loss))

        if val_loss < min_loss:
            min_loss = val_loss
            min_model = model

        # temporary save model
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

    logging.info("finish training.")
