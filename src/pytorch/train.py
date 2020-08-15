import argparse
import logging

import torch
import torch.optim as opt
import numpy as np

# from torch.utils.tensorboard import SummaryWriter

from src.dataset.shapenet import ShapeDiffDataset
from src.pytorch.vae import VariationalAutoEncoder

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--model_path',
                    # default='C:/Users/sharon/Documents/Research/ObjectCompletion3D/model/model.pt')
                    default='models/model.pt')
parser.add_argument('--train_path',
                    # default='C:\\Users\\sharon\\Documents\\Research\\data\\dataset2019\\shapenet\\chair\\')
                    # default='C:\\Users\\sharon\\Documents\\Research\\data\\dataset2019\\shapenet\\chair\\')
                    default='/home/yonatan/data/oc3d/chair/')
parser.add_argument('--max_epoch', type=int, default=1, help='Epoch to run [default: 100]')
parser.add_argument('--bins', type=int, default=20 ** 3, help='resolution of main cube [default: 10]')
parser.add_argument('--train', type=int, default=1, help='1 if training, 0 otherwise [default: 1]')
parser.add_argument('--eval', type=int, default=1, help='1 if evaluating, 0 otherwise [default:0]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
parser.add_argument('--object_id', default='03001627', help='object id = sub folder name [default: 03001627 (chair)]')
parser.add_argument('--threshold', default=0.001, help='cube probability threshold')
args = parser.parse_args()

# Model Life-Cycle
##################
dev = torch.device("cuda")  if torch.cuda.is_available() else torch.device("cpu")
logging.info("Device: " + str(dev))

bins = args.bins
threshold = args.threshold
model_path = args.model_path
train_path = args.train_path
object_id = args.object_id

# Prepare the Data
if args.train:
    train_dataset = ShapeDiffDataset(train_path, object_id)
    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True)

if args.eval:
    val_dataset = ShapeDiffDataset(args.train_path, args.object_id, val=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, 1, shuffle=True)

# writer = SummaryWriter(args.log_dir)

r = lambda: np.random.rand()


# Define the Model
def get_model():
    vae = VariationalAutoEncoder(num_cubes=bins, dev=dev).double().to(dev)
    return vae.to(dev), opt.Adam(vae.parameters(), lr=0.0001, betas=(0.9, 0.999))


def loss_batch(model, input, prob_target, x_diff_target, opt=None, idx=1):
    """

    :param idx:
    :param model:
    :param input:
    :param prob_pred:
    :param prob_target:
    :param x_diff_pred:
    :param x_diff_target:
    :param opt:
    :return:
    """

    # loss_batch(model, x.transpose(2, 1), h.flatten(), d, op, i)

    # vae(in_data, gt_diff, gt_prob)
    x_diff_pred = model(input, x_diff_target, prob_target)

    if idx % 1000 == 1:
        logging.info("Finished " + str(idx) + " batches.")

    loss = sum([m.loss for m in model.modules() if hasattr(m, 'loss')])

    if opt is not None:
        # training
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), input.shape[0]


# Train the Model
def fit(epochs, model, op):

    min_loss = 10000000

    for epoch in range(epochs):

        model.train()

        losses, nums = zip(
            *[loss_batch(model, x.transpose(2, 1), h.flatten(), d, op, i) for i, (x, h, e, d) in
              enumerate(train_loader)]
        )

        train_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        logging.info("Epoch : % 3d, Training error : % 5.5f" % (epoch, train_loss))

        model.eval()

        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, x.transpose(2, 1), h.flatten(), x_diff_target=d, idx=i) for i, (x, h, e, d) in
                  enumerate(val_loader)]
            )

        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        logging.info("Epoch : % 3d, Validation error : % 5.5f" % (epoch, val_loss))

        if val_loss < min_loss:
            min_loss = val_loss
            # temporary save model
            torch.save(model.state_dict(), args.model_path)


if __name__ == '__main__':

    if args.train:
        # run model
        model, opt = get_model()

        # train model
        fit(args.max_epoch, model, opt)

    logging.info("finish training.")
