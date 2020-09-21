import argparse
import collections
import logging
from pathlib import Path
import torch.nn as nn

import torch
import pandas as pd
from datetime import datetime

from chamfer_distance.chamfer_distance import chamfer_distance_with_batch_v2
from dataset.shapeDiff import ShapeDiffDataset
from pytorch.vae import VariationalAutoEncoder


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--notes', default='', help='Experiments notes [default: log]')
parser.add_argument('--model_path',
                    default='C:/Users/sharon/Documents/Research/ObjectCompletion3D/model/')
# default='/home/coopers/models/')
parser.add_argument('--train_path',
                    default='C:\\Users\\sharon\\Documents\\Research\\data\\dataset2019\\shapenet\\train\\gt\\')
# default='/home/coopers/data/train/gt/')
parser.add_argument('--max_epoch', type=int, default=1000, help='Epoch to run [default: 100]')
parser.add_argument('--bins', type=int, default=5, help='resolution of main cube [default: 10]')
parser.add_argument('--voxel_sample', type=int, default=20, help='number of samples per voxel [default: 20]')
parser.add_argument('--train', type=int, default=1, help='1 if training, 0 otherwise [default: 1]')
parser.add_argument('--eval', type=int, default=1, help='1 if evaluating, 0 otherwise [default:0]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--object_id', default='04256520', help='object id = sub folder name [default: 03001627 (chair)]')
parser.add_argument('--threshold', default=0.01, help='cube probability threshold')
parser.add_argument('--lr', default=0.01, help='cube probability threshold')
parser.add_argument('--momentum', default=0.09, help='cube probability threshold')
parser.add_argument('--cf_coeff', default=1)
parser.add_argument('--cfc_coeff', default=1)
parser.add_argument('--bce_coeff', default=1)
parser.add_argument('--reg_start_iter', type=int, default=150)

args = parser.parse_args()

# Model Life-Cycle
##################
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logging.info("Device: " + str(dev))

# writer = SummaryWriter(args.log_dir)

loss_capture = collections.defaultdict(list)
bce_loss = nn.BCELoss(reduction='mean')

#########################
# TRACKING and PARAMETERS
#########################

run_id = "{:%m%d_%H%M}".format(datetime.now())
bins = args.bins
threshold = args.threshold
object_id = args.object_id
model_path = args.model_path + "model_" + str(run_id) + ".pt"
train_path = Path(args.train_path, object_id)
val_path = Path(args.train_path.replace('train', 'val'), object_id)
cd_coeff = args.cf_coeff
cdc_coeff = args.cfc_coeff
bce_coeff = args.bce_coeff
batch_size = args.batch_size
learning_rate = args.lr
momentum = args.momentum
voxel_sample = args.voxel_sample
notes = args.notes


def update_tracking(
        id, field, value, csv_file="./tracking.csv",
        integer=False, digits=None, nround=6,
        drop_broken_runs=False, field_mark_done="finish_time"):
    """
    Tracking function for keep track of model parameters and
    CV scores. `integer` forces the value to be an int.
    """
    try:
        df = pd.read_csv(csv_file, index_col=[0])
    except FileNotFoundError:
        df = pd.DataFrame()
    if drop_broken_runs:
        try:
            df = df.dropna(subset=[field_mark_done])
        except KeyError:
            logging.warning("No loss column found  in tracking file")
    if integer:
        value = round(value)
    elif digits is not None:
        value = round(value, digits)
    df.loc[id, field] = value  # Model number is index
    df = df.round(nround)
    df.to_csv(csv_file)


#############
# TRAIN UTILS
#############

def get_model():
    vae = VariationalAutoEncoder(n_bins=bins, dev=dev, voxel_sample=voxel_sample, threshold=threshold)

    return vae.to(dev), torch.optim.SGD(vae.parameters(), lr=learning_rate, momentum=momentum)


def loss_batch(mdl, input, prob_target, x_diff_target, opt=None, idx=1):
    """

    :param idx:
    :param mdl:
    :param input:
    :param prob_target:
    :param x_diff_target:
    :param opt:
    :return:
    """
    train_reg = idx >= args.reg_start_iter
    diff_pred, probs_pred = mdl(input, pred_pc=train_reg)

    pred_loss = bce_loss(probs_pred, prob_target)
    acc = ((probs_pred > 0.5) == prob_target).float().mean()
    if train_reg:
        if diff_pred.shape[1] == 0:
            logging.info("Found partial with no positive probability cubes: " + str(diff_pred.shape))
            CD = torch.tensor(0.)
        else:
            CD = chamfer_distance_with_batch_v2(diff_pred.reshape(diff_pred.shape[0], -1, 3),
                                                x_diff_target, method="max")
            # penalty for centers in objects' missing parts
            CD2 = chamfer_distance_with_batch_v2(mdl.centers.reshape(diff_pred.shape[0], -1, 3),
                                                 x_diff_target, method="mean")
        c_loss = CD + CD2
    else:
        c_loss = torch.tensor(0.)

    total_loss = args.bce_coeff * pred_loss + cd_coeff * c_loss

    if opt is not None:
        # training
        total_loss.backward()
        opt.step()
        opt.zero_grad()

    d = {'total_loss': total_loss,
         'pred_loss': pred_loss,
         'c_loss': c_loss,
         'acc': acc}
    return {k: v.item() for k, v in d.items()}


def fit(epochs, model, op):
    # x, d, h = next(iter(train_loader))
    for epoch in range(epochs):
        metrics = collections.defaultdict(lambda: 0.)
        model.train()
        for x, d, h in train_loader:
            tmp_metrics = loss_batch(mdl=model,
                                     input=x.transpose(2, 1),
                                     prob_target=h.flatten(),
                                     x_diff_target=d,
                                     opt=op,
                                     idx=epoch)

            metrics = {k: metrics[k] + tmp_metrics[k] for k in tmp_metrics.keys()}

        metrics = {k: metrics[k] / len(train_dataset) for k in metrics.keys()}
        metrics["epoch"] = epoch
        logging.info(
            "Epoch : %(epoch)3d, (Train) total loss : %(total_loss)5.4f, pred_loss: %(pred_loss).4f, c_loss: %("
            "c_loss).3f accuracy : %(acc).4f" % metrics)

        # model.eval()
        # metrics = collections.defaultdict(lambda: 0.)
        # for x, d, h in val_loader:
        #     with torch.no_grad():
        #         tmp_metrics = loss_batch(mdl=model,
        #                                  input=x.transpose(2, 1),
        #                                  prob_target=h.flatten(),
        #                                  x_diff_target=d,
        #                                  idx=epoch)
        #         metrics = {k: metrics[k] + tmp_metrics[k] for k in tmp_metrics.keys()}
        # metrics = {k: metrics[k] / len(train_dataset) for k in metrics.keys()}
        # metrics["epoch"] = epoch
        #
        # logging.info("Epoch : %(epoch)3d, (Val) total loss : %(total_loss)5.4f, pred_loss: %(pred_loss).4f, c_loss: "
        #              "%(""c_loss).3f accuracy : %(acc).4f" % metrics)

        if epoch == args.reg_start_iter:
            min_loss = metrics['total_loss']

        if epoch >= args.reg_start_iter and metrics['total_loss'] <= min_loss:
            min_loss = metrics['total_loss']

            # save minimum model
            torch.save(model.state_dict(), model_path)
    # return x, d, h
    # update matching to model loss
    metrics['total_loss'] = min_loss
    return metrics


# Prepare the Data
if args.train:
    train_dataset = ShapeDiffDataset(train_path, bins, dev, seed=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True)

if args.eval:
    val_dataset = ShapeDiffDataset(val_path, bins, dev)
    val_loader = torch.utils.data.DataLoader(val_dataset, 1, shuffle=True)

if __name__ == '__main__':
    update_tracking(run_id, "bins", bins)
    update_tracking(run_id, "threshold", threshold)
    update_tracking(run_id, "object_id", object_id)
    update_tracking(run_id, "cd_coeff", cd_coeff)
    update_tracking(run_id, "cdc_coeff", cdc_coeff)
    update_tracking(run_id, "bce_coeff", bce_coeff)
    update_tracking(run_id, "batch_size", batch_size)
    update_tracking(run_id, "learning_rate", learning_rate)
    update_tracking(run_id, "momentum", momentum)
    update_tracking(run_id, "voxel_sample", voxel_sample)
    update_tracking(run_id, "note", notes)

    # if args.train:
    # run model
    model, opt = get_model()

    # train model
    metric = fit(args.max_epoch, model, opt)
    # plot centers
    # pred = model(x.transpose(1, 2), pred_pc=True)
    # plot_pc_mayavi([pred[0].detach().numpy(), x], colors=((1., 1., 1.), (0., 0., 1.)))
    # plot_pc([d[0].cpu(), x[0].cpu()],
    #         colors=["red", "blue", "black"])

    update_tracking(run_id, "total_loss", metric["total_loss"])
    update_tracking(run_id, "pred_loss", metric["pred_loss"])
    update_tracking(run_id, "finish_time", "{:%m%d_%H%M}".format(datetime.now()))

    logging.info("finish training.")
