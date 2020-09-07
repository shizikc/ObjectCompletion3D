import argparse
import collections
import logging
from pathlib import Path
import torch.nn as nn

import torch
import torch.optim as opt
# from torch.utils.tensorboard import SummaryWriter
from src.chamfer_distance.chamfer_distance import chamfer_distance_with_batch_v2
from src.dataset.shapeDiff import ShapeDiffDataset
from src.pytorch.vae import VariationalAutoEncoder

# from src.pytorch.visualization import plot_pc_mayavi

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--model_path',
                    default='C:/Users/sharon/Documents/Research/ObjectCompletion3D/model/')
                    # default='/home/coopers/models/')
parser.add_argument('--train_path',
                    default='C:\\Users\\sharon\\Documents\\Research\\data\\dataset2019\\shapenet\\train\\gt\\')
                    # default='/home/coopers/data/train/gt/')
parser.add_argument('--max_epoch', type=int, default=3000, help='Epoch to run [default: 100]')
parser.add_argument('--bins', type=int, default=15, help='resolution of main cube [default: 10]')
parser.add_argument('--train', type=int, default=1, help='1 if training, 0 otherwise [default: 1]')
parser.add_argument('--eval', type=int, default=1, help='1 if evaluating, 0 otherwise [default:0]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--object_id', default='04256520', help='object id = sub folder name [default: 03001627 (chair)]')
parser.add_argument('--regular_method', default='abs')
parser.add_argument('--threshold', default=0.01, help='cube probability threshold')
parser.add_argument('--cf_coeff', default=1)
parser.add_argument('--bce_coeff', default=1)
parser.add_argument('--reg_start_iter', default=100)
args = parser.parse_args(["@args.txt"])

# Model Life-Cycle
##################
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logging.info("Device: " + str(dev))

bins = args.bins
threshold = args.threshold
object_id = args.object_id
model_path = args.model_path + "model_" + str(object_id) + "_" + str(threshold) + ".pt"
train_path = Path(args.train_path, object_id)
val_path = Path(args.train_path.replace('train', 'val'), object_id)
cd_coeff = args.cf_coeff
bce_coeff = args.bce_coeff
regular_method = args.regular_method
batch_size = args.batch_size

# Prepare the Data
if args.train:
    train_dataset = ShapeDiffDataset(train_path, bins, dev, seed=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True)

if args.eval:
    val_dataset = ShapeDiffDataset(val_path, bins, dev)
    val_loader = torch.utils.data.DataLoader(val_dataset, 1, shuffle=True)

# writer = SummaryWriter(args.log_dir)

loss_capture = collections.defaultdict(list)
bce_loss = nn.BCELoss(reduction='mean')


# Define the Model
def get_model():
    vae = VariationalAutoEncoder(n_bins=bins, dev=dev, voxel_sample=20,
                                 threshold=threshold, regular_method=regular_method)

    # return vae.to(dev), torch.optim.Adam(vae.parameters(), lr=0.001, betas=(0.9, 0.999))
    return vae.to(dev), torch.optim.SGD(vae.parameters(), lr=0.001, momentum=0.9)


def loss_batch(mdl, input, prob_target, x_diff_target, opt=None, idx=1):
    """

    :param idx:
    :param mdl:
    :param input:
    :param prob_pred:
    :param prob_target:
    :param x_diff_pred:
    :param x_diff_target:
    :param opt:
    :return:
    """
    train_reg = idx >= args.reg_start_iter
    diff_pred, probs_pred = mdl(input, pred_pc=train_reg)

    pred_loss = bce_loss(probs_pred, prob_target)
    acc = ((probs_pred > 0.5) == prob_target).float().mean()
    if idx == 500:
        pass
    if train_reg:
        if diff_pred.shape[1] == 0:
            logging.info("Found partial with no positive probability cubes: " + str(diff_pred.shape))
            CD = torch.tensor(0.)
        else:
            CD = chamfer_distance_with_batch_v2(diff_pred.reshape(diff_pred.shape[0], -1, 3), x_diff_target)
        c_loss = CD
    else:
        c_loss = torch.tensor(0.)

    total_loss = args.bce_coeff * pred_loss + c_loss

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
    x, d, h = next(iter(train_loader))
    for epoch in range(epochs):
        metrics = collections.defaultdict(lambda: 0.)
        model.train()
        # for x, d, h in train_loader:
        metrics = loss_batch(mdl=model,
                                 input=x.transpose(2, 1),
                                 prob_target=h.flatten(),
                                 x_diff_target=d,
                                 opt=op,
                                 idx=epoch)

            # metrics = {k: metrics[k] + tmp_metrics[k] for k in tmp_metrics.keys()}

        # metrics = {k: metrics[k] / len(train_dataset) for k in metrics.keys()}
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

        if epoch == 0:
            min_loss = metrics['total_loss']

        if metrics['total_loss'] <= min_loss:
            min_loss = metrics['total_loss']

            # temporary save model
            torch.save(model.state_dict(), model_path)


if __name__ == '__main__':

    if args.train:
        # run model
        model, opt = get_model()

        # train model
        fit(args.max_epoch, model, opt)
        # plot centers
        # pred = model(x.transpose(1, 2), pred_pc=True)
        # plot_pc_mayavi([pred[0].detach().numpy(), x], colors=((1., 1., 1.), (0., 0., 1.)))

    logging.info("finish training.")
