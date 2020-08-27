import argparse
import logging
from pathlib import Path
import torch.nn as nn
import ops
import torch
import torch.optim as opt
# from torch.utils.tensorboard import SummaryWriter
from src.pytorch.visualization import plot_pc_mayavi
from src.chamfer_distance.chamfer_distance import chamfer_distance_with_batch
from src.dataset.shapeDiff import ShapeDiffDataset
from src.pytorch.vae import VariationalAutoEncoder

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--model_path',
                    default='C:/Users/sharon/Documents/Research/ObjectCompletion3D/model/')
                    # default='/home/coopers/models/')
                    # default='./models/')
parser.add_argument('--train_path',
                    default='C:\\Users\\sharon\\Documents\\Research\\data\\dataset2019\\shapenet\\train\\gt\\')
                    # default='/home/yonatan/data/oc3d/chair/train/gt/')
parser.add_argument('--max_epoch', type=int, default=500, help='Epoch to run [default: 100]')
parser.add_argument('--bins', type=int, default=5, help='resolution of main cube [default: 10]')
parser.add_argument('--train', type=int, default=1, help='1 if training, 0 otherwise [default: 1]')
parser.add_argument('--eval', type=int, default=1, help='1 if evaluating, 0 otherwise [default:0]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--object_id', default='03001627', help='object id = sub folder name [default: 03001627 (chair)]')
parser.add_argument('--regular_method', default='abs')
parser.add_argument('--threshold', default=0.01, help='cube probability threshold')
parser.add_argument('--cf_coeff', default=1)
parser.add_argument('--bce_coeff', default=100)
parser.add_argument('--rc_coeff', default=0.01)
#additional args read from file - put an empty 'args.txt' file in working directory to run without errors
args = parser.parse_args(['@args.txt'])

# Model Life-Cycle
##################
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logging.info("Device: " + str(dev))

bins = args.bins
threshold = args.threshold
object_id = args.object_id
model_path = args.model_path + "model_" + str(object_id) + "_" + str(threshold) + ".pt"
train_path = Path(args.train_path, object_id)
val_path = args.train_path.replace('train', 'val')
cf_coeff = args.cf_coeff
rc_coeff = args.rc_coeff
bce_coeff = args.bce_coeff
regular_method = args.regular_method

# Prepare the Data
if args.train:
    train_dataset = ShapeDiffDataset(train_path, bins, dev)
    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True)

if args.eval:
    val_dataset = ShapeDiffDataset(val_path, bins, dev)
    val_loader = torch.utils.data.DataLoader(val_dataset, 1, shuffle=True)


# writer = SummaryWriter(args.log_dir)


# Define the Model
def get_model():
    vae = VariationalAutoEncoder(n_bins=bins, dev=dev, voxel_sample=20, cf_coeff=cf_coeff,
                                 threshold=threshold, rc_coeff=rc_coeff, bce_coeff=bce_coeff,
                                 regular_method=regular_method)

    return vae.to(dev), opt.Adam(vae.parameters(), lr=0.0001, betas=(0.9, 0.999))


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

    x_diff_pred, voxel_scores = mdl(input, x_diff_target, prob_target)
    chamfer_loss = chamfer_distance_with_batch(x_diff_pred, x_diff_target.transpose(2,1), 0)
    # pred_loss = torch.nn.BCELoss(voxel_scores, prob_target)
    pred_loss = 0
    if idx % 50 == 1:
        logging.info("Finished " + str(idx) + " batches. chamfer loss: %.4f, pred_loss: %.4f" %(chamfer_loss, pred_loss))

    total_loss = chamfer_loss  # + pred_loss

    # losses = ops.get_module_losses(model)
    # loss = sum(losses.values())


    if opt is not None:
        # training
        opt.zero_grad()
        total_loss.backward()
        opt.step()


    return total_loss


def fit(epochs, model, op):
    x, d, h = next(iter(train_loader))

    for epoch in range(epochs):
        model.train()
        loss = loss_batch(mdl=model,
                          input=x.transpose(2, 1),
                          prob_target=h.flatten(),
                          x_diff_target=d.transpose(2, 1),
                          opt=op,
                          idx=epochs)

        # losses, nums = zip(
        #     *[loss_batch(mdl=model, input=x.transpose(2, 1), prob_target=h.flatten(), x_diff_target=d, opt=op, idx=i)
        #       for i, (x, d, h) in enumerate(train_loader)]
        # )
        # print("**" + str(model.mu))
        # train_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        logging.info("Epoch : % 3d, Training error : % 5.5f" % (epoch, loss))

        # model.eval()
        # with torch.no_grad():
        #     losses, nums = zip(
        #         *[loss_batch(mdl=model, input=x.transpose(2, 1), prob_target=h.flatten(), x_diff_target=d, idx=i)
        #           for i, (x, d, h) in enumerate(val_loader)]
        #     )
        #
        # val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        # logging.info("Epoch : % 3d, Validation error : % 5.5f" % (epoch, val_loss))
        #
        if epoch == 0:
            min_loss = loss

        ## commented cause not worried about checkpoints during model debugging
        # if loss <= min_loss:
        #     min_loss = loss
        #     # temporary save model
        #     torch.save(model.state_dict(), model_path)


def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        torch.nn.init.zeros_(m.weight)
        m.bias.data.fill_(0)


if __name__ == '__main__':

    if args.train:
        # run model
        model, opt = get_model()
        model.apply(init_weights)
        # train model
        fit(args.max_epoch, model, opt)
        # plot_pc_mayavi([model.mu[0].view(model.voxel_centers.shape).detach().numpy(),
        #                 model.voxel_centers.detach().numpy()],
        #                colors=((1., 1., 1.), (0., 0., 1.)))

        model.eval()
        loss_batch(mdl=model, input=x.transpose(2, 1), prob_target=h.flatten(), x_diff_target=d, opt=op,
                   idx=epochs)
    logging.info("finish training.")
