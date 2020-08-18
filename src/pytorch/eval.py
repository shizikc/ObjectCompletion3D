# evaluate


import torch

from src.dataset.data_utils import plot_pc
from src.dataset.shapenet import ShapeDiffDataset
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


        # # cube indicator prediction
        # pred_ind = ((pred[0] - threshold) > 0)  # torch.Size([1000])
        # print("positive pred:", pred_ind.int().sum())

        # h_ind = (hist[0] > 0).flatten()  # torch.Size([1000])
        # d = h_ind.int().sum()
        # print("positive gt:", d)
        #
        # ## Accuracy mesurment
        # cond1 = torch.tensor(h_ind == True)
        # cond2 = torch.tensor(pred_ind == h_ind)
        # acc_ind = cond1 & cond2  # torch.Size([1000])
        #
        # logging.info("Indicator True Positive % f", acc_ind.float().sum() / d)
        # logging.info("Indicator Accuracy % f", acc_ind.float().sum())
        #
        # # continues uniform distribution
        # err = torch.abs(pred - hist[0].flatten())  # torch.Size([1, 1000])
        #
        # logging.info("Continues Accuracy % f", 1 - err[0].mean())
        #
        # mesh = np.meshgrid(edges[0][0][0:args.bins], edges[0][1][0:args.bins], edges[0][2][0:args.bins])
        # h_ind = h_ind.reshape(args.bins, args.bins, args.bins)
        # pred_ind = pred_ind.reshape(args.bins, args.bins, args.bins)
        #
        # ax = set_fig(edges[0])
        #
        # # plot_mesh(ax, mesh[1][idx_h], mesh[0][idx_h], mesh[2][idx_h], ivl=ivl, col="red") #gt box
        #
        # plot_mesh(ax, mesh[0][pred_ind], mesh[1][pred_ind], mesh[2][pred_ind], ivl=ivl, col="red")  # pred box
        # ax.scatter(x_partial[0][:, 0], x_partial[0][:, 1], x_partial[0][:, 2], s=4, color="grey")  # gt partial
        # ax.scatter(x_diff[0][:, 0], x_diff[0][:, 1], x_diff[0][:, 2], s=4, color="black")  # gt diff
        #
        # # ax.scatter(mesh[0][pred_ind], mesh[1][pred_ind], mesh[2][pred_ind], s=7, color="red")  # gt diff
        # # ax.scatter( mesh[1][h_ind], mesh[0][h_ind], mesh[2][h_ind], s=4, col or="red")  # gt diff
        break
