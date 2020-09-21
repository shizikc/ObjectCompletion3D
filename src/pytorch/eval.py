# evaluate
import torch

# from src.pytorch.visualization import plot_pc_mayavi
from src.dataset.shapeDiff import ShapeDiffDataset
from src.train import get_model, bins, batch_size, train_path, dev, threshold
from src.pytorch.visualization import plot_pc_mayavi

model_path = "C:\\Users\\sharon\\Documents\\Research\\ObjectCompletion3D\\model\\model_0920_1856.pt"
model, _ = get_model()
model.load_state_dict(torch.load(model_path, map_location=dev))
model.eval()

train_dataset = ShapeDiffDataset(train_path, bins, dev=dev, seed=0)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=False)

if __name__ == '__main__':
    total_acc = 0.
    for i, (x, d, h) in enumerate(train_loader):
        if i == 10:
            break
        pred = model(x.transpose(2, 1), pred_pc=True)

        # plot_pc([d[0].cpu(), pred[0].view(-1, 3).detach().numpy()], colors=["black", "red"])
        plot_pc_mayavi([pred[0].view(-1, 3).detach().numpy(), d[0].cpu()],
                       colors=((0., 0., 0.), (.9, .9, .9), (1., 0., 0.)))
        # colors=["black", "red"])
        b = model.voxel_centers[pred[1] > threshold]
        plot_pc_mayavi([b, d[0]], colors=((0., 1., 0.), (1., 0., 0.)))

        # # uniform sample from bounding box
        b_gt = h.flatten() > 0.
        b_pred = pred[1] > threshold
        acc_ratio = (b_gt == b_pred).sum().float() / len(b_gt)
        total_acc += acc_ratio
        print(acc_ratio)

    total_acc /= (i + 1)

    # print(torch.where(b_pred))

    # c = model.voxel_centers[b_gt].detach().numpy()
    # t = model.voxel_centers[b_pred].detach().numpy()

    # t = model.centers.view(b_gt.shape[0], 3)[b_pred].detach().numpy()
    # t = model.centers.view(b_gt.shape[0], 3)[b_gt].detach().numpy()

    # plot_pc_mayavi([c, t, x_diff], colors=((1., 1., 1.), (0., 0., 1.), (1., 0., 0.))) #, pred[0].detach().numpy()

    # plot_pc_mayavi([pred[0].detach().numpy(), x_diff], colors=((1., 1., 1.), (0., 0., 1.)))
