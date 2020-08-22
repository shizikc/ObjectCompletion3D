import torch

def get_voxel_centers(dim=5):
    """
    Assumptions: volume space is [-1, 1] ^ 3, and voxels are a dim^3 array
    """
    voxel_size = 2 / dim
    ticks = torch.arange(-1,  1, voxel_size) + voxel_size / 2.
    xs, ys, zs = torch.meshgrid(*([ticks]*3))  # each is (20,20,20)
    centers = torch.stack([xs, ys, zs], -1).reshape(-1, 3)
    return centers
