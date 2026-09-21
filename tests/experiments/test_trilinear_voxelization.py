import torch

from vheat3d.modules.sparse_voxelization import SparseVoxelization


def make_vox(mode="trilinear", grid=8):
    return SparseVoxelization(grid_size=(grid, grid, grid), voxel_mode=mode,
                              sigma=0.1, bounds_mode="batch")


def test_trilinear_output_shapes_and_range():
    torch.manual_seed(0)
    pts = torch.rand(2, 64, 3)
    vox = make_vox()
    field, mask = vox(pts)
    assert field.shape == (2, 1, 8, 8, 8)
    assert mask.shape == field.shape
    assert float(field.min()) >= 0.0
    assert float(field.max()) <= 1.0
    assert float(mask.min()) >= 0.0 and float(mask.max()) <= 1.0


def test_trilinear_mass_roughly_conserved():
    torch.manual_seed(0)
    pts = torch.rand(1, 64, 3)
    vox = make_vox()
    field, _ = vox(pts)
    # every point distributes unit mass across 8 neighbors (clamped at borders
    # and saturated at 1.0), so total mass <= N and reasonably close to it
    total = float(field.sum())
    assert 0.5 * 64 <= total <= 64.0


def test_trilinear_exact_grid_point_lands_on_voxel_center():
    grid = 8
    # a single point at an exact voxel center should put mass 1 on that voxel
    coord = torch.tensor([[[0.5, 0.5, 0.5]]])
    vox = SparseVoxelization(grid_size=(grid, grid, grid), voxel_mode="trilinear",
                             sigma=0.1, bounds_mode="batch")
    field, mask = vox(coord)
    assert float(field.max()) > 0.9
    assert int((field > 0).sum()) >= 1


def test_trilinear_deterministic():
    torch.manual_seed(1)
    pts = torch.rand(1, 32, 3)
    a, _ = make_vox()(pts)
    b, _ = make_vox()(pts)
    assert torch.allclose(a, b)


def test_gaussian_unchanged_by_trilinear_addition():
    torch.manual_seed(2)
    pts = torch.rand(1, 32, 3)
    g, _ = make_vox("gaussian")(pts)
    assert g.shape == (1, 1, 8, 8, 8)
    assert float(g.max()) <= 1.0
