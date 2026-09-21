import torch

from vheat3d.modules.sparse_voxelization import SparseVoxelization


def test_default_is_batch_mode():
    v = SparseVoxelization(grid_size=(8, 8, 8), voxel_mode="gaussian")
    assert v.bounds_mode == "batch"


def test_per_sample_bounds_match_single_sample_batches():
    torch.manual_seed(0)
    x = torch.rand(2, 128, 3) * 2 - 1
    v_ps = SparseVoxelization(grid_size=(8, 8, 8), voxel_mode="gaussian",
                              bounds_mode="per_sample")
    v_b = SparseVoxelization(grid_size=(8, 8, 8), voxel_mode="gaussian")
    with torch.no_grad():
        f_ps, m_ps = v_ps(x)
        f0, m0 = v_b(x[0:1])
        f1, m1 = v_b(x[1:2])
    assert torch.allclose(f_ps[0], f0[0], atol=1e-6)
    assert torch.allclose(f_ps[1], f1[0], atol=1e-6)
    assert torch.equal(m_ps[0], m0[0])
    assert torch.equal(m_ps[1], m1[0])


def test_per_sample_is_composition_independent():
    torch.manual_seed(0)
    x1 = torch.rand(1, 128, 3) * 2 - 1
    x2 = x1 * 0.3 + 0.1  # different extent partner
    v = SparseVoxelization(grid_size=(8, 8, 8), voxel_mode="gaussian",
                           bounds_mode="per_sample")
    with torch.no_grad():
        alone, _ = v(x1)
        together, _ = v(torch.cat([x1, x2], 0))
    assert torch.allclose(alone[0], together[0], atol=1e-6)
