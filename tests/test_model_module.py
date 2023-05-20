import torch

from audiozen.models.module.res_rnn import GroupMLP, GroupResRNNLayer
from audiozen.models.module.tac import TransformAverageConcatenate


def test_tac_dim(random):
    input = torch.rand(2, 4, 64, 256)
    model = TransformAverageConcatenate(64, 256)
    output = model(input)
    assert output.shape == input.shape


def test_group_res_rnn_dim(random):
    input = torch.rand(2, 256, 256)
    model = GroupResRNNLayer(256, 256, num_groups=4)
    output = model(input)
    assert output.shape == torch.Size([2, 4, 256 // 4, 256])


def test_group_mlp_dim():
    input = torch.rand(2, 256, 256)
    model = GroupMLP(256, 256, 512, num_groups=4)
    output = model(input)
    assert output.shape == torch.Size([2, 4, 512 // 4, 256])
