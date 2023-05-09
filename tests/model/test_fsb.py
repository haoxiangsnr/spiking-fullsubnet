import torch

from audiozen.models.fsb.model import FrequencyCommunication, Model


def test_freqency_communication():
    batch_size, num_channels, num_freqs, num_frames, embed_size, num_bands = (
        2,
        2,
        256,
        64,
        32,
        34,
    )
    input = torch.rand(batch_size, num_channels, embed_size, num_frames, num_bands)
    model = FrequencyCommunication(
        sb_num_center_freqs=[2, 8, 16, 32],
        freq_cutoffs=[0, 32, 128, 192, 256],
        freq_communication_hidden_size=64,
        embed_size=32,
    )
    output = model(input)
    assert output.shape == (batch_size, num_channels, num_freqs, num_frames)


def test_fsb_block():
    batch_size, num_channels, num_freqs, num_frames, embed_size, num_bands = (
        2,
        2,
        256,
        64,
        32,
        34,
    )
    input = torch.rand(batch_size, num_channels, num_freqs, num_frames)

    model = Model()
    output = model(input)
