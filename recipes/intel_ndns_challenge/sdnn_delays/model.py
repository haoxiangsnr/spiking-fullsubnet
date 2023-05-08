import h5py
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from lava.lib.dl import slayer  # type: ignore


class Model(torch.nn.Module):
    def __init__(
        self, threshold=0.1, tau_grad=0.1, scale_grad=0.8, max_delay=64, out_delay=0
    ):
        super().__init__()
        self.stft_mean = 0.2
        self.stft_var = 1.5
        self.stft_max = 140
        self.out_delay = out_delay

        sigma_params = {  # sigma-delta neuron parameters
            "threshold": threshold,  # delta unit threshold
            "tau_grad": tau_grad,  # delta unit surrogate gradient relaxation parameter
            "scale_grad": scale_grad,  # delta unit surrogate gradient scale parameter
            "requires_grad": False,  # trainable threshold
            "shared_param": True,  # layer wise threshold
        }
        sdnn_params = {
            **sigma_params,
            "activation": F.relu,  # activation function
        }

        self.input_quantizer = lambda x: slayer.utils.quantize(x, step=1 / 64)

        self.blocks = torch.nn.ModuleList(
            [
                slayer.block.sigma_delta.Input(sdnn_params),
                slayer.block.sigma_delta.Dense(
                    sdnn_params,
                    257,
                    512,
                    weight_norm=False,
                    delay=True,
                    delay_shift=True,
                ),
                slayer.block.sigma_delta.Dense(
                    sdnn_params,
                    512,
                    512,
                    weight_norm=False,
                    delay=True,
                    delay_shift=True,
                ),
                slayer.block.sigma_delta.Output(
                    sdnn_params, 512, 257, weight_norm=False
                ),
            ]
        )

        self.blocks[0].pre_hook_fx = self.input_quantizer

        self.blocks[1].delay.max_delay = max_delay
        self.blocks[2].delay.max_delay = max_delay

    def forward(self, noisy):
        x = noisy - self.stft_mean

        for block in self.blocks:
            x = block(x)

        mask = torch.relu(x + 1)
        return slayer.axon.delay(noisy, self.out_delay) * mask

    def grad_flow(self, path):
        # helps monitor the gradient flow
        grad = [b.synapse.grad_norm for b in self.blocks if hasattr(b, "synapse")]

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + "gradFlow.png")
        plt.close()

        return grad

    def validate_gradients(self):
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (
                    torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
                )
                if not valid_gradients:
                    break
        if not valid_gradients:
            self.zero_grad()

    def export_hdf5(self, filename):
        # network export to hdf5 format
        h = h5py.File(filename, "w")
        layer = h.create_group("layer")
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f"{i}"))
