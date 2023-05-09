import torch
import torch.nn as nn

from audiozen.models.module.custom_lstm import script_lnlstm
from audiozen.models.module.groupGRU import SharedGroupGRU
from audiozen.models.module.mogrifier_lstm import MogrifierLSTMWrapper


class SequenceModel(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        num_layers,
        bidirectional,
        sequence_model="GRU",
        output_activate_function="Tanh",
        num_groups=4,
        mogrify_steps=5,
        dropout=0.0,
    ):
        """
        Wrapper of conventional sequence models

        Args:
            input_size: input size
            output_size: when projection_size> 0, the linear layer is used for projection. Otherwise no linear layer.
            hidden_size: hidden size
            num_layers:  num layers
            bidirectional: if bidirectional
            sequence_model: LSTM, GRU, SharedGroupGRU, LayerNormLSTM
            output_activate_function: Tanh, ReLU, ReLU6, LeakyReLU, PReLU
        """
        super().__init__()
        if sequence_model == "LSTM":
            self.sequence_model = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout,
            )
        elif sequence_model == "GRU":
            self.sequence_model = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout,
            )
        elif sequence_model == "SharedGroupGRU":
            self.sequence_model = SharedGroupGRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_groups=num_groups,
                bidirectional=bidirectional,
            )
        elif sequence_model == "LayerNormLSTM":
            self.sequence_model = script_lnlstm(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
            )
        elif sequence_model == "MogrifierLSTM":
            assert (
                bidirectional is False
            ), "MogrifierLSTM does not support bidirectional"
            self.sequence_model = MogrifierLSTMWrapper(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                mogrify_steps=mogrify_steps,
            )
        else:
            raise NotImplementedError(f"Not implemented {sequence_model}")

        # Fully connected layer
        if int(output_size):
            if bidirectional:
                self.fc_output_layer = nn.Linear(hidden_size * 2, output_size)
            else:
                self.fc_output_layer = nn.Linear(hidden_size, output_size)

        # Activation function layer
        if output_activate_function:
            if output_activate_function == "Tanh":
                self.activate_function = nn.Tanh()
            elif output_activate_function == "ReLU":
                self.activate_function = nn.ReLU()
            elif output_activate_function == "ReLU6":
                self.activate_function = nn.ReLU6()
            elif output_activate_function == "LeakyReLU":
                self.activate_function = nn.LeakyReLU()
            elif output_activate_function == "PReLU":
                self.activate_function = nn.PReLU()
            else:
                raise NotImplementedError(
                    f"Not implemented activation function {self.activate_function}"
                )

        self.output_activate_function = output_activate_function
        self.output_size = output_size

        # only for custom_lstm and are needed to clean up
        self.sequence_model_name = sequence_model
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x):
        """
        Args:
            x: [B, F, T]
        Returns:
            [B, F, T]
        """
        n_dim = x.dim()
        assert n_dim in (3, 4), f"Shape is {x.shape}."

        if n_dim == 4:
            batch_size, num_channels, _, num_frames = x.shape
            x = x.reshape(batch_size, -1, num_frames)

        batch_size, _, _ = x.shape

        if self.sequence_model_name == "LayerNormLSTM":
            states = [
                (
                    torch.zeros(batch_size, self.hidden_size, device=x.device),
                    torch.zeros(batch_size, self.hidden_size, device=x.device),
                )
                for _ in range(self.num_layers)
            ]
        else:
            states = None

        x = x.permute(2, 0, 1).contiguous()  # [B, F, T] => [T, B, F]
        o, _ = self.sequence_model(x, states)  # [T, B, F] => [T, B, F]

        if self.output_size:
            o = self.fc_output_layer(o)  # [T, B, F] => [T, B, F]

        if self.output_activate_function:
            o = self.activate_function(o)

        o = o.permute(1, 2, 0).contiguous()  # [T, B, F] => [B, F, T]

        if n_dim == 4:
            o = o.reshape(batch_size, num_channels, -1, num_frames)

        return o


def _print_networks(nets: list):
    print(f"This project contains {len(nets)} networks, the number of the parameters: ")
    params_of_all_networks = 0
    for i, net in enumerate(nets, start=1):
        params_of_network = 0
        for param in net.parameters():
            params_of_network += param.numel()

        print(f"\tNetwork {i}: {params_of_network / 1e6} million.")
        params_of_all_networks += params_of_network

    print(
        f"The amount of parameters in the project is {params_of_all_networks / 1e6} million."
    )


if __name__ == "__main__":
    # with torch.no_grad():

    #
    #     ipt = torch.rand(batch, input_size, seq_len)
    #     # states = [LSTMState(torch.zeros(batch, hidden_size),
    #     #                     torch.zeros(batch, hidden_size))
    #     #           for _ in range(num_layers)]
    #
    #     model = SequenceModel(
    #         input_size=input_size,
    #         output_size=2,
    #         hidden_size=512,
    #         bidirectional=False,
    #         num_layers=3,
    #         sequence_model="LayerNormLSTM"
    #     )
    #
    #     start = datetime.datetime.now()
    #     opt = model(ipt)
    #     end = datetime.datetime.now()
    #     print(f"{end - start}")
    #     _print_networks([model, ])

    batch_size = 2
    hidden_size = 512
    output_size = 258
    num_layers = 3
    seq_len = 1000
    input_size = 257
    ipt = torch.rand(batch_size, input_size, seq_len)
    ipt = ipt.to("cuda:0")

    model = SequenceModel(
        input_size=input_size,
        output_size=output_size,
        hidden_size=hidden_size,
        bidirectional=False,
        num_layers=num_layers,
        sequence_model="LSTM",
    )

    # model.to("cuda:0")
    #
    # opt = model(ipt)
    # print(opt.shape)
