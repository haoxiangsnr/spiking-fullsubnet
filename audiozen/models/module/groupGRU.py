import torch
import torch.nn as nn


class SharedGroupGRULayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        is_first_layer,
        num_groups: int,
        batch_first: bool = True,
        bias: bool = True,
        dropout: float = 0,
        bidirectional: bool = False,
    ):
        super().__init__()
        assert hidden_size % num_groups == 0, "Should be divided by input size"

        if not is_first_layer:
            assert input_size % num_groups == 0

        kwargs = {
            "bias": bias,
            "batch_first": batch_first,
            "dropout": dropout,
            "bidirectional": bidirectional,
        }

        self.is_first_layer = is_first_layer

        if self.is_first_layer:
            self.input_size = input_size
        else:
            self.input_size = input_size // num_groups

        self.hidden_size = hidden_size // num_groups
        self.out_size = hidden_size

        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.num_groups = num_groups
        self.batch_first = batch_first

        # Several small GRUs
        self.GRU_layers = nn.ModuleList(
            (
                nn.GRU(self.input_size, self.hidden_size, **kwargs)
                for _ in range(num_groups)
            )
        )

    def initialize_h0(
        self, batch_size: int = 1, device: torch.device = torch.device("cpu")
    ):
        return torch.zeros(
            self.num_groups * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device,
        )

    def forward(self, input, h0=None):
        """

        Args:
            input: [B, T, F] if batch_first else [T, B, I], B: batch_size, F: input_size
            h0: [G*D, B, H], where G: groups, D: num_directions, H: hidden_size

        Returns:

        """
        if h0 is None:
            dim0, dim1 = input.shape[:2]  # [B, T]
            batch_size = dim0 if self.batch_first else dim1
            h0 = self.initialize_h0(batch_size, device=input.device)

        outputs = []
        output_states = []

        # 实际上并没有并行的哈，循环执行四次，将大型矩阵运算变成了四个小矩阵矩阵运算后的加法
        for i, GRU_layer in enumerate(self.GRU_layers):
            o, s = GRU_layer(
                input
                if self.is_first_layer
                else input[..., i * self.input_size : (i + 1) * self.input_size],
                h0[i * self.num_directions : (i + 1) * self.num_directions].detach(),
            )
            outputs.append(o)
            output_states.append(s)

        output = torch.cat(outputs, dim=-1)
        hidden_state = torch.cat(output_states, dim=0)

        return output, hidden_state


class SharedGroupGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        num_groups: int = 4,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0,
        bidirectional: bool = False,
        shuffle: bool = True,
        add_outputs: bool = False,
    ):
        super().__init__()

        kwargs = {
            "num_groups": num_groups,
            "bias": bias,
            "batch_first": batch_first,
            "dropout": dropout,
            "bidirectional": bidirectional,
        }

        assert hidden_size % num_groups == 0, "Should be divided by input size"

        self.num_groups = num_groups
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.hidden_size = hidden_size // num_groups

        if num_groups == 1:
            shuffle = False  # Fully connected, no need to shuffle

        self.shuffle = shuffle
        self.add_outputs = add_outputs

        self.GRUs = nn.ModuleList()

        self.GRUs.append(  # First layer only
            SharedGroupGRULayer(input_size, hidden_size, True, **kwargs)
        )
        for _ in range(1, num_layers):  # Other layers
            self.GRUs.append(
                SharedGroupGRULayer(hidden_size, hidden_size, False, **kwargs)
            )

    def initialize_h0(
        self, batch_size: int, device: torch.device = torch.device("cpu")
    ):
        return torch.zeros(
            (
                self.num_layers * self.num_groups * self.num_directions,
                batch_size,
                self.hidden_size,
            ),
            device=device,
        )

    def forward(self, input, state=None):
        """

        Args:
            input: [B, T, F]
            state: [B, F]

        Returns:

        """
        dim0, dim1, _ = input.shape
        batch_size = dim0 if self.batch_first else dim1

        if state is None:
            state = self.initialize_h0(batch_size, input.device)

        output = torch.zeros(
            dim0,
            dim1,
            self.hidden_size * self.num_directions * self.num_groups,
            device=input.device,
        )

        output_states = []
        h = self.num_groups * self.num_directions

        for i, GRU in enumerate(self.GRUs):
            input, hidden_state = GRU(input, state[i * h : (i + 1) * h])
            output_states.append(hidden_state)

            if (
                self.shuffle and i < self.num_layers - 1
            ):  # TODO: Not the final layer, but WHY?
                input = (
                    input.view(dim0, dim1, -1, self.num_groups)
                    .transpose(2, 3)
                    .reshape(dim0, dim1, -1)
                )

            if self.add_outputs:
                output += input  # TODO: WHY?
            else:
                output = input

        output_state = torch.cat(output_states, dim=0)
        return output, output_state


class GroupGRULayer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_groups: int,
        batch_first: bool = True,
        bias: bool = True,
        dropout: float = 0,
        bidirectional: bool = False,
    ):
        super().__init__()
        assert (
            input_size % num_groups == 0 and hidden_size % num_groups == 0
        ), "Should be divided by input size"

        kwargs = {
            "bias": bias,
            "batch_first": batch_first,
            "dropout": dropout,
            "bidirectional": bidirectional,
        }

        self.input_size = input_size // num_groups
        self.hidden_size = hidden_size // num_groups
        self.out_size = hidden_size

        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.num_groups = num_groups
        self.batch_first = batch_first

        # Several small GRUs
        self.GRU_layers = nn.ModuleList(
            (
                nn.GRU(self.input_size, self.hidden_size, **kwargs)
                for _ in range(num_groups)
            )
        )

    def initialize_h0(
        self, batch_size: int = 1, device: torch.device = torch.device("cpu")
    ):
        return torch.zeros(
            self.num_groups * self.num_directions,
            batch_size,
            self.hidden_size,
            device=device,
        )

    def forward(self, input, h0=None):
        """

        Args:
            input: [B, T, F] if batch_first else [T, B, I], B: batch_size, F: input_size
            h0: [G*D, B, H], where G: groups, D: num_directions, H: hidden_size

        Returns:

        """
        if h0 is None:
            dim0, dim1 = input.shape[:2]  # [B, T]
            batch_size = dim0 if self.batch_first else dim1
            h0 = self.initialize_h0(batch_size, device=input.device)

        outputs = []
        output_states = []

        # 实际上并没有并行的哈，循环执行四次，将大型矩阵运算变成了四个小矩阵矩阵运算后的加法
        for i, GRU_layer in enumerate(self.GRU_layers):
            o, s = GRU_layer(
                input[
                    ..., i * self.input_size : (i + 1) * self.input_size
                ],  # iterative sub_input
                h0[i * self.num_directions : (i + 1) * self.num_directions].detach(),
            )
            outputs.append(o)
            output_states.append(s)

        output = torch.cat(outputs, dim=-1)
        h = torch.cat(output_states, dim=0)

        return output, h


class GroupGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        num_groups: int = 4,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = 0,
        bidirectional: bool = False,
        shuffle: bool = True,
        add_outputs: bool = False,
    ):
        super().__init__()

        kwargs = {
            "num_groups": num_groups,
            "bias": bias,
            "batch_first": batch_first,
            "dropout": dropout,
            "bidirectional": bidirectional,
        }

        assert (
            input_size % num_groups == 0 and hidden_size % num_groups == 0
        ), "Should be divided by input size"

        self.num_groups = num_groups
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.hidden_size = hidden_size // num_groups

        if num_groups == 1:
            shuffle = False  # Fully connected, no need to shuffle

        self.shuffle = shuffle
        self.add_outputs = add_outputs

        self.GRUs = nn.ModuleList()

        self.GRUs.append(  # First layer only
            GroupGRULayer(input_size, hidden_size, **kwargs)
        )
        for _ in range(1, num_layers):  # Other layers
            self.GRUs.append(GroupGRULayer(hidden_size, hidden_size, **kwargs))

    def initialize_h0(
        self, batch_size: int, device: torch.device = torch.device("cpu")
    ):
        return torch.zeros(
            (
                self.num_layers * self.num_groups * self.num_directions,
                batch_size,
                self.hidden_size,
            ),
            device=device,
        )

    def forward(self, input, state=None):
        """

        Args:
            input: [B, T, F]
            state: [B, F]

        Returns:

        """
        dim0, dim1, _ = input.shape
        batch_size = dim0 if self.batch_first else dim1

        if state is None:
            state = self.initialize_h0(batch_size, input.device)

        output = torch.zeros(
            dim0,
            dim1,
            self.hidden_size * self.num_directions * self.num_groups,
            device=input.device,
        )

        output_states = []
        h = self.num_groups * self.num_directions

        for i, GRU in enumerate(self.GRUs):
            input, hidden_state = GRU(input, state[i * h : (i + 1) * h])
            output_states.append(hidden_state)

            if (
                self.shuffle and i < self.num_layers - 1
            ):  # TODO: Not the final layer, but WHY?
                input = (
                    input.view(dim0, dim1, -1, self.num_groups)
                    .transpose(2, 3)
                    .reshape(dim0, dim1, -1)
                )

            if self.add_outputs:
                output += input  # TODO: WHY?
            else:
                output = input

        output_state = torch.cat(output_states, dim=0)
        return output, output_state


class GroupedLinear(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_groups: int = 1,
        shuffle: bool = True,
    ):
        super().__init__()
        assert (
            input_size % num_groups == 0 and hidden_size % num_groups == 0
        ), "Should be divided by input size"

        self.groups = num_groups

        self.input_size = input_size // num_groups
        self.hidden_size = hidden_size // num_groups

        if num_groups == 1:
            shuffle = False

        self.shuffle = shuffle
        self.layers = nn.ModuleList(
            nn.Linear(self.input_size, self.hidden_size) for _ in range(num_groups)
        )

    def forward(self, x):
        outputs = []

        for i, layer in enumerate(self.layers):
            outputs.append(
                layer(x[..., i * self.input_size : (i + 1) * self.input_size])
            )

        output = torch.cat(outputs, dim=-1)

        if self.shuffle:
            orig_shape = output.shape
            output = (
                output.view(-1, self.hidden_size, self.groups)
                .transpose(-1, -2)
                .reshape(orig_shape)
            )

        return output


if __name__ == "__main__":
    g = 10  # groups
    h = 400  # hidden_size
    i = 33  # input_size
    b = 32  # batch_size
    t = 100  # time_steps
    m = SharedGroupGRULayer(i, h, True, g, batch_first=True)
    input = torch.randn((b, t, i))
    h0 = m.initialize_h0(b)
    assert list(h0.shape) == [g, b, h // g]
    out, hout = m(input, h0)
    print(out.shape, hout.shape)

    # now grouped gru
    num = 5
    m = SharedGroupGRU(i, h, num, g, batch_first=True, shuffle=True)
    h0 = m.initialize_h0(b)
    assert list(h0.shape) == [num * g, b, h // g]
    out, hout = m(input, h0)
    print(out.shape, hout.shape)
