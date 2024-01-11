import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional

from audiozen.constant import EPSILON


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    @staticmethod
    def freq_unfold(input, num_neighbors):
        """Split the overlapped subband units along the frequency axis.

        Args:
            input: four-dimension input with the shape [B, C, F, T].
            num_neighbors: number of neighbors in each side for each subband unit.

        Returns:
            Overlapped sub-band units specified as [B, N, C, F_s, T], where `F_s` represents the frequency axis of
            each sub-band unit and `N` is the number of sub-band unit, e.g., [2, 161, 1, 19, 100].
        """
        assert input.dim() == 4, f"The dim of the input is {input.dim()}. It should be four dim."
        batch_size, num_channels, num_freqs, num_frames = input.size()

        if num_neighbors <= 0:  # No change to the input
            return input.permute(0, 2, 1, 3).reshape(batch_size, num_freqs, num_channels, 1, num_frames)

        output = input.reshape(batch_size * num_channels, 1, num_freqs, num_frames)  # [B * C, 1, F, T]
        sub_band_unit_size = num_neighbors * 2 + 1

        # Pad the top and bottom of the original spectrogram
        output = functional.pad(output, [0, 0, num_neighbors, num_neighbors], mode="reflect")  # [B * C, 1, F, T]

        # Unfold the spectrogram into sub-band units
        # [B * C, 1, F, T] => [B * C, sub_band_unit_size, num_frames, N], N is equal to the number of frequencies.
        output = functional.unfold(output, kernel_size=(sub_band_unit_size, num_frames))  # move on the F and T axes
        assert output.shape[-1] == num_freqs, f"n_freqs != N (sub_band), {num_freqs} != {output.shape[-1]}"

        # Split the dimension of the unfolded feature
        output = output.reshape(batch_size, num_channels, sub_band_unit_size, num_frames, num_freqs)
        output = output.permute(0, 4, 1, 2, 3).contiguous()  # [B, N, C, F_s, T]

        return output

    @staticmethod
    def unfold_time(input, kernel_size, stride):
        """Unfold (like sliding window) input tensor along the time axis.

        Shapes:
            input: [B, C, F, T]. PyTorch only support 4-D (image-like) tensor.
            kernel_size: Tuple with two values
            stride: Tuple with two values
            return: [B, C, kernel_size_1, kernel_size_2, N]
        """
        batch_size, num_channels, num_freqs, num_frames = input.shape

        # [B, C * prod(kernel_size), N]
        output = torch.nn.functional.unfold(input, kernel_size=kernel_size, stride=stride)
        num_blocks = output.shape[-1]

        # [B, C, kernel_size[0], kernel_size[1], N]
        output = output.reshape(batch_size, num_channels, *kernel_size, num_blocks)

        return output

    @staticmethod
    def time_fold(input, kernel_size, stride, output_size):
        """Fold (like sliding window) input tensor along the time axis.

        Shapes:
            input: [B, C, kernel_size[0], kernel_size[1], N]
            kernel_size: Tuple with two values
            stride: Tuple with two values
            output_size: Tuple of the spatial shape, e.g., (num_freqs, num_frames)
            return: [B, C, F, T]
        """
        batch_size, num_channels, ks_1, ks_2, num_blocks = input.shape
        input = input.reshape(batch_size, -1, num_blocks)  # [B, C * prod(kernel_size), N]

        output = torch.nn.functional.fold(
            input, kernel_size=kernel_size, stride=stride, output_size=output_size
        )  # [B, C, F, T]

        return output

    @staticmethod
    def _pad_length(tensor, win_length):
        """Pad the length of the tensor to be divisible by the win_length.

        Args:
            tensor: [*, T].
            win_length: window length.

        Returns:
            Padded tensor with the shape of [*, T_pad].
        """
        seq_len = tensor.shape[-1]
        remainder = seq_len % win_length

        pad_size = win_length - remainder if remainder > 0 else 0

        if pad_size > 0:
            tensor = torch.cat(
                (
                    tensor,
                    torch.zeros([*(tensor.shape[:-1]), pad_size], device=tensor.device),
                ),
                dim=-1,
            )

        return tensor, pad_size

    @staticmethod
    def _reduce_complexity_separately(sub_band_input, full_band_output, device):
        """
        Group Dropout for FullSubNet

        Args:
            sub_band_input: [60, 257, 1, 33, 200]
            full_band_output: [60, 257, 1, 3, 200]
            device:

        Notes:
            1. 255 and 256 freq not able to be trained
            2. batch size 应该被 3 整除，否则最后一部分 batch 内的频率无法很好的训练

        Returns:
            [60, 85, 1, 36, 200]
        """
        batch_size = full_band_output.shape[0]
        n_freqs = full_band_output.shape[1]
        sub_batch_size = batch_size // 3
        final_selected = []

        for idx in range(3):
            # [0, 60) => [0, 20)
            sub_batch_indices = torch.arange(idx * sub_batch_size, (idx + 1) * sub_batch_size, device=device)
            full_band_output_sub_batch = torch.index_select(full_band_output, dim=0, index=sub_batch_indices)
            sub_band_output_sub_batch = torch.index_select(sub_band_input, dim=0, index=sub_batch_indices)

            # Avoid to use padded value (first freq and last freq)
            # i = 0, (1, 256, 3) = [1, 4, ..., 253]
            # i = 1, (2, 256, 3) = [2, 5, ..., 254]
            # i = 2, (3, 256, 3) = [3, 6, ..., 255]
            freq_indices = torch.arange(idx + 1, n_freqs - 1, step=3, device=device)
            full_band_output_sub_batch = torch.index_select(full_band_output_sub_batch, dim=1, index=freq_indices)
            sub_band_output_sub_batch = torch.index_select(sub_band_output_sub_batch, dim=1, index=freq_indices)

            # ([30, 85, 1, 33 200], [30, 85, 1, 3, 200]) => [30, 85, 1, 36, 200]

            final_selected.append(torch.cat([sub_band_output_sub_batch, full_band_output_sub_batch], dim=-2))

        return torch.cat(final_selected, dim=0)

    @staticmethod
    def forgetting_norm(input, sample_length=192):
        """
        Using the mean value of the near frames to normalization

        Args:
            input: feature
            sample_length: length of the training sample, used for calculating smooth factor

        Returns:
            normed feature

        Shapes:
            input: [B, C, F, T]
            sample_length_in_training: 192
        """
        assert input.ndim == 4
        batch_size, num_channels, num_freqs, num_frames = input.size()
        input = input.reshape(batch_size, num_channels * num_freqs, num_frames)

        eps = EPSILON
        mu = 0
        alpha = (sample_length - 1) / (sample_length + 1)

        mu_list = []
        # TODO bugs
        # TODO 是否需要修改为包含F的计算方法？
        for frame_idx in range(num_frames):
            if frame_idx < sample_length:
                alp = torch.min(torch.tensor([(frame_idx - 1) / (frame_idx + 1), alpha]))
                mu = alp * mu + (1 - alp) * torch.mean(input[:, :, frame_idx], dim=1).reshape(batch_size, 1)  # [B, 1]
            else:
                current_frame_mu = torch.mean(input[:, :, frame_idx], dim=1).reshape(batch_size, 1)  # [B, 1]
                mu = alpha * mu + (1 - alpha) * current_frame_mu

            mu_list.append(mu)

            # print("input", input[:, :, idx].min(), input[:, :, idx].max(), input[:, :, idx].mean())
            # print(f"alp {idx}: ", alp)
            # print(f"mu {idx}: {mu[128, 0]}")

        mu = torch.stack(mu_list, dim=-1)  # [B, 1, T]
        output = input / (mu + eps)

        output = output.reshape(batch_size, num_channels, num_freqs, num_frames)
        return output

    @staticmethod
    def hybrid_norm(input, sample_length_in_training=192):
        """
        Args:
            input: [B, F, T]
            sample_length_in_training:

        Returns:
            [B, F, T]
        """
        assert input.ndim == 3
        device = input.device
        data_type = input.dtype
        batch_size, n_freqs, n_frames = input.size()
        eps = EPSILON

        mu = 0
        alpha = (sample_length_in_training - 1) / (sample_length_in_training + 1)
        mu_list = []
        for idx in range(input.shape[-1]):
            if idx < sample_length_in_training:
                alp = torch.min(torch.tensor([(idx - 1) / (idx + 1), alpha]))
                mu = alp * mu + (1 - alp) * torch.mean(input[:, :, idx], dim=1).reshape(batch_size, 1)  # [B, 1]
                mu_list.append(mu)
            else:
                break
        initial_mu = torch.stack(mu_list, dim=-1)  # [B, 1, T]

        step_sum = torch.sum(input, dim=1)  # [B, T]
        cumulative_sum = torch.cumsum(step_sum, dim=-1)  # [B, T]

        entry_count = torch.arange(n_freqs, n_freqs * n_frames + 1, n_freqs, dtype=data_type, device=device)
        entry_count = entry_count.reshape(1, n_frames)  # [1, T]
        entry_count = entry_count.expand_as(cumulative_sum)  # [1, T] => [B, T]

        cum_mean = cumulative_sum / entry_count  # B, T

        cum_mean = cum_mean.reshape(batch_size, 1, n_frames)  # [B, 1, T]

        # print(initial_mu[0, 0, :50])
        # print("-"*60)
        # print(cum_mean[0, 0, :50])
        cum_mean[:, :, :sample_length_in_training] = initial_mu

        return input / (cum_mean + eps)

    @staticmethod
    def offline_laplace_norm(input, return_mu=False):
        """Normalize the input with the utterance-level mean.

        Args:
            input: [B, C, F, T]

        Returns:
            [B, C, F, T]

        Notes:
            As mentioned in the paper, the offline normalization is used.
            Based on a bunch of experiments, the offline normalization have the same performance as the cumulative one and have a faster convergence than the cumulative one.
            Therefore, we use the offline normalization as the default normalization method.
        """
        # utterance-level mu
        mu = torch.mean(input, dim=list(range(1, input.dim())), keepdim=True)

        normed = input / (mu + EPSILON)

        if return_mu:
            return normed, mu
        else:
            return normed

    @staticmethod
    def cumulative_laplace_norm(input):
        """Normalize the input with the cumulative mean

        Args:
            input: [B, C, F, T]

        Returns:

        """
        batch_size, num_channels, num_freqs, num_frames = input.size()
        input = input.reshape(batch_size * num_channels, num_freqs, num_frames)

        step_sum = torch.sum(input, dim=1)  # [B * C, F, T] => [B, T]
        cumulative_sum = torch.cumsum(step_sum, dim=-1)  # [B, T]

        entry_count = torch.arange(
            num_freqs,
            num_freqs * num_frames + 1,
            num_freqs,
            dtype=input.dtype,
            device=input.device,
        )
        entry_count = entry_count.reshape(1, num_frames)  # [1, T]
        entry_count = entry_count.expand_as(cumulative_sum)  # [1, T] => [B, T]

        cumulative_mean = cumulative_sum / entry_count  # B, T
        cumulative_mean = cumulative_mean.reshape(batch_size * num_channels, 1, num_frames)

        normed = input / (cumulative_mean + EPSILON)

        return normed.reshape(batch_size, num_channels, num_freqs, num_frames)

    @staticmethod
    def offline_gaussian_norm(input):
        """
        Zero-Norm
        Args:
            input: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        mu = torch.mean(input, dim=list(range(1, input.dim())), keepdim=True)
        std = torch.std(input, dim=list(range(1, input.dim())), keepdim=True)

        normed = (input - mu) / (std + EPSILON)
        return normed

    @staticmethod
    def cumulative_layer_norm(input):
        """
        Online zero-norm

        Args:
            input: [B, C, F, T]

        Returns:
            [B, C, F, T]
        """
        batch_size, num_channels, num_freqs, num_frames = input.size()
        input = input.reshape(batch_size * num_channels, num_freqs, num_frames)

        step_sum = torch.sum(input, dim=1)  # [B * C, F, T] => [B, T]
        step_pow_sum = torch.sum(torch.square(input), dim=1)

        cumulative_sum = torch.cumsum(step_sum, dim=-1)  # [B, T]
        cumulative_pow_sum = torch.cumsum(step_pow_sum, dim=-1)  # [B, T]

        entry_count = torch.arange(
            num_freqs,
            num_freqs * num_frames + 1,
            num_freqs,
            dtype=input.dtype,
            device=input.device,
        )
        entry_count = entry_count.reshape(1, num_frames)  # [1, T]
        entry_count = entry_count.expand_as(cumulative_sum)  # [1, T] => [B, T]

        cumulative_mean = cumulative_sum / entry_count  # [B, T]
        cumulative_var = (
            cumulative_pow_sum - 2 * cumulative_mean * cumulative_sum
        ) / entry_count + cumulative_mean.pow(2)  # [B, T]
        cumulative_std = torch.sqrt(cumulative_var + EPSILON)  # [B, T]

        cumulative_mean = cumulative_mean.reshape(batch_size * num_channels, 1, num_frames)
        cumulative_std = cumulative_std.reshape(batch_size * num_channels, 1, num_frames)

        normed = (input - cumulative_mean) / cumulative_std

        return normed.reshape(batch_size, num_channels, num_freqs, num_frames)

    def norm_wrapper(self, norm_type: str):
        if norm_type == "offline_laplace_norm":
            norm = self.offline_laplace_norm
        elif norm_type == "cumulative_laplace_norm":
            norm = self.cumulative_laplace_norm
        elif norm_type == "offline_gaussian_norm":
            norm = self.offline_gaussian_norm
        elif norm_type == "cumulative_layer_norm":
            norm = self.cumulative_layer_norm
        elif norm_type == "forgetting_norm":
            norm = self.forgetting_norm
        else:
            raise NotImplementedError(
                "You must set up a type of Norm. "
                "e.g. offline_laplace_norm, cumulative_laplace_norm, forgetting_norm, etc."
            )
        return norm

    def weight_init(self, m):
        """
        Usage:
            model = Model()
            model.apply(weight_init)
        """
        if isinstance(m, nn.Conv1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.BatchNorm1d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm3d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.LSTMCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRU):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRUCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
