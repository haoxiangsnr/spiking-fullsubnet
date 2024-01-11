from itertools import permutations

import torch


class PairwiseNegSDR:
    def __init__(self, zero_mean=True, EPS=1e-8):
        self.zero_mean = zero_mean
        self.EPS = EPS

    def __call__(self, est, ref):
        """_summary_

        Args:
            est (`torch.Tensor` of shape [batch_size, num_sources, sequence_length]):
                The batch of target estimates.
            ref: _description_

        Returns:
            `torch.Tensor` of shape [batch_size, num_sources, num_sources]:
                The pairwise losses between all permutations of the sources. Dim 1 corresp. to estimates and dim 2 to references.
        """
        if ref.shape != est.shape or ref.ndim != 3:
            raise TypeError(f"Inputs must be of shape [batch, n_src, time], got {ref.shape} and {est.shape} instead")

        # Zero-mean norm for removing DC offset
        if self.zero_mean:
            mean_ref = torch.mean(ref, dim=2, keepdim=True)
            mean_est = torch.mean(est, dim=2, keepdim=True)
            ref = ref - mean_ref
            est = est - mean_est

        # Pair-wise SI-SDR. (Reshape to use broadcast)
        s_est = torch.unsqueeze(est, dim=2)  # [batch, n_src, 1, time]
        s_ref = torch.unsqueeze(ref, dim=1)  # [batch, 1, n_src, time]

        # [batch, n_src, n_src, 1]
        pair_wise_dot = torch.sum(s_est * s_ref, dim=3, keepdim=True)

        # [batch, 1, n_src, 1]
        s_ref_energy = torch.sum(s_ref**2, dim=3, keepdim=True) + self.EPS

        # project est on target
        # [batch, n_src, n_src, time]
        pair_wise_proj = pair_wise_dot * s_ref / s_ref_energy

        # Distortion
        e_noise = s_est - pair_wise_proj

        # SDR
        # [batch, n_src, n_src]
        pair_wise_sdr = torch.sum(pair_wise_proj**2, dim=3) / (torch.sum(e_noise**2, dim=3) + self.EPS)

        pair_wise_sdr = 10 * torch.log10(pair_wise_sdr + self.EPS)

        return -pair_wise_sdr


class PITWrapper:
    def __init__(self, loss_func):
        self.loss_func = loss_func

    def find_best_perm(self, pair_wise_losses):
        """Find the best permutation using the simplest method.

        Args:
            pw_losses (`torch.Tensor` of shape [batch_size, num_sources, num_sources]):
                The pairwise losses between all permutations of the sources.
        """
        num_sources = pair_wise_losses.shape[1]

        # After transposition, dim 1 corresp. to sources and dim 2 to estimates
        pwl = pair_wise_losses.transpose(-1, -2)

        # Create all possible permutations of the sources
        # e.g., [(0, 1), (1, 0)] with num_sources = 2 (shape = [2!, 2, 1])
        perms = pwl.new_tensor(list(permutations(range(num_sources))), dtype=torch.long)

        # Column permutation indices
        idx = perms.unsqueeze(2)

        # Loss mean of each permutation
        # one-hot, [n_src!, n_src, n_src]
        perms_one_hot = pwl.new_zeros((*perms.size(), num_sources)).scatter_(2, idx, 1)
        loss_set = torch.einsum("bij,pij->bp", [pwl, perms_one_hot])
        loss_set /= num_sources

        # Indexes and values of min losses for each batch element
        min_loss, min_loss_idx = torch.min(loss_set, dim=1)

        # Permutation indices for each batch.
        batch_indices = torch.stack([perms[m] for m in min_loss_idx], dim=0)

        return min_loss, batch_indices

    @staticmethod
    def reorder_source(source, batch_indices):
        r"""Reorder sources according to the best permutation.

        Args:
            source (torch.Tensor): Tensor of shape :math:`(batch, n_src, time)`
            batch_indices (torch.Tensor): Tensor of shape :math:`(batch, n_src)`.
                Contains optimal permutation indices for each batch.
        """
        reordered_sources = torch.stack([torch.index_select(s, 0, b) for s, b in zip(source, batch_indices)])
        return reordered_sources

    def __call__(self, est, ref, **kwargs):
        """

        Args:
            est (`torch.Tensor` of shape [batch_size, num_sources, ...]):
                The batch of target estimates.
            ref: _description_
        """
        num_sources = est.shape[1]
        assert num_sources < 10, f"Expected source axis along dim 1, found {num_sources}"

        # [batch_size, num_sources, num_sources]
        pw_losses = self.loss_func(est, ref, **kwargs)
        min_loss, batch_indices = self.find_best_perm(pw_losses)
        mean_loss = torch.mean(min_loss)
        reordered = self.reorder_source(est, batch_indices)
        return mean_loss, reordered


if __name__ == "__main__":
    ref = torch.rand(2, 3, 16000)
    est = torch.ones(2, 3, 16000)

    loss_func = PairwiseNegSDR(zero_mean=False)
    pit_loss = PITWrapper(loss_func)
    loss, reordered = pit_loss(est, ref)
    print(loss)
