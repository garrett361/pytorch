"""
Mock ups of TP convolutions.
"""

import torch
import torch.nn.functional as F


class TestConv1D:
    world_size = 4
    batch_size = 3
    C_in = C_out = 8
    d_model = 16 * world_size
    kernel_size = 5
    padding = 0
    stride = 1
    dilation = 1
    n_conv_dims = 1
    conv = F.conv1d

    def get_weight_tensor(self, seed: int = 42) -> torch.Tensor:
        torch.manual_seed(seed)
        return torch.randn(
            self.C_in, self.C_out, *(self.kernel_size for _ in range(self.n_conv_dims))
        )

    def get_bias_tensor(self, seed: int = 42) -> torch.Tensor:
        torch.manual_seed(seed)
        return torch.randn(self.C_out)

    def test_simple(self) -> None:
        torch.manual_seed(42)
        inputs = torch.randn(self.batch_size, self.C_in, self.d_model)
        inputs_chunked = inputs.chunk(chunks=self.world_size, dim=-1)

        inputs = inputs.requires_grad_()
        inputs_chunked = [i.requires_grad_() for i in inputs_chunked]

        weight = self.get_weight_tensor()
        bias = self.get_bias_tensor()
        outputs = self.conv(
            inputs,
            weight=weight,
            bias=bias,
            stride=tuple(self.stride for _ in range(self.n_conv_dims)),
            padding=tuple(self.padding for _ in range(self.n_conv_dims)),
            dilation=tuple(self.dilation for _ in range(self.n_conv_dims)),
        )

        L_out_expected = (
            1
            + (
                self.d_model
                + 2 * self.padding
                - self.dilation * (self.kernel_size - 1)
                - 1
            )
            // self.stride
        )
        assert L_out_expected == outputs.shape[-1]

        L_in_shard = inputs.shape[-1] // self.world_size
        L_out_shard = outputs.shape[-1] // self.world_size

        outputs_chunked = []
        for rank in range(self.world_size):
            input_chunk = inputs[
                ...,
                rank * L_out_shard : self.kernel_size - 1 + (rank + 1) * L_out_shard,
            ]
            outputs_chunked.append(
                self.conv(
                    input_chunk,
                    weight=weight,
                    bias=bias,
                    stride=tuple(self.stride for _ in range(self.n_conv_dims)),
                    padding=tuple(self.padding for _ in range(self.n_conv_dims)),
                    dilation=tuple(self.dilation for _ in range(self.n_conv_dims)),
                )
            )
        outputs_chunked_cat = torch.cat(outputs_chunked, dim=-1)
        torch.testing.assert_close(outputs, outputs_chunked_cat)


