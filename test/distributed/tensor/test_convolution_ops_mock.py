"""
Mock ups of TP convolutions.
"""

from typing import Optional
import torch
import torch.nn.functional as F
import pytest

CONV_FNS = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}


class TestConv:
    world_size = 4
    batch_size = 3
    C_in = C_out = 8
    d_model = 16 * world_size
    # Defaults
    kernel_size = 5
    padding = 0
    stride = 1
    dilation = 1
    n_conv_dims = 1

    def get_weight_tensor(
        self, kernel_size: int, n_conv_dims: int, seed: int = 42
    ) -> torch.Tensor:
        torch.manual_seed(seed)
        return torch.randn(
            self.C_in, self.C_out, *(kernel_size for _ in range(n_conv_dims))
        )

    def get_bias_tensor(self, seed: int = 42) -> torch.Tensor:
        torch.manual_seed(seed)
        return torch.randn(self.C_out)

    def _test_template_fwd(
        self,
        kernel_size: Optional[int] = None,
        stride: Optional[int] = None,
        padding: Optional[int] = None,
        dilation: Optional[int] = None,
        n_conv_dims: Optional[int] = None,
        d_model: Optional[int] = None,
        world_size: Optional[int] = None,
    ) -> None:
        """
        Compute the full convolution, then break into sharded inputs, recompute and verify forward
        correctness. Chunking along final dim.
        """
        kernel_size = kernel_size if kernel_size is not None else self.kernel_size
        stride = stride if stride is not None else self.stride
        padding = padding if padding is not None else self.padding
        dilation = dilation if dilation is not None else self.dilation
        n_conv_dims = n_conv_dims if n_conv_dims is not None else self.n_conv_dims
        d_model = d_model if d_model is not None else self.d_model
        world_size = world_size if world_size is not None else self.world_size

        torch.manual_seed(42)
        inputs = torch.randn(
            self.batch_size, self.C_in, *(d_model for _ in range(n_conv_dims))
        )

        weight = self.get_weight_tensor(kernel_size, n_conv_dims)
        bias = self.get_bias_tensor()
        outputs = CONV_FNS[n_conv_dims](
            inputs,
            weight=weight,
            bias=bias,
            stride=tuple(stride for _ in range(n_conv_dims)),
            padding=tuple(padding for _ in range(n_conv_dims)),
            dilation=tuple(dilation for _ in range(n_conv_dims)),
        )

        L_out_expected = (
            1 + (d_model + 2 * padding - dilation * (kernel_size - 1) - 1) // stride
        )
        assert L_out_expected == outputs.shape[-1]

        L_in_shard = (inputs.shape[-1] + world_size - 1) // world_size
        L_out_shard = (L_out_expected + world_size - 1) // world_size


        # # Create slice points for the necessary chunks. Each chunk needs to produce L_out_shard
        # # outputs, except maybe the final rank. Only the lead and final ranks will use any padding.
        # slice_start = [0]
        # slice_stop = []
        # # The number of output elements due to padding one one side of the input
        # n_pad_outputs = padding // stride
        # effective_kernel_size =  (dilation * (kernel_size - 1) + 1)
        # # Creating L_out_shard outputs requires this many inputs:
        # n_input_elem_per_chunk = stride * (L_out_shard-1) + effective_kernel_size
        # # Lead rank:
        # slice_stop.append(n_input_elem_per_chunk)
        # # Middle ranks:
        # # for _ in range(world_size-2):
        # #     slice_stop.append(slice_stop[-1] + step_size)
        # # Last rank
        # slice_stop.append(-1)


        outputs_chunked = []
        for rank in range(world_size):
            input_chunk = inputs[
                ...,
                rank * stride * L_out_shard : (dilation * (kernel_size - 1) + 1)
                - 1
                + (rank + 1) * stride * L_out_shard,
            ]
            not_first_or_last_rank = rank not in (0, world_size - 1)
            out_chunk = CONV_FNS[n_conv_dims](
                input_chunk,
                weight=weight,
                bias=bias,
                stride=tuple(stride for _ in range(n_conv_dims)),
                # If we're not the first or last rank, we shouldn't be using any padding at all.
                padding=tuple(
                    0 if not_first_or_last_rank else padding for _ in range(n_conv_dims)
                ),
                dilation=tuple(dilation for _ in range(n_conv_dims)),
            )
            outputs_chunked.append(out_chunk)
        outputs_chunked_cat = torch.cat(outputs_chunked, dim=-1)
        torch.testing.assert_close(outputs, outputs_chunked_cat)

    def test_1d_fwd(self) -> None:
        self._test_template_fwd(n_conv_dims=1)

    def test_2d_fwd(self) -> None:
        self._test_template_fwd(n_conv_dims=2)

    def test_3d_fwd(self) -> None:
        self._test_template_fwd(n_conv_dims=3)

    def test_1d_fwd_strided(self) -> None:
        self._test_template_fwd(n_conv_dims=1, stride=2)

    def test_2d_fwd_strided(self) -> None:
        self._test_template_fwd(n_conv_dims=2, stride=2)

    def test_3d_fwd_strided(self) -> None:
        self._test_template_fwd(n_conv_dims=3, stride=2)

    def test_1d_fwd_dilated(self) -> None:
        self._test_template_fwd(n_conv_dims=1, dilation=2)

    def test_2d_fwd_dilated(self) -> None:
        self._test_template_fwd(n_conv_dims=2, dilation=2)

    def test_3d_fwd_dilated(self) -> None:
        self._test_template_fwd(n_conv_dims=3, dilation=2)

    @pytest.mark.parametrize("world_size", (2, 3, 4))
    @pytest.mark.parametrize("n_conv_dims", (1, 2, 3))
    @pytest.mark.parametrize("stride", (1, 2, 3, 4))
    @pytest.mark.parametrize("dilation", (1, 2, 3, 4))
    @pytest.mark.parametrize("d_model", (63, 64, 65))
    def test_fwd_multi(
        self,
        world_size: int,
        n_conv_dims: int,
        stride: int,
        dilation: int,
        d_model: int,
    ) -> None:
        self._test_template_fwd(
            world_size=world_size,
            n_conv_dims=n_conv_dims,
            stride=stride,
            dilation=dilation,
            d_model=d_model,
        )
