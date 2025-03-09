"""
Mock ups of TP convolutions.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

import pytest
import torch

aten = torch.ops.aten


@dataclass
class PyTestParams:
    arg_name: str
    vals: tuple[Any, ...]

    @property
    def ids(self) -> tuple[str, ...]:
        return [f"({self.arg_name}={v})" for v in self.vals]


WORLD_SIZES = PyTestParams("world_size", (2, 3, 4))
N_CONV_DIMS = PyTestParams("n_conv_dims", (1, 2, 3))
STRIDES = PyTestParams("stride", (1, 2, 3, 4))
D_MODELS = PyTestParams("d_model", (63, 64, 65))
DILATIONS = PyTestParams("dilation", (1, 2, 3, 4))
PADDINGS = PyTestParams("padding", (0, 1, 2, 3))


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
        outputs = aten.convolution.default(
            inputs,
            weight=weight,
            bias=bias,
            stride=tuple(stride for _ in range(n_conv_dims)),
            padding=tuple(padding for _ in range(n_conv_dims)),
            dilation=tuple(dilation for _ in range(n_conv_dims)),
            transposed=False,
            groups=1,
            output_padding=tuple(0 for _ in range(n_conv_dims)),
        )

        eff_kernel_size = dilation * (kernel_size - 1) + 1
        L_out_expected = 1 + (d_model + 2 * padding - eff_kernel_size) // stride
        assert L_out_expected == outputs.shape[-1]

        # Create slice points for the necessary chunks. Salient facts:
        # 1) Each rank creates at least L_out_expected // world_size elements, and any leftover
        #    are passed out in order. Let L_out_shard[rank] be the number of output elements.
        # 2) In the absence of padding, a total of n_input_elem[rank] = eff_kernel_size + stride *
        #    (L_out_shard[rank] - 1) elements from the inputs are required to create the right
        #    number of outputs. Padding just means we load padding fewer elements for the first
        #    sharded output computation.
        # 3) If a shard covers the slice [slice_start[rank]:slice_start[rank]] for a given rank,
        #    then the kernel covered up to and including the position slice_start[rank] - 1. Rank
        #    rank +1 will then start at slice_start[rank + 1] = slice_stop[rank] - eff_kernel_size +
        #    stride and cover n_input_elem[rank] total elements.

        L_out_shard_min, remainder = divmod(L_out_expected, world_size)
        L_out_shard = [
            L_out_shard_min + (rank < remainder) for rank in range(world_size)
        ]
        n_input_elem = [
            eff_kernel_size + stride * (L_out_shard[rank] - 1)
            for rank in range(world_size)
        ]
        slice_start = [0]
        slice_stop = [n_input_elem[0] - padding]
        for rank in range(1, world_size):
            slice_start.append(slice_stop[-1] + stride - eff_kernel_size)
            slice_stop.append(slice_start[-1] + n_input_elem[rank])

        # Padding considerations:
        # 1) The first shard uses unwanted padding along the sharded dim at its upper range, and
        #    this is removed by only taking the first L_out_shard elements.
        # 2) None of the middle shards should use padding along the sharded dim.
        # 3) The final slice must use both use at least the requested amount of padding at its upper
        #    end, while also ensuring that the kernel becomes naturally aligned with the start of its
        #    non-padding data.  Solution: use stride * ceil(padding / stride) and take the output
        #    slice starting from index ceil(padding / stride) and containing the number of output
        #    elements expected in the final shard (which may be fewer than L_out_shard!)

        last_shard_out_idx_start = (padding + stride - 1) // stride
        last_shard_padding = stride * last_shard_out_idx_start

        stride_list = [stride for _ in range(n_conv_dims)]
        dilation_list = [dilation for _ in range(n_conv_dims)]
        # Ranks differ in their padding along the sharded_dim
        padding_list_start = [padding for _ in range(n_conv_dims - 1)]

        outputs_chunked = []
        for rank in range(world_size):
            is_lead_rank = rank == 0
            is_last_rank = rank == world_size - 1
            # Get the right sharded dim padding:
            if is_lead_rank:
                padding_list = padding_list_start + [padding]
            elif is_last_rank:
                padding_list = padding_list_start + [last_shard_padding]
            else:
                padding_list = padding_list_start + [0]
            # Need a bounds check for `slice_start[rank]` equal to the end of the inputs slice.

            input_chunk = inputs[..., slice_start[rank] : slice_stop[rank]]
            # If we're not the first or last rank, we shouldn't be using any padding at all
            # along the sharded dim.

            out_chunk = aten.convolution.default(
                input_chunk,
                weight=weight,
                bias=bias,
                stride=stride_list,
                padding=padding_list,
                dilation=dilation_list,
                transposed=False,
                groups=1,
                output_padding=tuple(0 for _ in range(n_conv_dims)),
            )
            if padding > 0 and is_lead_rank:
                out_chunk = out_chunk[..., : L_out_shard[0]]
            if padding > 0 and is_last_rank:
                out_chunk = out_chunk[
                    ...,
                    last_shard_out_idx_start : last_shard_out_idx_start
                    + L_out_shard[-1],
                ]
            outputs_chunked.append(out_chunk)
        outputs_chunked_cat = torch.cat(outputs_chunked, dim=-1)
        torch.testing.assert_close(outputs, outputs_chunked_cat)

    @pytest.mark.parametrize(
        WORLD_SIZES.arg_name, WORLD_SIZES.vals, ids=WORLD_SIZES.ids
    )
    @pytest.mark.parametrize(
        N_CONV_DIMS.arg_name, N_CONV_DIMS.vals, ids=N_CONV_DIMS.ids
    )
    @pytest.mark.parametrize(STRIDES.arg_name, STRIDES.vals, ids=STRIDES.ids)
    @pytest.mark.parametrize(DILATIONS.arg_name, DILATIONS.vals, ids=DILATIONS.ids)
    @pytest.mark.parametrize(D_MODELS.arg_name, D_MODELS.vals, ids=D_MODELS.ids)
    @pytest.mark.parametrize(PADDINGS.arg_name, PADDINGS.vals, ids=PADDINGS.ids)
    def test_fwd_multi(
        self,
        world_size: int,
        n_conv_dims: int,
        stride: int,
        dilation: int,
        d_model: int,
        padding: int,
    ) -> None:
        self._test_template_fwd(
            world_size=world_size,
            n_conv_dims=n_conv_dims,
            stride=stride,
            dilation=dilation,
            d_model=d_model,
            padding=padding,
        )

    def _test_template_bwd(
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
            self.batch_size,
            self.C_in,
            *(d_model for _ in range(n_conv_dims)),
            requires_grad=True,
        )
        inputs_tp = deepcopy(inputs)

        weight = self.get_weight_tensor(kernel_size, n_conv_dims)
        bias = self.get_bias_tensor()

        # Create some random upstream grads.
        with torch.no_grad():
            outputs = aten.convolution.default(
                inputs,
                weight=weight,
                bias=bias,
                stride=tuple(stride for _ in range(n_conv_dims)),
                padding=tuple(padding for _ in range(n_conv_dims)),
                dilation=tuple(dilation for _ in range(n_conv_dims)),
                transposed=False,
                groups=1,
                output_padding=tuple(0 for _ in range(n_conv_dims)),
            )

            grad_outputs = torch.randn_like(outputs)

        # Signature: convolution_backward(Tensor grad_output,
        #                                 Tensor input,
        #                                 Tensor weight,
        #                                 SymInt[]? bias_sizes,
        #                                 SymInt[] stride,
        #                                 SymInt[] padding,
        #                                 SymInt[] dilation,
        #                                 bool transposed,
        #                                 SymInt[] output_padding,
        #                                 SymInt groups,
        #                                 bool[3] output_mask) -> (Tensor, Tensor, Tensor)
        # Where the three outputs are grad_inputs, grad_weight, grad_bias

        grad_inputs, grad_weight, grad_bias = aten.convolution_backward.default(
            grad_outputs,
            inputs,
            weight,
            bias,
            stride=tuple(stride for _ in range(n_conv_dims)),
            padding=tuple(padding for _ in range(n_conv_dims)),
            dilation=tuple(dilation for _ in range(n_conv_dims)),
            transposed=False,
            groups=1,
            output_padding=tuple(0 for _ in range(n_conv_dims)),
            output_mask=(True, True, True),
        )

        eff_kernel_size = dilation * (kernel_size - 1) + 1
        L_out_expected = 1 + (d_model + 2 * padding - eff_kernel_size) // stride
        assert L_out_expected == outputs.shape[-1]

        # Create slice points for the necessary chunks. Salient facts:
        # 1) Each rank creates at least L_out_expected // world_size elements, and any leftover
        #    are passed out in order. Let L_out_shard[rank] be the number of output elements.
        # 2) In the absence of padding, a total of n_input_elem[rank] = eff_kernel_size + stride *
        #    (L_out_shard[rank] - 1) elements from the inputs are required to create the right
        #    number of outputs. Padding just means we load padding fewer elements for the first
        #    sharded output computation.
        # 3) If a shard covers the slice [slice_start[rank]:slice_start[rank]] for a given rank,
        #    then the kernel covered up to and including the position slice_start[rank] - 1. Rank
        #    rank +1 will then start at slice_start[rank + 1] = slice_stop[rank] - eff_kernel_size +
        #    stride and cover n_input_elem[rank] total elements.

        L_out_shard_min, remainder = divmod(L_out_expected, world_size)
        L_out_shard = [
            L_out_shard_min + (rank < remainder) for rank in range(world_size)
        ]
        n_input_elem = [
            eff_kernel_size + stride * (L_out_shard[rank] - 1)
            for rank in range(world_size)
        ]
        slice_start = [0]
        slice_stop = [n_input_elem[0] - padding]
        for rank in range(1, world_size):
            slice_start.append(slice_stop[-1] + stride - eff_kernel_size)
            slice_stop.append(slice_start[-1] + n_input_elem[rank])

        # Padding considerations:
        # 1) The first shard uses unwanted padding along the sharded dim at its upper range, and
        #    this is removed by only taking the first L_out_shard elements.
        # 2) None of the middle shards should use padding along the sharded dim.
        # 3) The final slice must use both use at least the requested amount of padding at its upper
        #    end, while also ensuring that the kernel becomes naturally aligned with the start of its
        #    non-padding data.  Solution: use stride * ceil(padding / stride) and take the output
        #    slice starting from index ceil(padding / stride) and containing the number of output
        #    elements expected in the final shard (which may be fewer than L_out_shard!)
        last_shard_out_idx_start = (padding + stride - 1) // stride
        last_shard_padding = stride * last_shard_out_idx_start

        stride_list = [stride for _ in range(n_conv_dims)]
        dilation_list = [dilation for _ in range(n_conv_dims)]
        # Ranks differ in their padding along the sharded_dim
        padding_list_start = [padding for _ in range(n_conv_dims - 1)]

        out_slice_start = [0]
        out_slice_stop = [L_out_shard[0]]
        for rank in range(1, world_size):
            out_slice_start.append(out_slice_stop[-1])
            out_slice_stop.append(out_slice_start[-1] + L_out_shard[rank])

        grad_inputs_chunked = []
        grad_weight_chunked = grad_bias_chunked =0.
        for rank in range(world_size):
            is_lead_rank = rank == 0
            is_last_rank = rank == world_size - 1
            # Get the right sharded dim padding:
            if is_lead_rank:
                padding_list = padding_list_start + [padding]
            elif is_last_rank:
                padding_list = padding_list_start + [last_shard_padding]
            else:
                padding_list = padding_list_start + [0]
            # Need a bounds check for `slice_start[rank]` equal to the end of the inputs slice.

            input_chunk = inputs[..., slice_start[rank] : slice_stop[rank]]
            grad_outputs_chunk = grad_outputs[
                ..., out_slice_start[rank] : out_slice_stop[rank]
            ]
            # If we're not the first or last rank, we shouldn't be using any padding at all
            # along the sharded dim.

            grad_inputs_chunk, grad_weight_partial, grad_bias_partial = aten.convolution_backward.default(
                grad_outputs_chunk,
                input_chunk,
                weight,
                bias,
                stride=stride_list,
                padding=padding_list,
                dilation=dilation_list,
                transposed=False,
                groups=1,
                output_padding=tuple(0 for _ in range(n_conv_dims)),
                output_mask=(True, True, True),
            )


            grad_weight_chunked+=grad_weight_partial
            grad_bias_chunked+=grad_bias_partial
            # if padding > 0 and is_lead_rank:
            #     out_chunk = out_chunk[..., : L_out_shard[0]]
            # if padding > 0 and is_last_rank:
            #     out_chunk = out_chunk[
            #         ...,
            #         last_shard_out_idx_start : last_shard_out_idx_start
            #         + L_out_shard[-1],
            #     ]
            grad_inputs_chunked.append(grad_inputs_chunk)
        grad_inputs_cat = torch.cat(grad_inputs_chunked, dim=-1)
        torch.testing.assert_close(grad_weight, grad_weight_chunked)
        torch.testing.assert_close(grad_bias, grad_bias_chunked)
        torch.testing.assert_close(grad_inputs, grad_inputs_cat)

    def test_bwd(self) -> None:
        self._test_template_bwd()
