# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor

import torch
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import OpSchema, OutputSharding
from torch.distributed.tensor._ops.utils import register_prop_rule


aten = torch.ops.aten


def _get_conv_output_shape_stride(
    in_shape: torch.Size,
    weight_shape: torch.Size,
    stride: list[int],
    padding: list[int],
    dilation: list[int],
) -> tuple[list[int], tuple[int, ...]]:
    C_out = weight_shape[0]

    W_in_t = torch.tensor(in_shape[2:])
    stride_t = torch.tensor(stride)
    padding_t = torch.tensor(padding)
    dilation_t = torch.tensor(dilation)
    dilation_t = torch.tensor(dilation)
    weight_shape_t = torch.tensor(weight_shape[2:])
    W_out_t = (
        W_in_t + 2 * padding_t - dilation_t * (weight_shape_t - 1) - 1
    ) // stride_t + 1


    # A batch dim is optional for convolutions.
    has_batch_dim = len(in_shape) == len(weight_shape)
    if has_batch_dim:
        output_shape = [in_shape[0], C_out, *W_out_t.tolist()]
    else:
        output_shape = [C_out, *W_out_t.tolist()]
    output_stride = (
        torch.tensor([1] + output_shape[-1:0:-1])
        .cumprod(dim=-1)
        .flip(dims=(-1,))
        .tolist()
    )

    return output_shape, output_stride


@register_prop_rule(aten.convolution.default)
def convolution_rules(op_schema: OpSchema) -> OutputSharding:
    (
        input_spec,
        weight_spec,
        bias_spec,
        stride,
        padding,
        dilation,
        _transposed,
        _output_padding,
        _groups,
    ) = op_schema.args_schema

    assert isinstance(input_spec, DTensorSpec)
    assert isinstance(weight_spec, DTensorSpec)
    assert isinstance(bias_spec, DTensorSpec) or bias_spec is None
    assert input_spec.tensor_meta is not None
    assert weight_spec.tensor_meta is not None
    in_shape = input_spec.tensor_meta.shape
    weight_shape = weight_spec.tensor_meta.shape
    assert isinstance(stride, list)
    assert isinstance(padding, list)
    assert isinstance(dilation, list)
    assert isinstance(weight_shape, torch.Size)

    output_shape, output_stride = _get_conv_output_shape_stride(
        in_shape,
        weight_shape,
        stride,
        padding,
        dilation,
    )

    tensor_meta = TensorMeta(
        torch.Size(output_shape),
        output_stride,
        input_spec.tensor_meta.dtype,
    )
    return OutputSharding(
        DTensorSpec.from_dim_map(
            input_spec.mesh,
            dim_map=input_spec.dim_map,
            sums=input_spec.sums,
            tensor_meta=tensor_meta,
        )
    )


@register_prop_rule(aten.convolution_backward.default)
def convolution_backward_rules(op_schema: OpSchema) -> OutputSharding:
    input_spec = op_schema.args_schema[0]
    (
        grad_output_spec,
        input_spec,
        weight_spec,
        bias_shape_opt,
        _stride,
        _padding,
        _dilation,
        _transposed,
        _output_padding,
        _groups,
        _output_mask,
    ) = op_schema.args_schema

    assert isinstance(grad_output_spec, DTensorSpec)
    assert isinstance(input_spec, DTensorSpec)
    assert isinstance(weight_spec, DTensorSpec)
    assert isinstance(bias_shape_opt, list)
    assert input_spec.tensor_meta is not None

    # _output_mask order is (input, weight, bias)
    grad_input_spec = input_spec if _output_mask[0] else None
    if _output_mask[1]:
        weight_tensor_meta = weight_spec.tensor_meta
        grad_weight_spec = DTensorSpec.from_dim_map(
            input_spec.mesh,
            [-1 for _ in range(len(weight_tensor_meta.shape))],
            [0],
            tensor_meta=weight_tensor_meta,
        )
    else:
        grad_weight_spec = None
    if _output_mask[2]:
        bias_tensor_meta = TensorMeta(
            torch.Size(bias_shape_opt),
            (1,),
            input_spec.tensor_meta.dtype,
        )
        grad_bias_spec = DTensorSpec.from_dim_map(
            input_spec.mesh,
            [-1],
            [0],
            tensor_meta=bias_tensor_meta,
        )
    else:
        grad_bias_spec = None
    # TODO: actually the output_mask is not respected here, we should
    # set the corresponding spec to `None` if the output_mask is not `False`
    # for a certain output Tensor. This also applies to the conv handler
    # in torch/distributed/tensor/_tp_conv.py
    return OutputSharding([grad_input_spec, grad_weight_spec, grad_bias_spec])
