# Copyright (c) Meta Platforms, Inc. and affiliates
# implement matrix related ops for distributed tensor

import torch
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._op_schema import OpSchema, OutputSharding
from torch.distributed.tensor._ops.utils import register_prop_rule


aten = torch.ops.aten


def _get_conv_output_shape_stride(
    n_conv_dims: int,
    in_shape: torch.Size,
    weight_shape: torch.Size,
    stride: list[int],
    padding: list[int],
    dilation: list[int],
) -> tuple[list[int], tuple[int, ...]]:
    if n_conv_dims == 1:
        N, W_in = in_shape[0], in_shape[-1]
        C_out = weight_shape[0]
        W_out = (
            W_in + 2 * padding[0] - dilation[0] * (weight_shape[-1] - 1) - 1
        ) // stride[0] + 1
        output_shape = [N, C_out, W_out]
        output_stride = (C_out * W_out, W_out, 1)
    elif n_conv_dims == 2:
        N, H_in, W_in = in_shape[0], in_shape[2], in_shape[3]
        C_out = weight_shape[0]
        H_out = (
            H_in + 2 * padding[0] - dilation[0] * (weight_shape[2] - 1) - 1
        ) // stride[0] + 1
        W_out = (
            W_in + 2 * padding[1] - dilation[1] * (weight_shape[3] - 1) - 1
        ) // stride[1] + 1
        output_shape = [N, C_out, H_out, W_out]
        output_stride = (C_out * H_out * W_out, H_out * W_out, W_out, 1)
    else:
        #TODO: @goon - Conv3d
        raise ValueError(f"{n_conv_dims}D convolutions not supported")
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

    n_conv_dims = len(weight_spec.tensor_meta.shape) - 2
    output_shape, output_stride = _get_conv_output_shape_stride(
        n_conv_dims,
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
    grad_input_spec =   input_spec if _output_mask[0] else None
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
