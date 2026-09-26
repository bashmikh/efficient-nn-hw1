
import math

import numpy as np
import torch.nn as nn

from models import ConvNN

MODEL = ConvNN().eval()
_FP = 4


def _pair(v):
    return v if isinstance(v, tuple) else (v, v)


def _conv_out_hw(hw, kernel_size, stride, padding, dilation):
    out = []
    for size, k, s, p, d in zip(hw, _pair(kernel_size), _pair(stride),
                                _pair(padding), _pair(dilation)):
        out.append((size + 2 * p - d * (k - 1) - 1) // s + 1)
    return tuple(out)


def _out_shape(layer, in_shape):
    if isinstance(layer, nn.Conv2d):
        b, _, h, w = in_shape
        return (b, layer.out_channels,
                *_conv_out_hw((h, w), layer.kernel_size, layer.stride,
                              layer.padding, layer.dilation))

    if isinstance(layer, nn.MaxPool2d):
        b, c, h, w = in_shape
        return (b, c, *_conv_out_hw((h, w), layer.kernel_size, layer.stride,
                                    layer.padding, layer.dilation))

    if isinstance(layer, nn.AdaptiveAvgPool2d):
        b, c = in_shape[0], in_shape[1]
        return (b, c, *_pair(layer.output_size))

    if isinstance(layer, nn.Flatten):
        return (in_shape[0], math.prod(in_shape[1:]))

    if isinstance(layer, nn.Linear):
        return (*in_shape[:-1], layer.out_features)

    return in_shape


def _numel(shape):
    return math.prod(shape)


def _layer_params(layer):
    return (sum(p.numel() for p in layer.parameters(recurse=False))
            + sum(b.numel() for b in layer.buffers(recurse=False)
                  if b.dtype.is_floating_point))


def _walk(image_size, batch):
    s = np.asarray(image_size)
    b = np.asarray(batch)
    in_shape = (b, 3, s, s)
    for layer in MODEL.model:
        out_shape = _out_shape(layer, in_shape)
        yield layer, in_shape, out_shape
        in_shape = out_shape


def _is_alias(layer):
    return getattr(layer, "inplace", False) or isinstance(layer, nn.Flatten)


def _result(value):
    out = np.asarray(value, dtype=float)
    return float(out) if out.ndim == 0 else out


def _layer_flops(layer, in_shape, out_shape):
    if isinstance(layer, nn.Conv2d):
        kh, kw = _pair(layer.kernel_size)
        f = (2 * (layer.in_channels // layer.groups) * layer.out_channels
             * kh * kw * _numel(out_shape[2:]) * out_shape[0])
        if layer.bias is not None:
            f = f + _numel(out_shape)
        return f

    if isinstance(layer, nn.Linear):
        f = 2 * layer.in_features * layer.out_features * _numel(in_shape[:-1])
        if layer.bias is not None:
            f = f + _numel(out_shape)
        return f

    if isinstance(layer, nn.MaxPool2d):
        kh, kw = _pair(layer.kernel_size)
        return (kh * kw - 1) * _numel(out_shape)

    if isinstance(layer, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
        return _numel(in_shape)

    if isinstance(layer, nn.BatchNorm2d):
        return 2 * _numel(in_shape)

    if isinstance(layer, nn.ReLU):
        return _numel(in_shape)

    return 0 


def flops(image_size, batch) -> float:
    total = 0
    for layer, in_shape, out_shape in _walk(image_size, batch):
        total = total + _layer_flops(layer, in_shape, out_shape)
    return _result(total)


def _layer_bytes(layer, in_shape, out_shape):
    if isinstance(layer, nn.Flatten):
        return 0 
    return _FP * (_numel(in_shape) + _numel(out_shape) + _layer_params(layer))


def bytes_moved(image_size, batch) -> float:
    total = 0
    for layer, in_shape, out_shape in _walk(image_size, batch):
        total = total + _layer_bytes(layer, in_shape, out_shape)
    return _result(total)


def memory(image_size, batch) -> float:
    weights = _FP * sum(_layer_params(layer) for layer in MODEL.model)

    live = {}
    peak = 0
    cur_id, next_id = 0, 1

    for i, (layer, in_shape, out_shape) in enumerate(_walk(image_size, batch)):
        if i == 0:
            live[0] = _numel(in_shape)
            peak = live[0]

        if _is_alias(layer):
            candidate = sum(live.values())
        else:
            candidate = sum(live.values()) + _numel(out_shape)

        peak = np.maximum(peak, candidate)

        if not _is_alias(layer):
            if cur_id != 0:
                del live[cur_id]
            live[next_id] = _numel(out_shape)
            cur_id, next_id = next_id, next_id + 1

    return _result(_FP * peak + weights)


def latency(image_size, batch, theta: tuple[float, float, float]) -> float:
    """
    Latency = K * t_0 + sum_l max(F_l / V_c, M_l / V_m)

    theta = (t_0, V_c, V_m).

    """
    t_0, V_c, V_m = theta

    n_kernels = 0
    t = 0
    for layer, in_shape, out_shape in _walk(image_size, batch):
        if isinstance(layer, nn.Flatten):
            continue
        n_kernels += 1
        t = t + np.maximum(_layer_flops(layer, in_shape, out_shape) / V_c,
                           _layer_bytes(layer, in_shape, out_shape) / V_m)

    return _result(t + t_0 * n_kernels)


def energy(image_size, batch, theta_energy) -> float:
    """
    Energy = P_idle * Latency + e_flop * sum_l F_l + e_byte * sum_l M_l

    theta_energy = (p_idle [Вт], e_flop [Дж/FLOP], e_byte [Дж/байт], theta_latency).
    """
    p_idle, e_flop, e_byte, theta_latency = theta_energy

    e = (p_idle * latency(image_size, batch, theta_latency)
         + e_flop * flops(image_size, batch)
         + e_byte * bytes_moved(image_size, batch))
    return _result(e)
