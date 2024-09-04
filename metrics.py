import torch
import pandas as pd
import os
import numpy as np


def MSE(pred, target):
    """return mean square error
    Args:
        pred (Tensor): model output tensor of shape (bs, t, x1, ..., xd, v)
        target (Tensor): ground truth tensor of shape (bs, t, x1, ..., xd, v)

    Returns:
        res (Tensor): MSE with size (bs, v)
    """
    assert pred.shape == target.shape
    temp_shape = [0, len(pred.shape)-1]
    temp_shape.extend([i for i in range(1, len(pred.shape)-1)])
    pred = pred.permute(temp_shape) # (bs, t, x1, ..., xd, v) -> (bs, v, t, x1, ..., xd)
    target = target.permute(temp_shape) # (bs, t, x1, ..., xd, v) -> (bs, v, t, x1, ..., xd)
    nb, nc = pred.shape[0], pred.shape[1]
    errors = pred.reshape([nb, nc, -1]) - target.reshape([nb, nc, -1]) # (bs, v, t*x1*x2*...*xd)
    res = torch.mean(errors**2, dim=2)
    return res # (bs, v)


def RMSE(pred, target):
    """return root mean square error
    Args:
        pred (Tensor): model output tensor of shape (bs, t, x1, ..., xd, v)
        target (Tensor): ground truth tensor of shape (bs, t, x1, ..., xd, v)

    Returns:
        res (Tensor): RMSE with size (bs, v)
    """
    return torch.sqrt(MSE(pred, target)) # (bs, v)


def L2RE(pred, target):
    """return l2 relative error (nMSE in PDEBench)
    Args:
        pred (Tensor): model output tensor of shape (bs, t, x1, ..., xd, v)
        target (Tensor): ground truth tensor of shape (bs, t, x1, ..., xd, v)

    Returns:
        res (Tensor): L2RE with size (bs, v)
    """
    assert pred.shape == target.shape
    temp_shape = [0, len(pred.shape)-1]
    temp_shape.extend([i for i in range(1, len(pred.shape)-1)])
    pred = pred.permute(temp_shape) # (bs, t, x1, ..., xd, v) -> (bs, v, t, x1, ..., xd)
    target = target.permute(temp_shape) # (bs, t, x1, ..., xd, v) -> (bs, v, t, x1, ..., xd)
    nb, nc = pred.shape[0], pred.shape[1]
    errors = pred.reshape([nb, nc, -1]) - target.reshape([nb, nc, -1]) # (bs, v, t*x1*x2*...*xd)
    res = torch.sum(errors**2, dim=2) / torch.sum(target.reshape([nb, nc, -1])**2, dim=2)
    return torch.sqrt(res) # (bs, v)


def MaxError(pred, target):
    """return max error in a batch
    Args:
        pred (Tensor): model output tensor of shape (bs, t, x1, ..., xd, v)
        target (Tensor): ground truth tensor of shape (bs, t, x1, ..., xd, v)

    Returns:
        res (Tensor): Max error with size (bs, v)
    """
    assert pred.shape == target.shape
    errors = torch.abs(pred - target)
    nc = errors.shape[-1]
    res, _ = torch.max(errors.reshape([-1, nc]), dim=0) # retain the last dim
    return res # (v)


def NMSE(pred, target):
    """return normalized MSE in a batch
    Args:
        pred (Tensor): model output tensor of shape (bs, t, x1, ..., xd, v)
        target (Tensor): ground truth tensor of shape (bs, t, x1, ..., xd, v)

    Returns:
        res (Tensor): NMSE with size (bs, v)
    """
    assert pred.shape == target.shape
    temp_shape = [0, len(pred.shape)-1]
    temp_shape.extend([i for i in range(1, len(pred.shape)-1)])
    pred = pred.permute(temp_shape) # (bs, t, x1, ..., xd, v) -> (bs, v, t, x1, ..., xd)
    target = target.permute(temp_shape) # (bs, t, x1, ..., xd, v) -> (bs, v, t, x1, ..., xd)
    nb, nc = pred.shape[0], pred.shape[1]
    errors = pred.reshape([nb, nc, -1]) - target.reshape([nb, nc, -1]) # (bs, v, t*x1*x2*...*xd)
    norm = target.reshape([nb, nc, -1])
    res = torch.sum(errors**2, dim=2) / torch.sum(norm**2, dim=2)
    return res

def MAE(pred, target):
    """return mean absolute error in a batch
    Args:
        pred (Tensor): model output tensor of shape (bs, t, x1, ..., xd, v)
        target (Tensor): ground truth tensor of shape (bs, t, x1, ..., xd, v)

    Returns:
        res (Tensor): NMSE with size (bs, v)
    """
    assert pred.shape == target.shape
    temp_shape = [0, len(pred.shape)-1]
    temp_shape.extend([i for i in range(1, len(pred.shape)-1)])
    pred = pred.permute(temp_shape) # (bs, t, x1, ..., xd, v) -> (bs, v, t, x1, ..., xd)
    target = target.permute(temp_shape) # (bs, t, x1, ..., xd, v) -> (bs, v, t, x1, ..., xd)
    nb, nc = pred.shape[0], pred.shape[1]
    errors = pred.reshape([nb, nc, -1]) - target.reshape([nb, nc, -1]) # (bs, v, t*x1*x2*...*xd)
    return torch.mean(errors.abs(),dim=-1) # (bs, v)