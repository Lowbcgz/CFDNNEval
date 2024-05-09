import torch
import pandas as pd
import os
import numpy as np


def MSE(pred, target):
    """return mean square error

    pred: model output tensor of shape (bs, x1, ..., xd, t, v)
    target: ground truth tensor of shape (bs, x1, ..., xd, t, v)
    """
    assert pred.shape == target.shape
    temp_shape = [0, len(pred.shape)-1]
    temp_shape.extend([i for i in range(1, len(pred.shape)-1)])
    pred = pred.permute(temp_shape) # (bs, x1, ..., xd, v) -> (bs, v, x1, ..., xd)
    target = target.permute(temp_shape) # (bs, x1, ..., xd, v) -> (bs, v, x1, ..., xd)
    nb, nc = pred.shape[0], pred.shape[1]
    errors = pred.reshape([nb, nc, -1]) - target.reshape([nb, nc, -1]) # (bs, v, x1*x2*...*xd*t)
    res = torch.mean(errors**2, dim=2)
    return res # (bs, v)


def RMSE(pred, target):
    """return root mean square error

    pred: model output tensor of shape (bs, x1, ..., xd, t, v)
    target: ground truth tensor of shape (bs, x1, ..., xd, t, v)
    """
    return torch.sqrt(MSE(pred, target)) # (bs, v)


def L2RE(pred, target):
    """l2 relative error (nMSE in PDEBench)

    pred: model output tensor of shape (bs, x1, ..., xd, t, v)
    target: ground truth tensor of shape (bs, x1, ..., xd, t, v)
    """
    assert pred.shape == target.shape
    temp_shape = [0, len(pred.shape)-1]
    temp_shape.extend([i for i in range(1, len(pred.shape)-1)])
    pred = pred.permute(temp_shape) # (bs, x1, ..., xd, t, v) -> (bs, v, x1, ..., xd, t)
    target = target.permute(temp_shape) # (bs, x1, ..., xd, t, v) -> (bs, v, x1, ..., xd, t)
    nb, nc = pred.shape[0], pred.shape[1]
    errors = pred.reshape([nb, nc, -1]) - target.reshape([nb, nc, -1]) # (bs, v, x1*x2*...*xd*t)
    res = torch.sum(errors**2, dim=2) / torch.sum(target.reshape([nb, nc, -1])**2, dim=2)
    return torch.sqrt(res) # (bs, v)

def MaxError(pred, target):
    """return max error in a batch

    pred: model output tensor of shape (bs, x1, ..., xd, t, v)
    target: ground truth tensor of shape (bs, x1, ..., xd, t, v)
    """
    errors = torch.abs(pred - target)
    nc = errors.shape[-1]
    res, _ = torch.max(errors.reshape([-1, nc]), dim=0) # retain the last dim
    return res # (v)

def NMSE(pred, target):
    assert pred.shape == target.shape
    temp_shape = [0, len(pred.shape)-1]
    temp_shape.extend([i for i in range(1, len(pred.shape)-1)])
    pred = pred.permute(temp_shape) # (bs, x1, ..., xd, v) -> (bs, v, x1, ..., xd)
    target = target.permute(temp_shape) # (bs, x1, ..., xd, v) -> (bs, v, x1, ..., xd)
    nb, nc = pred.shape[0], pred.shape[1]
    errors = pred.reshape([nb, nc, -1]) - target.reshape([nb, nc, -1]) # (bs, v, x1*x2*...*xd*t)
    norm = pred.reshape([nb, nc, -1])
    res = torch.sum(errors**2, dim=2) / torch.sum(norm**2, dim=2)
    return res

def MAE(pred, target):
    # mean absolute error
    assert pred.shape == target.shape
    temp_shape = [0, len(pred.shape)-1]
    temp_shape.extend([i for i in range(1, len(pred.shape)-1)])
    pred = pred.permute(temp_shape) # (bs, x1, ..., xd, v) -> (bs, v, x1, ..., xd)
    target = target.permute(temp_shape) # (bs, x1, ..., xd, v) -> (bs, v, x1, ..., xd)
    nb, nc = pred.shape[0], pred.shape[1]
    errors = pred.reshape([nb, nc, -1]) - target.reshape([nb, nc, -1]) # (bs, v, x1*x2*...*xd*t)
    return torch.mean(errors.abs(),dim=-1) # (bs, v)

def print_res(res):
    for u, v in res.items():
        dim = len(v)
        if dim == 1:
            print(u, "{0:.6f}".format(v.item()))
        else:
            for i in range(dim):
                if i == 0:
                    print(u, "\t{0:.6f}".format(v[i].item()), end='\t')
                else:
                    print("{0:.6f}".format(v[i].item()), end='\t')
            print("")
    return

def write_res(res, filename, tag, append=True):
    df = pd.DataFrame()
    # Iterate over the metrics in res
    for metric, values in res.items():
        dim = len(values)
        values = [x.item() for x in values]
        if "field" not in df.columns:
            if dim > 1:
                df.insert(0, 'field', [tag+"_x"+str(k) for k in range(dim)]+[tag + "_mean"])
            else:
                df.insert(0, 'field', [tag+"_x"+str(k) for k in range(dim)])
        if dim > 1:
            df[metric] = [*values]+ [np.mean(values)]
        else:
            df[metric] = [*values]
    
    if append:
        if os.path.exists(filename):
            df.to_csv(filename, mode='a', header=False, index=False)
        else:
            df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, index=False)