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
    cw_res = torch.mean(errors**2, dim=2)
    sw_res = torch.mean(cw_res, dim=1)
    return cw_res, sw_res # channel-wise: (bs, v) and sample-wise: (bs,)


def RMSE(pred, target):
    """return root mean square error

    pred: model output tensor of shape (bs, x1, ..., xd, t, v)
    target: ground truth tensor of shape (bs, x1, ..., xd, t, v)
    """
    cw_res, sw_res = MSE(pred, target)
    return torch.sqrt(cw_res), torch.sqrt(sw_res) # channel-wise: (bs, v) and sample-wise: (bs,)


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
    cw_res = torch.sum(errors**2, dim=2) / torch.sum(target.reshape([nb, nc, -1])**2, dim=2)
    sw_res = torch.sum(errors.reshape([nb,-1])**2, dim=1) / torch.sum(target.reshape([nb, -1])**2, dim=1)
    return torch.sqrt(cw_res), torch.sqrt(sw_res) # channel-wise: (bs, v) and sample-wise: (bs,)

def MaxError(pred, target):
    """return max error in a batch

    pred: model output tensor of shape (bs, x1, ..., xd, t, v)
    target: ground truth tensor of shape (bs, x1, ..., xd, t, v)
    """
    errors = torch.abs(pred - target)
    nc = errors.shape[-1]
    res, _ = torch.max(errors.reshape([-1, nc]), dim=0) # retain the last dim
    max_res = torch.max(res)
    return res, max_res  # (v), () 

def NMSE(pred, target):
    assert pred.shape == target.shape
    temp_shape = [0, len(pred.shape)-1]
    temp_shape.extend([i for i in range(1, len(pred.shape)-1)])
    pred = pred.permute(temp_shape) # (bs, x1, ..., xd, v) -> (bs, v, x1, ..., xd)
    target = target.permute(temp_shape) # (bs, x1, ..., xd, v) -> (bs, v, x1, ..., xd)
    nb, nc = pred.shape[0], pred.shape[1]
    errors = pred.reshape([nb, nc, -1]) - target.reshape([nb, nc, -1]) # (bs, v, x1*x2*...*xd*t)
    cw_res = torch.sum(errors**2, dim=2) / torch.sum(target.reshape([nb, nc, -1])**2, dim=2)
    sw_res = torch.sum(errors.reshape([nb,-1])**2, dim=1) / torch.sum(target.reshape([nb, -1])**2, dim=1)
    return cw_res, sw_res # channel-wise: (bs, v) and sample-wise: (bs,)

def MAE(pred, target):
    # mean absolute error
    assert pred.shape == target.shape
    temp_shape = [0, len(pred.shape)-1]
    temp_shape.extend([i for i in range(1, len(pred.shape)-1)])
    pred = pred.permute(temp_shape) # (bs, x1, ..., xd, v) -> (bs, v, x1, ..., xd)
    target = target.permute(temp_shape) # (bs, x1, ..., xd, v) -> (bs, v, x1, ..., xd)
    nb, nc = pred.shape[0], pred.shape[1]
    errors = pred.reshape([nb, nc, -1]) - target.reshape([nb, nc, -1]) # (bs, v, x1*x2*...*xd*t)
    return torch.mean(errors.abs(),dim=-1), torch.mean(errors.reshape([nb,-1]).abs(), dim=-1) # channel-wise: (bs, v) and sample-wise: (bs,)

def print_res(res: dict):
    cw_res= res.get("cw_res") # channel-wise
    sw_res = res.get("sw_res", None) # sample-wise
    for u, v in cw_res.items():
        dim = len(v)
        if dim == 1:
            print(u, "{0:.6f}".format(v.item()))
        else:
            for i in range(dim):
                if i == 0:
                    print(u, "\t{0:.6f}".format(v[i].item()), end='\t')
                else:
                    print("{0:.6f}".format(v[i].item()), end='\t')
            print("mean: {0:.6f}".format(v.mean().item()), end='\t')
            if sw_res:
                print("global: {0:.6f}".format(sw_res[u].item()))
            else:
                print()
    return

def write_res(res, filename, tag, append=True):
    df = pd.DataFrame()
    metric, values = next(iter(res["cw_res"].items()))
    dim = len(values)
    # Iterate over the metrics in res
    for metric, values in res["cw_res"].items():
        values = [x.item() for x in values]
        if "field" not in df.columns:
            if dim > 1:
                df.insert(0, 'field', [tag+"_x"+str(k) for k in range(dim)]+[tag + "_mean"]+ [tag+"_global"])
            else:
                df.insert(0, 'field', [tag+"_x"+str(k) for k in range(dim)])
        
        if dim > 1:
            df[metric] = [*values]+ [np.mean(values)]+ [res["sw_res"][metric].item()]
        else:
            df[metric] = [*values]
    # add mean inference time
    if "Mean inference time" in res:
        df["Mean inference time"] = [res["Mean inference time"]]*len(df)

    if append:
        if os.path.exists(filename):
            df.to_csv(filename, mode='a', header=False, index=False)
        else:
            df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, index=False)