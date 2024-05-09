from typing import Union, Optional
import matplotlib.pyplot as plt
import typing
from pathlib import Path
import torch
from torch import Tensor

@typing.no_type_check
def plot_predictions(
    label: Tensor,
    pred: Tensor,
    out_dir: Path,
    step: int,
    inp: Optional[
        Tensor
    ] = None,  # non-autoregressive input func. is not plottable.
):
    assert all([isinstance(x, Tensor) for x in [label, pred]])
    assert label.shape == pred.shape, f"{label.shape}, {pred.shape}"

    if inp is not None:
        assert inp.shape == label.shape
        assert isinstance(inp, Tensor)
        inp_dir = out_dir / "input"
        inp_dir.mkdir(exist_ok=True, parents=True)
        inp_arr = inp.permute(0, 3, 1, 2)[0][0].cpu().detach().numpy()
    label_dir = out_dir / "label"
    label_dir.mkdir(exist_ok=True, parents=True)
    pred_dir = out_dir / "pred"
    pred_dir.mkdir(exist_ok=True, parents=True)

    pred_arr = pred.permute(0, 3, 1, 2)[0][0].cpu().detach().numpy()
    label_arr = label.permute(0, 3, 1, 2)[0][0].cpu().detach().numpy()

    # Plot and save images
    if inp is not None:
        u_min = min(inp_arr.min(), pred_arr.min(), label_arr.min())  # type: ignore  # noqa
        u_max = max(inp_arr.max(), pred_arr.max(), label_arr.max())  # type: ignore  # noqa
    else:
        u_min = min(pred_arr.min(), label_arr.min())  # type: ignore  # noqa
        u_max = max(pred_arr.max(), label_arr.max())  # type: ignore  # noqa

    if inp is not None:
        plt.axis("off")
        plt.imshow(
            inp_arr, vmin=inp_arr.min(), vmax=inp_arr.max(), cmap="coolwarm"  # type: ignore  # noqa
        )
        plt.savefig(
            inp_dir / f"{step:04}.png", bbox_inches="tight", pad_inches=0  # type: ignore  # noqa
        )
        plt.clf()

    plt.axis("off")
    plt.imshow(
        label_arr, vmin=label_arr.min(), vmax=label_arr.max(), cmap="coolwarm"
    )
    plt.savefig(
        label_dir / f"{step:04}.png", bbox_inches="tight", pad_inches=0
    )
    plt.clf()

    plt.axis("off")
    plt.imshow(
        pred_arr, vmin=pred_arr.min(), vmax=pred_arr.max(), cmap="coolwarm"
    )
    plt.savefig(pred_dir / f"{step:04}.png", bbox_inches="tight", pad_inches=0)
    plt.clf()

def plot_loss(losses, out: Path, fontsize: int = 12, linewidth: int = 2):
    plt.plot(losses, linewidth=linewidth)
    plt.xlabel("Step", fontsize=fontsize)
    plt.ylabel("Loss", fontsize=fontsize)
    plt.savefig(out)
    plt.clf()
    plt.close()

def plot_stream_line(pred, label, grid, out_dir: Path, step: int):
    label_dir = out_dir / "label"
    label_dir.mkdir(exist_ok=True, parents=True)
    pred_dir = out_dir / "pred"
    pred_dir.mkdir(exist_ok=True, parents=True)

    pred, label, grid = pred[0].detach().cpu().numpy(), label[0].detach().cpu().numpy(), grid[0].detach().cpu().numpy()
    xx, yy = grid[..., 1], grid[..., 0]
    
    plt.figure(figsize=(6, 6))
    u, v = pred[...,0], pred[...,1]
    plt.axis("off")
    plt.gca().set_aspect('equal')
    plt.streamplot(xx, yy, u, v, density = 1.0, linewidth=0.5, arrowsize=1, color='k',arrowstyle='->')
    plt.savefig(pred_dir / f"{step:04}.png", bbox_inches="tight", pad_inches=0)
    plt.clf()
    plt.close()

    plt.figure(figsize=(6, 6))
    u, v = label[...,0], label[...,1]
    plt.axis("off")
    plt.gca().set_aspect('equal')
    plt.streamplot(xx, yy, u, v, density = 1.0, linewidth=0.5, arrowsize=1, color='k',arrowstyle='->')
    plt.savefig(label_dir / f"{step:04}.png", bbox_inches="tight", pad_inches=0)
    plt.clf()
    plt.close()