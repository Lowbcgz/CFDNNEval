import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
import time
import os
from functools import partial
from torch.optim.lr_scheduler import StepLR, OneCycleLR
import yaml
from tensorboardX import SummaryWriter

from model.oformer.encoder_module import IrregSTEncoder2D
from model.oformer.decoder_module import IrregSTDecoder2D


from utils import load_checkpoint, save_checkpoint, ensure_dir, setup_seed, get_model, get_dataset, get_dataloader
import logging
from metrics import MSE, RMSE, MAE

# set flags / seeds
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.multiprocessing.set_sharing_strategy('file_system')
torch.autograd.set_detect_anomaly(True)


def build_model(flow_name, input_ch, output_ch=2):
    encoder = IrregSTEncoder2D(
        input_channels=input_ch,    # vx, vy, prs, dns, pos_x, pos_y
        time_window=1,
        in_emb_dim=128,
        out_chanels=128,
        max_node_type=3,
        heads=1,
        depth=4,
        res=200,
        use_ln=True,
        emb_dropout=0.0,
    )

    decoder = IrregSTDecoder2D(
        max_node_type=3,
        latent_channels=128,
        out_channels=output_ch,  # vx, vy, prs, dns
        res=200,
        scale=2,
        dropout=0.1
    )

    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad) +\
                      sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')
    return encoder, decoder


def pointwise_rel_loss(x, y, p=2):
    #   x, y [b, t, n, c]
    # print(x.shape, y.shape)
    assert x.shape == y.shape
    eps = 1e-5
    if p == 1:
        y_norm = 1.#y.abs() + eps
        diff = (x-y).abs()
    else:
        y_norm = 1.#y.pow(p) + eps
        diff = (x-y).pow(p)
    diff = diff / y_norm   # [b, c]
    diff = diff.sum(dim=-1)  # sum over channels
    diff = diff.mean(dim=(1, 2))  # sum over time
    diff = diff.mean()
    return diff

def rmse_loss(y, y_gt):
    loss_fn = nn.MSELoss()
    mse = loss_fn(y, y_gt)
    return torch.sqrt(mse)


def get_arguments(parser):
    parser.add_argument(
        "--config_file", type=str, help="Path to config file"
    )
    # basic training settings
    parser.add_argument(
        '--lr', type=float, default=3e-4, help='Specifies learing rate for optimizer. (default: 1e-3)'
    )
    parser.add_argument(
        '--resume', action='store_true', help='If set resumes training from provided checkpoint. (default: None)'
    )
    parser.add_argument(
        '--path_to_resume', type=str, default='', help='Path to checkpoint to resume training. (default: "")'
    )
    parser.add_argument(
        '--iters', type=int, default=100000, help='Number of training iterations. (default: 100k)'
    )
    parser.add_argument(
        '--log_dir', type=str, default='./', help='Path to log, save checkpoints. '
    )
    parser.add_argument(
        '--ckpt_every', type=int, default=5000, help='Save model checkpoints every x iterations. (default: 5k)'
    )

    # ===================================
    # for dataset
    parser.add_argument(
        '--batch_size', type=int, default=16, help='Size of each batch (default: 16)'
    )
    # parser.add_argument(
    #     '--train_dataset_path', type=str, required=True, help='Path to training dataset.'
    # )
    # parser.add_argument(
    #     '--test_dataset_path', type=str, required=True, help='Path to testing dataset.'
    # )
    parser.add_argument(
        '--flow_name', type=str, required=True, help='The flow name.'
    )
    parser.add_argument(
        '--case_name', type=str, required=True, help='The subset of flow.'
    )
    parser.add_argument(
        '--curriculum_steps', type=int, default=1, help='at initial stage, dont rollout too long'
    )
    parser.add_argument(
        '--curriculum_ratio', type=float, default=0.2, help='how long is the initial stage?'
    )

    return parser


# Start with main code
if __name__ == '__main__':
    setup_seed(0)
    # argparse for additional flags for experiment
    parser = argparse.ArgumentParser(
        description="Train a PDE transformer")
    parser = get_arguments(parser)
    opt = parser.parse_args()
    print('Using following options')
    print(opt)

    with open(opt.config_file, 'r') as f:
        args = yaml.safe_load(f)

    # if running on GPU and we want to use cuda move model there
    use_cuda = torch.cuda.is_available()
    print(f'Using cuda: {use_cuda}')

    # add code for datasets
    print('Preparing the data')
    flow_name = opt.flow_name
    case_name = opt.case_name
    dataset_args = args["dataset"]
    # data get dataloader
    train_data, val_data, test_data, test_ms_data = get_dataset(args)
    train_dataloader, test_dataloader, test_loader, test_ms_loader = get_dataloader(train_data, val_data, test_data, test_ms_data, args)

    # set some train args
    inputx, outputy, _, case_params, _, _, = next(iter(test_dataloader))
    input_chs = inputx.shape[-1] + case_params.shape[-1]
    output_chs = outputy.shape[-1]
    n_tolx = inputx.shape[1]*inputx.shape[2]

    # instantiate network
    print('Building network')
    encoder, decoder = build_model(flow_name, input_chs, output_chs)
    if use_cuda:
        encoder, decoder = encoder.cuda(), decoder.cuda()

    # typically we use tensorboardX to keep track of experiments
    writer = SummaryWriter()
    checkpoint_dir = os.path.join(opt.log_dir, f'{case_name}_model_ckpt')
    ensure_dir(checkpoint_dir)

    sample_dir = os.path.join(opt.log_dir, 'samples')
    ensure_dir(sample_dir)

    # save option information to the disk
    logger = logging.getLogger("LOG")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s_%s_%s.txt' % (opt.log_dir, 'logging_info', flow_name, case_name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('=======Option used=======')
    for arg in vars(opt):
        logger.info(f'{arg}: {getattr(opt, arg)}')


    # create optimizers
    enc_optim = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=opt.lr,
                                  weight_decay=1e-4)
    enc_scheduler = OneCycleLR(enc_optim, max_lr=opt.lr, total_steps=opt.iters,
                               div_factor=1e4,
                               pct_start=0.3,
                               final_div_factor=1e4,
                               )

    # load checkpoint if needed/ wanted
    start_n_iter = 0
    if opt.resume:
        print(f'Resuming checkpoint from: {opt.path_to_resume}')
        ckpt = load_checkpoint(opt.path_to_resume)  # custom method for loading last checkpoint
        encoder.load_state_dict(ckpt['encoder'])
        decoder.load_state_dict(ckpt['decoder'])
        start_n_iter = ckpt['n_iter']

        enc_optim.load_state_dict(ckpt['enc_optim'])

        enc_scheduler.load_state_dict(ckpt['enc_sched'])
        print("last checkpoint restored")

    # now we start the main loop
    n_iter = start_n_iter

    # for loop going through dataset
    with tqdm(total=opt.iters) as pbar:
        pbar.update(n_iter)
        train_data_iter = iter(train_dataloader)

        while True:

            encoder.train()
            decoder.train()
            start_time = time.time()

            try:
                data = next(train_data_iter)
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader
                del train_data_iter
                train_data_iter = iter(train_dataloader)
                data = next(train_data_iter)

            # data preparation
            # x, y, node_type, pos = data
            x, y, node_type, case_params, pos, _ = data
            # print(x.shape, y.shape, node_type.shape, case_params.shape, pos.shape)
            
            x = x.reshape(-1, 1, n_tolx, x.shape[-1])
            y = y.reshape(-1, 1, n_tolx, y.shape[-1])
            node_type = node_type.long().reshape(-1, 1, n_tolx, node_type.shape[-1])
            case_params = case_params.reshape(-1, 1, n_tolx, case_params.shape[-1])
            pos = pos.reshape(-1, n_tolx, pos.shape[-1])

            if use_cuda:
                x, y, node_type, case_params, pos = x.cuda(), y.cuda(), node_type.cuda(), case_params.cuda(), pos.cuda()
            input_pos = prop_pos = pos

            fx = torch.cat((x, case_params), dim=-1).detach()
            z = encoder.forward(fx, node_type[:,0,...].detach(), input_pos)
            pred = decoder.forward(z, prop_pos, node_type[:,0,...].detach(), 1, input_pos)
            # print(pred.shape, y.shape)
            all_loss = pointwise_rel_loss(pred, y, p=2)
            loss = all_loss
            enc_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 2.0)
            # Unscales gradients and calls
            enc_optim.step()

            enc_scheduler.step()
            # with torch.no_grad():
            #     x_out = torch.cat((pot, field_x, field_y), dim=-1)
            #     pot_los, field_loss = mse_loss(x_out, y)

            # udpate tensorboardX
            writer.add_scalar('train_loss', loss, n_iter)

            pbar.set_description(
                f'Total loss (1e-4): {loss.item()*1e4:.1f}||' 
                f'All loss: {all_loss.item()*1e4:.1f}||'
                f'lr: {enc_optim.param_groups[0]["lr"]:.1e}|| '
                f'Seq len {y.shape[1]}||')

            pbar.update(1)
            start_time = time.time()
            n_iter += 1

            if (n_iter-1) % opt.ckpt_every == 0 or n_iter >= opt.iters:
                logger.info('Tesing')
                print('Testing')

                encoder.eval()
                decoder.eval()

                all_mse  = []
                all_rmse = []
                all_mae  = []
                visualization_cache = {
                    'pred': [],
                    'gt': [],
                    'coords': [],
                    'cells': [],
                }
                picked = 0
                for j, data in enumerate(tqdm(test_dataloader)):
                    # data preparation
                    # x, y, x_mean, x_std = data

                    # data preparation
                    x, y, node_type, case_params, pos, _ = data 
                    # print(x.shape, y.shape, node_type.shape, case_params.shape, pos.shape)
                    x = x.reshape(-1, 1, n_tolx, x.shape[-1])
                    y = y.reshape(-1, 1, n_tolx, y.shape[-1])
                    node_type = node_type.long().reshape(-1, 1, n_tolx, node_type.shape[-1])
                    case_params = case_params.reshape(-1, 1, n_tolx, case_params.shape[-1])
                    pos = pos.reshape(-1, n_tolx, pos.shape[-1])

                    if use_cuda:
                        x, y, node_type, case_params, pos = x.cuda(), y.cuda(), node_type.cuda(), case_params.cuda(), pos.cuda() 

                    input_pos = prop_pos = pos
                    with torch.no_grad():
                        fx = torch.cat((x, case_params), dim=-1).detach()
                        z = encoder.forward(fx, node_type[:,0,...].detach(), input_pos)
                        pred = decoder.forward(z, prop_pos, node_type[:,0,...].detach(), 1, input_pos)
                        _mse = MSE(pred, y)[0]
                        all_mse.append(torch.mean(_mse).item())
                        _rmse = RMSE(pred, y)[0]
                        all_rmse.append(torch.mean(_rmse).item())
                        _mae = MAE(pred, y)[0]
                        all_mae.append(torch.mean(_mae).item())

                #     if picked < 8:
                #         idx = np.arange(0, min(8 - picked, y.shape[0]))
                #         # randomly pick a batch
                #         interv = y.shape[1] // 8
                #         y = y[idx, ::interv]
                #         pred = pred[idx, ::interv]
                #         pos = pos[idx]
                #         cells = cells[idx]

                #         visualization_cache['gt'].append(y)
                #         visualization_cache['pred'].append(pred)
                #         visualization_cache['coords'].append(pos)
                #         visualization_cache['cells'].append(cells)
                #         picked += y.shape[0]

                # gt = torch.cat(visualization_cache['gt'], dim=0)
                # pred = torch.cat(visualization_cache['pred'], dim=0)
                # coords = torch.cat(visualization_cache['coords'], dim=0)
                # cells = torch.cat(visualization_cache['cells'], dim=0)


                #
                # del visualization_cache
                writer.add_scalar('testing avg loss', np.mean(all_mse), global_step=n_iter)

                print(f'Testing avg mse (1e-3): {np.mean(all_mse)*1e3}')
                print(f'Testing avg rmse (1e-3): {np.mean(all_rmse)*1e3}')
                print(f'Testing avg mae (1e-3): {np.mean(all_mae)*1e3}')

                logger.info(f'Current iteration: {n_iter}')
                logger.info(f'Testing avg loss (1e-3): {np.mean(all_mse)*1e3}')
                logger.info(f'Testing avg rmse (1e-3): {np.mean(all_rmse)*1e3}')
                logger.info(f'Testing avg mae (1e-3): {np.mean(all_mae)*1e3}')

                # save checkpoint if needed
                ckpt = {
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(),
                    'n_iter': n_iter,
                    'enc_optim': enc_optim.state_dict(),
                    'enc_sched': enc_scheduler.state_dict(),
                }

                save_checkpoint(ckpt, os.path.join(checkpoint_dir, f'model_checkpoint{n_iter}.ckpt'))
                del ckpt
                if n_iter >= opt.iters:
                    break