import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import yaml
import argparse
import metrics
import tqdm
from functools import reduce
from functools import partial
from torch.utils.data import DataLoader
from timeit import default_timer
from utils import *
from visualize import *
from uno import UNO1d, UNO2d, UNO3d
from dataset import *


def get_dataset(args):
    dataset_args = args["dataset"]
    if(args["flow_name"] in ["tube"]):
        train_data = TubeDataset(filename=args['flow_name'] + '_train.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                delta_time=dataset_args['delta_time'],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                norm_bc = dataset_args['norm_bc'],
                                multi_step_size= dataset_args['multi_step_size']
                                )
        val_data = TubeDataset(filename=args['flow_name'] + '_dev.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                delta_time=dataset_args['delta_time'],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                norm_bc = dataset_args['norm_bc'],
                                multi_step_size= dataset_args['multi_step_size']
                                )
        test_data = TubeDataset(filename=args['flow_name'] + '_test.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                delta_time=dataset_args['delta_time'],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                norm_bc = dataset_args['norm_bc']
                                )
    elif args["flow_name"] == "NSCH":
        train_data = NSCHDataset(
                                filename='train.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                )
        val_data = NSCHDataset(
                                filename='val.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                )
        test_data = NSCHDataset(
                                filename='test.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                stable_state_diff = dataset_args['stable_state_diff'],
                                norm_props = dataset_args['norm_props'],
                                )
    elif args['flow_name'] == 'Darcy':
        train_data = DarcyDataset(
                                filename=args['flow_name'] + '_train.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                )
        val_data = DarcyDataset(
                                filename=args['flow_name'] + '_dev.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                )
        test_data = DarcyDataset(
                                filename=args['flow_name'] + '_test.hdf5',
                                saved_folder=dataset_args['saved_folder'],
                                case_name=dataset_args['case_name'],
                                reduced_resolution=dataset_args["reduced_resolution"],
                                reduced_batch=dataset_args["reduced_batch"],
                                )
    return train_data, val_data, test_data

def get_dataloader(train_data, val_data, test_data, args):
    dataloader_args = args["dataloader"]
    train_loader = DataLoader(train_data, shuffle=True, multiprocessing_context = 'spawn', generator=torch.Generator(device = 'cpu'), 
                              batch_size=dataloader_args['train_batch_size'], 
                              num_workers= dataloader_args['num_workers'], pin_memory=dataloader_args['pin_memory'])
    val_loader = DataLoader(val_data, shuffle=False, multiprocessing_context = 'spawn', generator=torch.Generator(device = 'cpu'), 
                            batch_size=dataloader_args['val_batch_size'],
                            num_workers= dataloader_args['num_workers'], pin_memory=dataloader_args['pin_memory'])
    test_loader = DataLoader(test_data, shuffle=False, drop_last=True,
                            batch_size=dataloader_args['test_batch_size'],
                            num_workers= dataloader_args['num_workers'], pin_memory=dataloader_args['pin_memory'])
    
    return train_loader, val_loader, test_loader

def get_model(spatial_dim, n_case_params, args):
    assert spatial_dim <= 3, "Spatial dimension of data can not exceed 3."

    model_args = args["model"]
    if args['flow_name'] in ["Darcy"]:
        if spatial_dim == 2:
            model = UNO2d(num_channels=model_args["input_channels"],
                          width=model_args["width"],
                          n_case_params = n_case_params,
                          output_channels=model_args["output_channels"])
        else:
            #TODO
            pass
    else:
        if spatial_dim == 1:
            model = UNO1d(num_channels=model_args["num_channels"],
                        width=model_args["width"],
                        n_case_params = n_case_params)
        elif spatial_dim == 2:
            model = UNO2d(num_channels=model_args['num_channels'],
                        width = model_args['width'],
                        n_case_params = n_case_params)
        elif spatial_dim == 3:
            model = UNO3d(num_channels=model_args["num_channels"],
                        width = model_args['width'],
                        n_case_params = n_case_params)
    
    return model

def train_loop(model, train_loader, optimizer, loss_fn, device, args):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    train_l_inf = 0
    step = 0
    for x, y, mask, case_params, grid, _ in train_loader:
        step += 1
        # batch_size = x.size(0)
        loss = 0
        x = x.to(device) # x: input tensor (The previous time step) [b, x1, ..., xd, v]
        y = y.to(device) # y: target tensor (The latter time step) [b, x1, ..., xd, v]
        grid = grid.to(device) # grid: meshgrid [b, x1, ..., xd, dims]
        mask = mask.to(device) # mask [b, x1, ..., xd, 1]
        case_params = case_params.to(device) #parameters [b, x1, ..., xd, p]
        y = y * mask

        if args["training_type"] in ['autoregressive']:
            if train_loader.dataset.multi_step_size ==1:
                #Model run one_step
                if case_params.shape[-1] == 0: #darcy
                    case_params = case_params.reshape(0)
                pred = model(x, case_params, mask, grid)
                # Loss calculation
                _batch = pred.size(0)
                loss = loss_fn(pred.reshape(_batch, -1), y.reshape(_batch, -1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_l2 += loss.item()
                train_l_inf = max(train_l_inf, torch.max((torch.abs(pred.reshape(_batch, -1) - y.reshape(_batch, -1)))))
            else:
                # Autoregressive loop
                preds=[]
                for i in range(train_loader.dataset.multi_step_size):
                    pred = model(x, case_params, mask[:,i], grid)
                    preds.append(pred)
                    x = pred
                preds=torch.stack(preds, dim=1)
                _batch = preds.size(0)
                loss = loss_fn(preds.reshape(_batch, -1), y.reshape(_batch, -1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_l2 += loss.item()
                train_l_inf = max(train_l_inf, torch.max((torch.abs(preds.reshape(_batch, -1) - y.reshape(_batch, -1)))))
    train_l2 /= step      
    t2 = default_timer()
    return train_l2, train_l_inf, t2 - t1

def val_loop(val_loader, model, loss_fn, device, training_type, output_dir, epoch, metric_names=['MSE', 'RMSE', 'L2RE', 'MaxError', 'NMSE', 'MAE'], plot_interval = 1):
    model.eval()
    val_l2 = 0
    val_l_inf = 0
    step = 0
    res_dict = {}
    for name in metric_names:
        res_dict[name] = []
    
    ckpt_dir = output_dir + f"/ckpt-{epoch}"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    with torch.no_grad():
        for x, y, mask, case_params, grid, case_id in val_loader:
            step += 1
            # batch_size = x.size(0)
            x = x.to(device) # x: input tensor (The previous time step) [b, x1, ..., xd, v]
            y = y.to(device) # y: target tensor (The latter time step) [b, x1, ..., xd, v]
            grid = grid.to(device) # grid: meshgrid [b, x1, ..., xd, dims]
            mask = mask.to(device) # mask [b, x1, ..., xd, 1]
            case_params = case_params.to(device) #parameters [b, x1, ..., xd, p]

            if training_type == 'autoregressive':
                if val_loader.dataset.multi_step_size ==1:
                # Autoregressive loop
                    # Model run
                    if case_params.shape[-1] == 0: #darcy
                        case_params = case_params.reshape(0)
                    pred = model(x, case_params, mask, grid)
                    # Loss calculation
                    _batch = pred.size(0)

                    val_l2 += loss_fn(pred.reshape(_batch, -1), y.reshape(_batch, -1)).item()
                    val_l_inf = max(val_l_inf, torch.max((torch.abs(pred.reshape(_batch, -1) - y.reshape(_batch, -1)))))

                    for name in metric_names:
                        metric_fn = getattr(metrics, name)
                        res_dict[name].append(metric_fn(pred, y))
                    
                    # if step % plot_interval == 0:
                    #     image_dir = Path(ckpt_dir + "/images")
                    #     if not os.path.exists(image_dir):
                    #         os.makedirs(image_dir)
                    #     plot_predictions(inp = x, label = y, pred = pred, out_dir=image_dir, step=step)
                else:
                    # Autoregressive loop
                    preds=[]
                    for i in range(val_loader.dataset.multi_step_size):
                        pred = model(x, case_params, mask[:,i], grid)
                        preds.append(pred)
                        x = pred
                    preds=torch.stack(preds, dim=1)
                    _batch = preds.size(0)
                    val_l2 += loss_fn(preds.reshape(_batch, -1), y.reshape(_batch, -1)).item()
                    val_l_inf = max(val_l_inf, torch.max((torch.abs(preds.reshape(_batch, -1) - y.reshape(_batch, -1)))))
                    for name in metric_names:
                        metric_fn = getattr(metrics, name)
                        res_dict[name].append(metric_fn(preds, y))

    NMSE_List = [i.mean().item() for i in res_dict["NMSE"]]
    plot_loss(NMSE_List, Path(ckpt_dir) / "loss.png")

    #reshape
    for name in metric_names:
        res_list = res_dict[name]
        if name == "MaxError":
            res = torch.stack(res_list, dim=0)
            res, _ = torch.max(res, dim=0)
        else:
            res = torch.cat(res_list, dim=0)
            res = torch.mean(res, dim=0)
        res_dict[name] = res
    metrics.print_res(res_dict)

    val_l2 /= step

    return val_l2, val_l_inf

def test_loop(test_loader, model, device, training_type, output_dir, metric_names=['MSE', 'RMSE', 'L2RE', 'MaxError', 'NMSE', 'MAE'], plot_interval = 10, test_type = 'frames'):
    model.eval()
    step = 0

    if test_type == 'frames':
        print('consider the result between frames')
    elif test_type == 'accumulate':
        print('consider the accumulate result')
    else:
        raise("test type error, plz set it as 'frames' or 'accumulate'")
    
    res_dict = {}
    for name in metric_names:
        res_dict[name] = []
    
    ckpt_dir = "./test/" + test_type + '/' + args["flow_name"] + '_' + args['dataset']['case_name']
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    prev_case_id = -1

    with torch.no_grad():
        for x, y, mask, case_params, grid, case_id in test_loader:
            case_id = case_id.item()
            if prev_case_id != case_id:
                prev_case_id = -1
                step = 0
            
            step += 1
            # batch_size = x.size(0)
            if test_type == 'frames' or prev_case_id == -1:
                x = x.to(device) # x: input tensor (The previous time step grand truth data) [b, x1, ..., xd, v]
            elif test_type == 'accumulate' and prev_case_id != -1:
                x = pred.detach().clone() # x: input tensor (The previous time step prediction) [b, x1, ..., xd, v]
            y = y.to(device) # y: target tensor (The latter time step) [b, x1, ..., xd, v]
            grid = grid.to(device) # grid: meshgrid [b, x1, ..., xd, dims]
            mask = mask.to(device) # mask [b, x1, ..., xd, 1]
            case_params = case_params.to(device) #parameters [b, x1, ..., xd, p]

            if case_params.shape[-1] == 0: #darcy
                case_params = case_params.reshape(0)

            if training_type == 'autoregressive':
                # Autoregressive loop
                # Model run
                pred = model(x, case_params, mask, grid)

                for name in metric_names:
                    metric_fn = getattr(metrics, name)
                    res_dict[name].append(metric_fn(pred, y))
                
                # if step % plot_interval == 0:
                #     image_dir = Path(ckpt_dir + '/case_id' + str(case_id) + "/images")
                #     if not os.path.exists(image_dir):
                #         os.makedirs(image_dir)
                #     if test_type == 'frames':
                #         plot_predictions(inp = x, label = y, pred = pred, out_dir=image_dir, step=step)
                #     elif test_type == 'accumulate':
                #         plot_predictions(label = y, pred = pred, out_dir=image_dir, step=step)

                #     #plot the stream line    
                #     streamline_dir = Path(ckpt_dir + '/case_id' + str(case_id) + "/streamline")
                #     if not os.path.exists(streamline_dir):
                #         os.makedirs(streamline_dir)
                #     plot_stream_line(pred = pred, label = y, grid = grid, out_dir = streamline_dir, step=step)
            prev_case_id = case_id

    NMSE_List = [i.mean().item() for i in res_dict["NMSE"]]
    plot_loss(NMSE_List, Path(ckpt_dir) / "loss.png")

    #reshape
    for name in metric_names:
        res_list = res_dict[name]
        if name == "MaxError":
            res = torch.stack(res_list, dim=0)
            res, _ = torch.max(res, dim=0)
        else:
            res = torch.cat(res_list, dim=0)
            res = torch.mean(res, dim=0)
        res_dict[name] = res
    metrics.print_res(res_dict)

    metrics.write_res(res_dict, test_type + '_results.csv', args["flow_name"] + '_' + args['dataset']['case_name'], append = True)
    return 

def main(args):
    #init
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = None
    saved_dir = os.path.join(args["saved_dir"], args["flow_name"] + '_' + args['dataset']['case_name'])
    output_dir = os.path.join(args["output_dir"], args["flow_name"] + '_' + args['dataset']['case_name'])

    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset_args = args["dataset"]

    saved_model_name = (args["model_name"] + 
                        f"_lr{args['optimizer']['lr']}" +
                        f"_bs{args['dataloader']['train_batch_size']}" +
                        args["flow_name"] + 
                        dataset_args['case_name'] +
                        '_UNO'
                        )
    
    saved_path = os.path.join(saved_dir, saved_model_name)
    
    # data get dataloader
    train_data, val_data, test_data = get_dataset(args)
    train_loader, val_loader, test_loader = get_dataloader(train_data, val_data, test_data, args)

    # set some train args
    sample, _, _, case_params, grid, _, = next(iter(val_loader))
    spatial_dim = grid.shape[-1]
    n_case_params = case_params.shape[-1]

    #model
    model = get_model(spatial_dim, n_case_params, args)
    ##
    if not args["if_training"]:
        print(f"Test mode, load checkpoint from {saved_path} -best.pt")
        checkpoint = torch.load(saved_path + "-best.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        print("start testing...")
        test_loop(test_loader, model, device, args["training_type"], output_dir, test_type='frames')
        test_loop(test_loader, model, device, args["training_type"], output_dir, test_type='accumulate')
        print("Done")
        return
    ## if continue training, resume model from checkpoint
    if args["continue_training"]:
        checkpoint = torch.load(saved_path + "-latest.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device) 
    model.train()

    # optimizer
    optim_args = args["optimizer"]
    optim_name = optim_args.pop("name")
    ## if continue training, resume optimizer and scheduler from checkpoint
    if args["continue_training"]:
        optimizer = getattr(torch.optim, optim_name)(model.parameters())
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        optimizer = getattr(torch.optim, optim_name)(model.parameters(), **optim_args)
    
    # scheduler
    start_epoch = 0
    min_val_loss = torch.inf
    if args["continue_training"]:
        start_epoch = checkpoint['epoch']
        min_val_loss = checkpoint['loss']
    sched_args = args["scheduler"]
    sched_name = sched_args.pop("name")
    scheduler = getattr(torch.optim.lr_scheduler, sched_name)(optimizer, last_epoch=start_epoch-1, **sched_args)

    # loss function
    loss_fn = nn.MSELoss(reduction="mean")

    # save loss history
    loss_history = []
    if args["continue_training"]:
        loss_history = np.load('./log/loss/' + args['flow_name'] + '_' + args['dataset']['case_name'] + '_loss_history.npy')
        loss_history = loss_history.tolist()

     # train loop
    print("start training...")
    total_time = 0
    for epoch in range(start_epoch, args["epochs"]):
        train_l2, train_l_inf, time = train_loop(model,train_loader, optimizer, loss_fn, device, args)
        scheduler.step()
        total_time += time
        loss_history.append(train_l2)
        print(f"[Epoch {epoch}] train_l2: {train_l2}, train_l_inf: {train_l_inf}, time_spend: {time:.3f}")
        ## save latest
        model_state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict()
        torch.save({"epoch": epoch+1, "loss": min_val_loss,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer.state_dict()
            }, saved_path + "-latest.pt")
        if (epoch+1) % args["save_period"] == 0:
            print("====================validate====================")
            val_l2_full, val_l_inf = val_loop(val_loader, model, loss_fn, device, args["training_type"], output_dir, epoch, plot_interval=args['plot_interval'])
            print(f"[Epoch {epoch}] val_l2_full: {val_l2_full} val_l_inf: {val_l_inf}")
            print("================================================")
            if val_l2_full < min_val_loss:
                min_val_loss = val_l2_full
                ## save best
                torch.save({"epoch": epoch + 1, "loss": min_val_loss,
                    "model_state_dict": model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                    }, saved_path + "-best.pt")
    print("Done.")
    loss_history = np.array(loss_history)
    if not os.path.exists('./log/loss/'):
        os.makedirs('./log/loss/')
    np.save('./log/loss/' + args['flow_name'] + '_' + args['dataset']['case_name'] + '_loss_history.npy', loss_history)
    print("avg_time : {0:.5f}".format(total_time / (args["epochs"] - start_epoch)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to config file")
    cmd_args = parser.parse_args()
    with open(cmd_args.config_file, 'r') as f:
        args = yaml.safe_load(f)
    setup_seed(args["seed"])
    print(args)
    main(args)
# print(torch.cuda.device_count())