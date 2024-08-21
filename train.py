import os
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import yaml
import argparse
import metrics
from timeit import default_timer
from functools import reduce
from utils import setup_seed, get_model, get_dataset, get_dataloader, get_min_max


def train_loop(model, train_loader, optimizer, batch_scheduler, loss_fn, device, args):
    model.train()
    t1 = default_timer()
    train_loss = 0
    train_l_inf = 0
    step = 0
    # (channel_min, channel_max) = args["channel_min_max"] 
    # channel_min, channel_max = channel_min.to(device), channel_max.to(device)
    for x, y, mask, case_params, grid, _ in train_loader:
        step += 1
        # batch_size = x.size(0)
        loss = 0
        x = x.to(device) # x: input tensor (The previous time step) [b, x1, ..., xd, v] (regular grid) or [b, Nx, v] (point clouds)
        y = y.to(device) # y: target tensor (The latter time step) [b, x1, ..., xd, v], [b, ms, x1, ..., xd,v] (multi-step), or [b, ms, Nx, v] (point clouds)
        grid = grid.to(device) # grid: meshgrid [b, x1, ..., xd, dims] or [b, Nx, dims] 
        mask = mask.to(device) # mask [b, x1, ..., xd, 1] or (b, Nx, 1)
        case_params = case_params.to(device) #parameters [b, x1, ..., xd, p] or [b, Nx, p]
        y = y * mask

        if args["model_name"] in ["OFormer"]:
            #Model run one_step or multi_step in the latent space
            if case_params.shape[-1] == 0: #darcy
                case_params = case_params.reshape(0)
            loss, pred, info = model.one_forward_step(x, case_params, mask,  grid, y, loss_fn=loss_fn) 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _batch = pred.size(0)
            train_loss += loss.item()
            train_l_inf = max(train_l_inf, torch.max((torch.abs(pred.reshape(_batch, -1) - y.reshape(_batch, -1)))))
            
        else:
            if getattr(train_loader.dataset,"multi_step_size", 1) ==1:
                #Model run one_step
                if case_params.shape[-1] == 0: #darcy
                    case_params = case_params.reshape(0)

                if args["model_name"] in ["NUFNO"]:  # have auxliary output
                    loss, pred , aux_out, info = model.one_forward_step(x, case_params, mask,  grid, y, loss_fn=loss_fn) 
                else:
                    loss, pred , info = model.one_forward_step(x, case_params, mask,  grid, y, loss_fn=loss_fn) 
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                _batch = pred.size(0)
                train_loss += loss.item()
                train_l_inf = max(train_l_inf, torch.max((torch.abs(pred.reshape(_batch, -1) - y.reshape(_batch, -1)))))
            else:
                # Autoregressive loop
                preds=[]
                total_loss = 0
                for i in range(train_loader.dataset.multi_step_size):
                    if args["model_name"] in ["NUFNO"]:  # have auxliary output as next input, since its input and output are incompatible.
                        loss, pred , aux_out, info = model.one_forward_step(x, case_params, mask[:,i],  grid, y[:,i], loss_fn=loss_fn)
                        x = aux_out
                    else:
                        loss, pred , info = model.one_forward_step(x, case_params, mask[:,i],  grid, y[:,i], loss_fn=loss_fn)
                        x = pred
                    
                    preds.append(pred)
                    total_loss = total_loss + loss
                    
                preds=torch.stack(preds, dim=1)
                _batch = preds.size(0)
                # loss = loss_fn(preds.reshape(_batch, -1), y.reshape(_batch, -1))
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                if batch_scheduler:
                    batch_scheduler.step()  # step per batch

                train_loss += total_loss.item()
                train_l_inf = max(train_l_inf, torch.max((torch.abs(preds.reshape(_batch, -1) - y.reshape(_batch, -1)))))
    train_loss /= step      
    t2 = default_timer()
    return train_loss, train_l_inf, t2 - t1

def val_loop(val_loader, model, loss_fn, device, output_dir, epoch, args, metric_names=['MSE', 'RMSE', 'L2RE', 'MaxError', 'NMSE', 'MAE'], plot_interval = 1):
    model.eval()
    val_l2 = 0
    val_l_inf = 0
    step = 0
    
    res_dict = {"cw_res":{},  # channel-wise
                "sw_res":{},}  # sample-wise
    for name in metric_names:
        res_dict["cw_res"][name] = []
        res_dict["sw_res"][name] = []
    

    with torch.no_grad():
        for x, y, mask, case_params, grid, case_id in val_loader:
            step += 1
            # batch_size = x.size(0)
            x = x.to(device) # x: input tensor (The previous time step) [b, x1, ..., xd, v]
            y = y.to(device) # y: target tensor (The latter time step) [b, x1, ..., xd, v]
            grid = grid.to(device) # grid: meshgrid [b, x1, ..., xd, dims]
            mask = mask.to(device) # mask [b, x1, ..., xd, 1]
            case_params = case_params.to(device) #parameters [b, x1, ..., xd, p]
            y = y * mask

            if args["model_name"] in ["OFormer"]:
                # Model run in the latent space
                if case_params.shape[-1] == 0: #darcy
                    case_params = case_params.reshape(0)
                pred = model(x, case_params, mask, grid)
                y = y.reshape(-1, val_loader.dataset.multi_step_size, args["model"]["num_points"], y.shape[-1])
                
                
                # Loss calculation
                _batch = pred.size(0)

                val_l2 += loss_fn(pred.reshape(_batch, -1), y.reshape(_batch, -1)).item()
                val_l_inf = max(val_l_inf, torch.max((torch.abs(pred.reshape(_batch, -1) - y.reshape(_batch, -1)))))
                

                for name in metric_names:
                    metric_fn = getattr(metrics, name)
                    cw, sw=metric_fn(pred, y)
                    res_dict["cw_res"][name].append(cw)
                    res_dict["sw_res"][name].append(sw)

            else:
                if getattr(val_loader.dataset,"multi_step_size", 1)==1:
                    # Model run
                    if case_params.shape[-1] == 0: #darcy
                        case_params = case_params.reshape(0)

                    if args["model_name"] in ["NUFNO"]:
                        pred, aux_out = model(x, case_params, mask, grid)
                    else:
                        pred = model(x, case_params, mask, grid)

                    # Loss calculation
                    _batch = pred.size(0)
                    val_l2 += loss_fn(pred.reshape(_batch, -1), y.reshape(_batch, -1)).item()
                    val_l_inf = max(val_l_inf, torch.max((torch.abs(pred.reshape(_batch, -1) - y.reshape(_batch, -1)))))
                    
                    for name in metric_names:
                        metric_fn = getattr(metrics, name)
                        cw, sw=metric_fn(pred, y)
                        res_dict["cw_res"][name].append(cw)
                        res_dict["sw_res"][name].append(sw)
                    
                else:
                    # Autoregressive loop
                    preds=[]
                    for i in range(val_loader.dataset.multi_step_size):
                        if args["model_name"] in ["NUFNO"]:
                            pred, aux_out = model(x, case_params, mask[:,i], grid)
                            x = aux_out
                        else:
                            pred = model(x, case_params, mask[:,i], grid)
                            x = pred

                        preds.append(pred)
                        
                    preds=torch.stack(preds, dim=1)
                    _batch = preds.size(0)
                    val_l2 += loss_fn(preds.reshape(_batch, -1), y.reshape(_batch, -1)).item()
                    val_l_inf = max(val_l_inf, torch.max((torch.abs(preds.reshape(_batch, -1) - y.reshape(_batch, -1)))))
                    for name in metric_names:
                        metric_fn = getattr(metrics, name)
                        cw, sw=metric_fn(preds, y)
                        res_dict["cw_res"][name].append(cw)
                        res_dict["sw_res"][name].append(sw)

    #aggregation
    for name in metric_names:
        cw_res_list = res_dict["cw_res"][name]
        sw_res_list = res_dict["sw_res"][name]
        if name == "MaxError":
            cw_res = torch.stack(cw_res_list, dim=0)
            cw_res, _ = torch.max(cw_res, dim=0)
            sw_res = torch.stack(sw_res_list)
            sw_res = torch.max(sw_res)
        else:
            cw_res = torch.cat(cw_res_list, dim=0)
            cw_res = torch.mean(cw_res, dim=0)
            sw_res = torch.cat(sw_res_list, dim=0)
            sw_res = torch.mean(sw_res, dim=0)
        res_dict["cw_res"][name] = cw_res
        res_dict["sw_res"][name] = sw_res
    metrics.print_res(res_dict)

    val_l2 /= step

    return val_l2, val_l_inf


def test_loop(test_loader, model, device, output_dir, args, metric_names=['MSE', 'RMSE', 'L2RE', 'MaxError', 'NMSE', 'MAE'], plot_interval = 10, test_type = 'frames'):
    model.eval()
    step = 0

    if test_type == 'frames':
        print('consider the result between frames')
    elif test_type == 'accumulate':
        print('consider the accumulate result')
    elif test_type == "multi_step":
        print("consider the accumulate result for multi_step")
    else:
        raise("test type error, plz set it as 'frames', 'accumulate' or 'multi_step'")

    
    res_dict = {"cw_res":{},  # channel-wise
                "sw_res":{},}  # sample-wise
    for name in metric_names:
        res_dict["cw_res"][name] = []
        res_dict["sw_res"][name] = []

    if args["use_norm"]:
        (channel_min, channel_max) = args["channel_min_max"] 
        channel_min, channel_max = channel_min.to(device), channel_max.to(device)

    prev_case_id = -1
    preds = []
    gts = []
    aux_data = torch.tensor(())
    t1 = default_timer()
    with torch.no_grad():
        for x, y, mask, case_params, grid, case_id in test_loader:
            case_id = case_id.item()
            if prev_case_id != case_id:
                prev_case_id = -1
                step = 0
            
            step += 1
            x = x.to(device)
            y = y.to(device) # y: target tensor  [b, x1, ..., xd, v] if mutli_step_size ==1 else [b, multi_step_size, x1, ..., xd, v]
            grid = grid.to(device) # grid: meshgrid [b, x1, ..., xd, dims] 
            mask = mask.to(device) # mask [b, x1, ..., xd, 1] if mutli_step_size ==1 else [b, multi_step_size, x1, ..., xd, 1]
            case_params = case_params.to(device) #parameters [b, x1, ..., xd, p]
            y = y * mask
            
            if getattr(test_loader.dataset,"multi_step_size", 1) ==1:
                # batch_size = x.size(0)
                if test_type == 'frames':
                    pass
                elif test_type == 'accumulate': 
                    if prev_case_id == -1:
                        # new case start
                        if len(preds)> 0: 
                            preds=torch.stack(preds, dim=0).unsqueeze(0)   # [1, t, x1, ...,xd, v]
                            # print(preds.shape)
                            gts = torch.stack(gts, dim=0).unsqueeze(0) # [1, t, x1, ...,xd, v]
                            for name in metric_names:
                                metric_fn = getattr(metrics, name)
                                cw, sw=metric_fn(preds, gts)
                                res_dict["cw_res"][name].append(cw)
                                res_dict["sw_res"][name].append(sw)

                        preds = []
                        gts = []
                    else:
                        if args["model_name"] in ["NUFNO"]:
                            x = aux_out.detach()  # aux_out as next input, since the preds is incompatible with inputs 
                        else:
                            x = pred.detach() # x: input tensor (The previous time step prediction) [b, x1, ..., xd, v]
                        
                else:
                    raise Exception(f"test_type {test_type} is not support for a single_step test_loader ")


                if args["model_name"] in ["NUFNO"]:
                    pred, aux_out = model(x, case_params, mask, grid)
                else: 
                    pred = model(x, case_params, mask, grid)

                #collect data, support reverse normalization
                if test_type == 'frames':
                    for name in metric_names:
                        metric_fn = getattr(metrics, name)
                        if args["use_norm"] and args["if_denorm"]:
                            cw, sw=metric_fn(pred * (channel_max - channel_min) + channel_min, y * (channel_max - channel_min) + channel_min)
                        else:
                            cw, sw=metric_fn(pred, y)
                        res_dict["cw_res"][name].append(cw)
                        res_dict["sw_res"][name].append(sw)
                else: # accumulate
                    if args["use_norm"] and args["if_denorm"]:
                        preds.append(pred * (channel_max - channel_min) + channel_min)
                        gts.append(y * (channel_max - channel_min) + channel_min) 
                    else:
                        preds.append(pred)
                        gts.append(y) 
                    
                prev_case_id = case_id
            else:
                # autoregressive loop for multi_step
                if args["model_name"] in ["OFormer"]:
                    preds=model(x, case_params, mask, grid)
                else:
                    preds=[]
                    for i in range(test_loader.dataset.multi_step_size):
                        if args["model_name"] in ["NUFNO"]:
                            pred, aux_out = model(x, case_params, mask[:,i], grid)
                            x = aux_out
                        else:
                            pred = model(x, case_params, mask[:,i], grid)
                            x = pred
                        preds.append(pred)
                        
                    preds=torch.stack(preds, dim=1)
                if args["use_norm"] and args["if_denorm"]:
                    preds = preds * (channel_max - channel_min) + channel_min
                    y = y * (channel_max - channel_min) + channel_min
                for name in metric_names:
                    metric_fn = getattr(metrics, name)
                    cw, sw=metric_fn(preds, y)
                    res_dict["cw_res"][name].append(cw)
                    res_dict["sw_res"][name].append(sw)
    if test_type == "accumulate" and len(preds)> 0:  # the last case
        preds=torch.stack(preds, dim=1)   # [1, t, x1, ...,xd, v]
        gts = torch.stack(gts, dim=1) # [1, t, x1, ...,xd, v]
        for name in metric_names:
            metric_fn = getattr(metrics, name)
            cw, sw=metric_fn(preds, gts)
            res_dict["cw_res"][name].append(cw)
            res_dict["sw_res"][name].append(sw) 


            
    t2 = default_timer()
    Mean_inference_time = (t2-t1)/len(test_loader.dataset)
    
    if test_type == 'frames':
        res_dict['Mean inference time'] = Mean_inference_time
        print("averge time: {0:.4f} s".format(Mean_inference_time))

    #reshape
    for name in metric_names:
        cw_res_list = res_dict["cw_res"][name]
        sw_res_list = res_dict["sw_res"][name]
        if name == "MaxError":
            cw_res = torch.stack(cw_res_list, dim=0)
            cw_res, _ = torch.max(cw_res, dim=0)
            sw_res = torch.stack(sw_res_list)
            sw_res = torch.max(sw_res)
        else:
            cw_res = torch.cat(cw_res_list, dim=0)
            cw_res = torch.mean(cw_res, dim=0)
            sw_res = torch.cat(sw_res_list, dim=0)
            sw_res = torch.mean(sw_res, dim=0)
        res_dict["cw_res"][name] = cw_res
        res_dict["sw_res"][name] = sw_res

    metrics.print_res(res_dict)
    denorm_str = '(denormed)' if args["if_denorm"] else ""
    metrics.write_res(res_dict, 
                      os.path.join(args["output_dir"],args["model_name"]+test_type + '_results.csv'),
                       args["flow_name"] + '_' + args['dataset']['case_name']+ denorm_str, 
                       append = True)
    return 



def main(args):
    #init
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = None
    saved_dir = os.path.join(args["saved_dir"], os.path.join(args["model_name"], args["flow_name"] + '_' + args['dataset']['case_name']))
    output_dir = os.path.join(args["output_dir"], os.path.join(args["model_name"],args["flow_name"] + '_' + args['dataset']['case_name']))

    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset_args = args["dataset"]

    saved_model_name = (args["model_name"] + 
                        f"_lr{args['optimizer']['lr']}" +
                        f"_bs{args['dataloader']['train_batch_size']}" +
                        args["flow_name"] + 
                        dataset_args['case_name']
                        )
    
    saved_path = os.path.join(saved_dir, saved_model_name)
    
    # data get dataloader
    train_data, val_data, test_data, test_ms_data = get_dataset(args)
    train_loader, val_loader, test_loader, test_ms_loader = get_dataloader(train_data, val_data, test_data, test_ms_data, args)

    # set some train args
    input, output, _, case_params, grid, _, _ = next(iter(val_loader))
    print("input tensor shape: ", input.shape[1:])
    print("output tensor shape: ", output.shape[1:] if val_loader.dataset.multi_step_size==1 else output.shape[2:])
    spatial_dim = grid.shape[-1]
    n_case_params = case_params.shape[-1]
    args["model"]["num_points"] = reduce(lambda x,y: x*y, grid.shape[1:-1])  # get num_points, especially of irregular geometry(point clouds)

    # get min_max per channel of train-set on the fly for normalization.
    
    if args["use_norm"]:
        channel_min, channel_max = get_min_max(train_loader, args)   
        args["channel_min_max"] = (channel_min, channel_max)
        print("use min_max normalization with min=", channel_min.tolist(), ", max=", channel_max.tolist())
        train_loader.dataset.apply_norm(channel_min, channel_max)
        val_loader.dataset.apply_norm(channel_min, channel_max)
        test_loader.dataset.apply_norm(channel_min, channel_max)
        if test_ms_data is not None:
            test_ms_loader.dataset.apply_norm(channel_min, channel_max)
        print("min-max normalization finished")

    #model
    model = get_model(spatial_dim, n_case_params, args)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model {args['model_name']} has {num_params} parameters")

    ##
    if not args["if_training"]:
        print(f"Test mode, load checkpoint from {saved_path}-best.pt")
        checkpoint = torch.load(saved_path + "-best.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        print("start testing...")
        print(f"Reverse normalization = {args['if_denorm']}")
        test_loop(test_loader, model, device, output_dir, args, test_type='frames')
        if test_ms_data is not None:  # not darcy
            test_loop(test_ms_loader, model, device, output_dir, args, test_type='multi_step')
        if args["flow_name"] not in ["Darcy"]:
            test_loop(test_loader, model, device, output_dir, args, test_type='accumulate')
        print("Done") 
        return
    ## if continue training, resume model from checkpoint
    if args["continue_training"]:
        checkpoint = torch.load(saved_path + "-latest.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
        print("loading latest checkpoint from ", saved_path + "-latest.pt")
        

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
    if sched_args.get("per_batch", False):   #  batch scheduler.  # in [GFormer, ]
        sched_name = sched_args.pop("name")
        sched_args.pop("per_batch")
        batch_scheduler = getattr(torch.optim.lr_scheduler, sched_name)(optimizer, total_steps=len(train_loader)*args['epochs'] ,
                                                                        last_epoch=start_epoch*len(train_loader)-1,
                                                                          **sched_args)
        scheduler=None
    else:   # epoch scheduler
        sched_name = sched_args.pop("name")
        scheduler = getattr(torch.optim.lr_scheduler, sched_name)(optimizer, last_epoch=start_epoch-1, **sched_args)
        batch_scheduler = None
        
    # loss function
    loss_fn = nn.MSELoss(reduction="mean")

    # loss history
    loss_history = []
    if args["continue_training"]:
        loss_history = checkpoint["history"]

    # train loop
    print("start training...")
    total_time = 0
    for epoch in range(start_epoch, args["epochs"]):
        train_loss, train_l_inf, time = train_loop(model,train_loader, optimizer, batch_scheduler, loss_fn, device, args)
        if scheduler:  # epoch scheduler
            scheduler.step()
        total_time += time
        loss_history.append(train_loss)
        print(f"[Epoch {epoch}] train_loss: {train_loss}, train_l_inf: {train_l_inf}, time_spend: {time:.3f}")
        ## save latest
        model_state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict()
        
        torch.save({"epoch": epoch+1, "loss": min_val_loss,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "history": loss_history
            }, saved_path + "-latest.pt")
        if (epoch+1) % args["save_period"] == 0:
            print("====================validate====================")
            val_l2_full, val_l_inf = val_loop(val_loader, model, loss_fn, device, output_dir, epoch, args, plot_interval=args['plot_interval'])
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
    print("avg_time : {0:.5f}".format(total_time / (args["epochs"] - start_epoch)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Path to config file")
    parser.add_argument("--test", action='store_true', help='test mode otherwise train mode')
    parser.add_argument("--no_denorm", action='store_true', help='no denorm in test mode')
    parser.add_argument("--continue_training", action='store_true', help='continue training')
    parser.add_argument("-c", "--case_name", type=str, default="", help="For the case, if no value is entered, the yaml file shall prevail")

    cmd_args = parser.parse_args()
    with open(cmd_args.config_file, 'r') as f:
        args = yaml.safe_load(f)
    args['if_training'] = not cmd_args.test
    args['continue_training'] = cmd_args.continue_training
    if len(cmd_args.case_name) > 0:
        args['dataset']['case_name'] = cmd_args.case_name

    setup_seed(args["seed"])

    # set use_norm
    use_norm_default=True
    if args["flow_name"] in ["TGV","Darcy"]:
        use_norm_default = False
    args["use_norm"] = args.get("use_norm", use_norm_default)
    args["if_denorm"] = not cmd_args.no_denorm
    
    print(args, flush=True)
    main(args)