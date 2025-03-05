from argparse import ArgumentParser
import os
import sys
import torch
from torch import optim
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
from statistics import mean
import torch.distributed as dist
import numpy as np
from torch.optim.lr_scheduler import OneCycleLR

from lavis.models import load_model_and_preprocess
from dataset import (
    targetpad_transform,
    ComposeDataset,
    SingleDataset,
    FashionIQDataset,
    STDataset,
)
from utils import (
    setup_seed,
    AverageMeter,
    select_caps_compose,
    save_model,
    add_prompt_cap,
    generate_randomized_fiq_caption,
    extract_index_fiq,
)
from validate import compute_blip_fiq


def train_compose(
    num_epochs,
    blip_model_name,
    learning_rate,
    batch_size,
    validation_frequency: int,
    **kwargs,
):
    global best_r_average
    device = kwargs["device"]
    local_rank = kwargs["local_rank"]
    time_str = datetime.now().strftime("%m-%d-%H")

    if local_rank != 0:
        sys.stdout = open(os.devnull, "w")

    blip_model, _, txt_processors = load_model_and_preprocess(
        name=blip_model_name, model_type="pretrain", is_eval=False, device=device
    )
    update_method = getattr(blip_model, "_update_f_former", None)
    if callable(update_method):
        blip_model._update_f_former()
    blip_model = torch.nn.parallel.DistributedDataParallel(
        blip_model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    input_dim = 224
    target_ratio = kwargs["target_ratio"]
    preprocess = targetpad_transform(target_ratio, input_dim)

    dataset_name = kwargs["dataset"]
    relative_train_dataset = ComposeDataset("train", "relative", preprocess, dataset_name)
    train_sampler = torch.utils.data.distributed.DistributedSampler(relative_train_dataset, shuffle=True)
    relative_train_loader = DataLoader(
        dataset=relative_train_dataset,
        batch_size=batch_size,
        num_workers=kwargs["num_workers"],
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
    )

    if local_rank != 0:
        sys.stdout = sys.__stdout__  # 上述代码print不会输出

    optimizer = optim.AdamW(
        [
            {
                "params": filter(lambda p: p.requires_grad, blip_model.parameters()),
                "lr": learning_rate,
                "betas": (0.9, 0.98),
                "eps": 1e-7,
                "weight_decay": 0.05,
            }
        ]
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        pct_start=1.5 / num_epochs,
        div_factor=100.0,
        steps_per_epoch=len(relative_train_loader),
        epochs=num_epochs,
    )
    scaler = torch.cuda.amp.GradScaler()
    best_r_average = 0
    for epoch in range(num_epochs):
        losses = AverageMeter()
        train_sampler.set_epoch(epoch)

        for idx, (
            ref_img,
            tar1_img,
            tar2_img,
            tar3_img,
            mod1,
            mod2,
            mod3,
            cap0,
            cap1,
            cap2,
            cap3,
        ) in enumerate(
            tqdm(
                relative_train_loader,
                desc=f"Train [{epoch}]",
                disable=(local_rank != 0),
            )
        ):
            optimizer.zero_grad()
            blip_model.train()

            ref_img = ref_img.to(device, non_blocking=True)
            tar1_img = tar1_img.to(device, non_blocking=True)
            tar2_img = tar2_img.to(device, non_blocking=True)
            tar3_img = tar3_img.to(device, non_blocking=True)
            # 注意文本处理形式
            mod1, mod2, mod3 = (
                select_caps_compose(mod1),
                select_caps_compose(mod2),
                select_caps_compose(mod3),
            )
            mod1_inputs = [txt_processors["eval"](caption) for caption in mod1]
            mod2_inputs = [txt_processors["eval"](caption) for caption in mod2]
            mod3_inputs = [txt_processors["eval"](caption) for caption in mod3]

            if dataset_name == "fc":
                cap0, cap1, cap_concat = (
                    add_prompt_cap(cap0, prompt="FIRST"),
                    add_prompt_cap(cap1, prompt="SECOND"),
                    add_prompt_cap(cap0, cap1),
                )
                cap0_inputs = [txt_processors["eval"](caption) for caption in cap0]
                cap1_inputs = [txt_processors["eval"](caption) for caption in cap1]
                cap2_inputs = [txt_processors["eval"](caption) for caption in cap2]
                cap3_inputs = [txt_processors["eval"](caption) for caption in cap3]
                cap_concat_inputs = [txt_processors["eval"](caption) for caption in cap_concat]
                with torch.cuda.amp.autocast():
                    loss_dict1, fus_token1 = blip_model(
                        {
                            "image": ref_img,
                            "target": tar1_img,
                            "text_input": mod1_inputs,
                            "tar_cap": cap1_inputs,
                            "ref_cap": cap0_inputs,
                        },
                        mode="single",
                    )
                    loss_dict2, fus_token2 = blip_model(
                        {
                            "image": tar1_img,
                            "target": tar2_img,
                            "text_input": mod2_inputs,
                            "tar_cap": cap2_inputs,
                            "ref_cap": cap1_inputs,
                            "fusion": fus_token1,
                        },
                        mode="single",
                    )
                    loss_dict3 = blip_model(
                        {
                            "image": tar2_img,
                            "target": tar3_img,
                            "text_input": mod3_inputs,
                            "tar_cap": cap3_inputs,
                            "ref_cap": cap_concat_inputs,
                            "fusion": fus_token2,
                        },
                        mode="multi",
                    )
                    loss = sum(loss_dict1.values()) + sum(loss_dict2.values()) + sum(loss_dict3.values())

            elif dataset_name == "fiq":
                cap0 = add_prompt_cap(cap0, prompt="FIRST")
                cap1 = add_prompt_cap(cap1, prompt="SECOND")
                cap2 = add_prompt_cap(cap2, prompt="THIRD")
                cap_concat = add_prompt_cap(cap0, cap1, cap2)
                cap0_inputs = [txt_processors["eval"](caption) for caption in cap0]
                cap1_inputs = [txt_processors["eval"](caption) for caption in cap1]
                cap2_inputs = [txt_processors["eval"](caption) for caption in cap2]
                cap3_inputs = [txt_processors["eval"](caption) for caption in cap3]
                cap_concat_inputs = [txt_processors["eval"](caption) for caption in cap_concat]
                with torch.cuda.amp.autocast():
                    loss_dict1, fus_token1 = blip_model(
                        {
                            "image": ref_img,
                            "target": tar1_img,
                            "text_input": mod1_inputs,
                            "tar_cap": cap1_inputs,
                            "ref_cap": cap0_inputs,
                        },
                        mode="single",
                    )
                    loss_dict2, fus_token2 = blip_model(
                        {
                            "image": tar1_img,
                            "target": tar2_img,
                            "text_input": mod2_inputs,
                            "tar_cap": cap2_inputs,
                            "ref_cap": cap1_inputs,
                            "fusion": fus_token1,
                        },
                        mode="single",
                    )
                    fus_token3 = blip_model(
                        {
                            "image": tar1_img,
                            "ref_cap": cap1_inputs,
                            "fusion": fus_token1,
                        },
                        mode="single_no_target",
                    )
                    loss_dict3 = blip_model(
                        {
                            "image": tar2_img,
                            "target": tar3_img,
                            "text_input": mod3_inputs,
                            "tar_cap": cap3_inputs,
                            "ref_cap": cap_concat_inputs,
                            "fusion": fus_token3,
                        },
                        mode="multi",
                    )
                    loss = sum(loss_dict1.values()) + sum(loss_dict2.values()) + sum(loss_dict3.values())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.update(loss.detach().cpu().item())

        if epoch % validation_frequency == 0 and local_rank == 0:
            print(
                "Train Epoch: [{0}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                    epoch,
                    loss=losses,
                )
            )
            save_model(
                f"/mnt/longvideo/chenyanzhe/Multiturn/ckpt/blip_{dataset_name}/{time_str}/epoch{epoch}.pth",
                epoch,
                blip_model.module,
            )

        torch.cuda.empty_cache()


def train_single(
    num_epochs,
    blip_model_name,
    learning_rate,
    batch_size,
    validation_frequency: int,
    **kwargs,
):
    global best_r_average
    device = kwargs["device"]
    local_rank = kwargs["local_rank"]
    time_str = datetime.now().strftime("%m-%d-%H")

    if local_rank != 0:
        sys.stdout = open(os.devnull, "w")

    blip_model, _, txt_processors = load_model_and_preprocess(
        name=blip_model_name, model_type="pretrain", is_eval=False, device=device
    )
    update_method = getattr(blip_model, "_update_f_former", None)
    if callable(update_method):
        blip_model._update_f_former()
    blip_model = torch.nn.parallel.DistributedDataParallel(
        blip_model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    input_dim = 224
    target_ratio = kwargs["target_ratio"]
    preprocess = targetpad_transform(target_ratio, input_dim)

    dataset_name = kwargs["dataset"]
    relative_train_dataset = SingleDataset("train", "relative", preprocess, dataset_name)
    train_sampler = torch.utils.data.distributed.DistributedSampler(relative_train_dataset, shuffle=True)
    relative_train_loader = DataLoader(
        dataset=relative_train_dataset,
        batch_size=batch_size,
        num_workers=kwargs["num_workers"],
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
    )

    if local_rank != 0:
        sys.stdout = sys.__stdout__  # 上述代码print不会输出

    optimizer = optim.AdamW(
        [
            {
                "params": filter(lambda p: p.requires_grad, blip_model.parameters()),
                "lr": learning_rate,
                "betas": (0.9, 0.98),
                "eps": 1e-7,
                "weight_decay": 0.05,
            }
        ]
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        pct_start=1.5 / num_epochs,
        div_factor=100.0,
        steps_per_epoch=len(relative_train_loader),
        epochs=num_epochs,
    )
    scaler = torch.cuda.amp.GradScaler()
    best_r_average = 0
    for epoch in range(num_epochs):
        losses = AverageMeter()
        train_sampler.set_epoch(epoch)

        for idx, (
            ref_img,
            tar1_img,
            tar2_img,
            tar3_img,
            mod1,
            mod2,
            mod3,
            cap0,
            cap1,
            cap2,
            cap3,
            last_ref_img,
            last_cap,
        ) in enumerate(
            tqdm(
                relative_train_loader,
                desc=f"Train [{epoch}]",
                disable=(local_rank != 0),
            )
        ):
            optimizer.zero_grad()
            blip_model.train()

            ref_img = ref_img.to(device, non_blocking=True)
            tar1_img = tar1_img.to(device, non_blocking=True)
            tar2_img = tar2_img.to(device, non_blocking=True)
            tar3_img = tar3_img.to(device, non_blocking=True)
            last_ref_img = last_ref_img.to(device, non_blocking=True)
            # 注意文本处理形式
            mod1, mod2, mod3 = (
                select_caps_compose(mod1),
                select_caps_compose(mod2),
                select_caps_compose(mod3),
            )
            mod1_inputs = [txt_processors["eval"](caption) for caption in mod1]
            mod2_inputs = [txt_processors["eval"](caption) for caption in mod2]
            mod3_inputs = [txt_processors["eval"](caption) for caption in mod3]

            cap0_inputs = [txt_processors["eval"](caption) for caption in cap0]
            cap1_inputs = [txt_processors["eval"](caption) for caption in cap1]
            cap2_inputs = [txt_processors["eval"](caption) for caption in cap2]
            last_cap_inputs = [txt_processors["eval"](caption) for caption in last_cap]
            with torch.cuda.amp.autocast():
                loss_dict1, fus_token1 = blip_model(
                    {
                        "image": ref_img,
                        "target": tar1_img,
                        "text_input": mod1_inputs,
                        "tar_cap": cap1_inputs,
                        "ref_cap": cap0_inputs,
                    },
                    mode="single",
                )
                loss_dict2, fus_token2 = blip_model(
                    {
                        "image": tar1_img,
                        "target": tar2_img,
                        "text_input": mod2_inputs,
                        "tar_cap": cap2_inputs,
                        "ref_cap": cap1_inputs,
                        "fusion": fus_token1,
                    },
                    mode="single",
                )
                loss_dict3, fus_token3 = blip_model(
                    {
                        "image": last_ref_img,
                        "target": tar3_img,
                        "text_input": mod3_inputs,
                        "tar_cap": cap2_inputs,
                        "ref_cap": last_cap_inputs,
                        "fusion": fus_token2,
                    },
                    mode="single",
                )
                loss = sum(loss_dict1.values()) + sum(loss_dict2.values()) + sum(loss_dict3.values())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.update(loss.detach().cpu().item())

        if epoch % validation_frequency == 0 and local_rank == 0:
            print(
                "Train Epoch: [{0}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                    epoch,
                    loss=losses,
                )
            )
            save_model(
                f"/mnt/longvideo/chenyanzhe/Multiturn/ckpt/blip_{dataset_name}/{time_str}/epoch{epoch}.pth",
                epoch,
                blip_model.module,
            )

        torch.cuda.empty_cache()


def train_fiq(
    num_epochs,
    blip_model_name,
    learning_rate,
    batch_size,
    validation_frequency: int,
    **kwargs,
):
    global best_r_average
    device = kwargs["device"]
    local_rank = kwargs["local_rank"]

    if local_rank != 0:
        sys.stdout = open(os.devnull, "w")

    blip_model, _, txt_processors = load_model_and_preprocess(
        name=blip_model_name, model_type="pretrain", is_eval=False, device=device
    )
    update_method = getattr(blip_model, "_update_f_former", None)
    if callable(update_method):
        blip_model._update_f_former()

    checkpoint_path = "/mnt/longvideo/chenyanzhe/Multiturn/ckpt/blip_fiq/05-07-14/epoch25.pth"  # TODO
    checkpoint = torch.load(checkpoint_path, map_location=device)
    blip_model.load_state_dict(checkpoint[blip_model.__class__.__name__], strict=False)

    blip_model = torch.nn.parallel.DistributedDataParallel(
        blip_model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    input_dim = 224
    target_ratio = kwargs["target_ratio"]
    preprocess = targetpad_transform(target_ratio, input_dim)

    idx_to_dress_mapping = {}
    relative_val_datasets = []
    classic_val_datasets = []
    types = ["dress", "toptee", "shirt"]

    # Define the validation datasets
    for idx, dress_type in enumerate(types):
        idx_to_dress_mapping[idx] = dress_type
        relative_val_dataset = FashionIQDataset(
            "val",
            [dress_type],
            "relative",
            preprocess,
        )
        relative_val_datasets.append(relative_val_dataset)
        classic_val_dataset = FashionIQDataset(
            "val",
            [dress_type],
            "classic",
            preprocess,
        )
        classic_val_datasets.append(classic_val_dataset)

    relative_train_dataset = FashionIQDataset("train", types, "relative", preprocess)
    train_sampler = torch.utils.data.distributed.DistributedSampler(relative_train_dataset, shuffle=True)
    relative_train_loader = DataLoader(
        dataset=relative_train_dataset,
        batch_size=batch_size,
        num_workers=kwargs["num_workers"],
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
    )

    if local_rank != 0:
        sys.stdout = sys.__stdout__  # 上述代码print不会输出

    optimizer = optim.AdamW(
        [
            {
                "params": filter(lambda p: p.requires_grad, blip_model.parameters()),
                "lr": learning_rate,
                "betas": (0.9, 0.98),
                "eps": 1e-7,
                "weight_decay": 0.05,
            }
        ]
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        pct_start=1.5 / num_epochs,
        div_factor=100.0,
        steps_per_epoch=len(relative_train_loader),
        epochs=num_epochs,
    )
    scaler = torch.cuda.amp.GradScaler()
    best_r_average = 0
    for epoch in range(num_epochs):
        losses = AverageMeter()
        train_sampler.set_epoch(epoch)

        for idx, (ref_img, tar_img, mod, ref_cap, tar_cap) in enumerate(
            tqdm(
                relative_train_loader,
                desc=f"Train [{epoch}]",
                disable=(local_rank != 0),
            )
        ):
            optimizer.zero_grad()
            blip_model.train()
            ref_img = ref_img.to(device, non_blocking=True)
            tar_img = tar_img.to(device, non_blocking=True)
            # ?注意文本处理形式
            flattened_captions: list = np.array(mod).T.flatten().tolist()
            modifiers = generate_randomized_fiq_caption(flattened_captions)
            mod_input = [txt_processors["eval"](caption) for caption in modifiers]
            cap_ref_inputs = [txt_processors["eval"](caption) for caption in ref_cap]
            cap_tar_inputs = [txt_processors["eval"](caption) for caption in tar_cap]

            with torch.cuda.amp.autocast():
                loss_dict, _ = blip_model(
                    {
                        "image": ref_img,
                        "target": tar_img,
                        "text_input": mod_input,
                        "tar_cap": cap_tar_inputs,
                        "ref_cap": cap_ref_inputs,
                    },
                    mode="fiq",  # TODO
                )
                loss = sum(loss_dict.values())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.update(loss.detach().cpu().item())

        if epoch % validation_frequency == 0:
            if local_rank == 0:
                print(
                    "Train Epoch: [{0}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                        epoch,
                        loss=losses,
                    )
                )

            blip_model.eval()
            recalls_at10_list = []
            recalls_at50_list = []
            for relative_val_dataset, classic_val_dataset, idx in zip(
                relative_val_datasets, classic_val_datasets, idx_to_dress_mapping
            ):
                index_feats, index_names = extract_index_fiq(
                    classic_val_dataset, blip_model.module, txt_processors, local_rank
                )
                recall_at10, recall_at50 = compute_blip_fiq(
                    relative_val_dataset,
                    blip_model.module,
                    index_feats,
                    index_names,
                    txt_processors,
                    local_rank,
                )

                recalls_at10_list.append(recall_at10)
                recalls_at50_list.append(recall_at50)
                if local_rank == 0:
                    print(
                        f"{types[idx]}:",
                        "    R@10: ",
                        recall_at10,
                        "    R@50: ",
                        recall_at50,
                    )
                torch.cuda.empty_cache()
            r_10, r_50 = mean(recalls_at10_list), mean(recalls_at50_list)
            r_average = (r_10 + r_50) / 2

            if local_rank == 0:
                print("R@10: ", r_10, "    R@50: ", r_50)
            if r_average > best_r_average:
                best_r_average = round(r_average, 5)
                if local_rank == 0:
                    print("Best Mean Now: ", best_r_average, "*" * 30)
            else:
                if local_rank == 0:
                    print(
                        "Mean Now: ",
                        r_average,
                        " Best Mean Before: ",
                        best_r_average,
                        "-" * 20,
                    )
        torch.cuda.empty_cache()


def train_st(
    num_epochs,
    blip_model_name,
    learning_rate,
    batch_size,
    validation_frequency: int,
    **kwargs,
):
    global best_r_average
    device = kwargs["device"]
    local_rank = kwargs["local_rank"]
    time_str = datetime.now().strftime("%m-%d-%H")

    if local_rank != 0:
        sys.stdout = open(os.devnull, "w")

    blip_model, _, txt_processors = load_model_and_preprocess(
        name=blip_model_name, model_type="pretrain", is_eval=False, device=device
    )
    update_method = getattr(blip_model, "_update_f_former", None)
    if callable(update_method):
        blip_model._update_f_former()

    blip_model = torch.nn.parallel.DistributedDataParallel(
        blip_model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    input_dim = 224
    target_ratio = kwargs["target_ratio"]
    preprocess = targetpad_transform(target_ratio, input_dim)
    dataset_name = kwargs["dataset"]
    relative_train_dataset = STDataset("train", "relative", preprocess, dataset_name)
    train_sampler = torch.utils.data.distributed.DistributedSampler(relative_train_dataset, shuffle=True)
    relative_train_loader = DataLoader(
        dataset=relative_train_dataset,
        batch_size=batch_size,
        num_workers=kwargs["num_workers"],
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
    )

    if local_rank != 0:
        sys.stdout = sys.__stdout__  # 上述代码print不会输出

    optimizer = optim.AdamW(
        [
            {
                "params": filter(lambda p: p.requires_grad, blip_model.parameters()),
                "lr": learning_rate,
                "betas": (0.9, 0.98),
                "eps": 1e-7,
                "weight_decay": 0.05,
            }
        ]
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        pct_start=1.5 / num_epochs,
        div_factor=100.0,
        steps_per_epoch=len(relative_train_loader),
        epochs=num_epochs,
    )
    scaler = torch.cuda.amp.GradScaler()
    best_r_average = 0
    for epoch in range(num_epochs):
        losses = AverageMeter()
        train_sampler.set_epoch(epoch)

        for idx, (ref_img, tar_img, mod, ref_cap, tar_cap) in enumerate(
            tqdm(
                relative_train_loader,
                desc=f"Train [{epoch}]",
                disable=(local_rank != 0),
            )
        ):
            optimizer.zero_grad()
            blip_model.train()
            ref_img = ref_img.to(device, non_blocking=True)
            tar_img = tar_img.to(device, non_blocking=True)
            # ?注意文本处理形式
            mod = select_caps_compose(mod)
            mod_input = [txt_processors["eval"](m) for m in mod]
            cap_ref_inputs = [txt_processors["eval"](caption) for caption in ref_cap]
            cap_tar_inputs = [txt_processors["eval"](caption) for caption in tar_cap]

            with torch.cuda.amp.autocast():
                loss_dict, _ = blip_model(
                    {
                        "image": ref_img,
                        "target": tar_img,
                        "text_input": mod_input,
                        "tar_cap": cap_tar_inputs,
                        "ref_cap": cap_ref_inputs,
                    },
                    mode="single",  # TODO
                )
                loss = sum(loss_dict.values())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.update(loss.detach().cpu().item())

        if epoch % validation_frequency == 0 and local_rank == 0:
            print(
                "Train Epoch: [{0}]\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                    epoch,
                    loss=losses,
                )
            )
            save_model(
                f"/mnt/longvideo/chenyanzhe/Multiturn/ckpt/blip_{dataset_name}/{time_str}/epoch{epoch}.pth",
                epoch,
                blip_model.module,
            )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    setup_seed(42)

    parser = ArgumentParser()
    parser.add_argument("--local_rank", default=os.getenv("LOCAL_RANK", -1), type=int)
    parser.add_argument("--loss-align", default=0.6, type=float)
    parser.add_argument("--loss-rtc", default=0.6, type=float)
    parser.add_argument("--loss-itm", default=1, type=float)
    parser.add_argument(
        "--validation-frequency",
        default=1,
        type=int,
        help="Validation frequency expressed in epochs",
    )
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument(
        "--transform",
        default="targetpad",
        type=str,
    )
    parser.add_argument(
        "--save-training",
        dest="save_training",
        action="store_true",
        help="Whether save the training model",
    )
    parser.add_argument(
        "--save-best",
        dest="save_best",
        action="store_true",
        help="Save only the best model during training",
    )
    parser.add_argument("--data-path", type=str, default="/mnt/longvideo/chenyanzhe/Multiturn/data/")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--blip-model-name", default="blip2_cir_align_prompt", type=str)

    #########################Parameters#########################
    parser.add_argument("--num-epochs", default=50, type=int, help="number training epochs")
    parser.add_argument("--dataset", type=str, default="fiq_origin")
    parser.add_argument("--learning-rate", default=1e-5, type=float)
    parser.add_argument("--batch-size", default=32, type=int)
    args = parser.parse_args()

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    training_hyper_params = {
        "num_epochs": args.num_epochs,
        "num_workers": args.num_workers,
        "blip_model_name": args.blip_model_name,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "validation_frequency": args.validation_frequency,
        "transform": args.transform,
        "target_ratio": args.target_ratio,
        "save_training": args.save_training,
        "save_best": args.save_best,
        "data_path": args.data_path,
        "loss_rtc": args.loss_rtc,
        "loss_align": args.loss_align,
        "loss_itm": args.loss_itm,
        "dataset": args.dataset.lower(),
        "device": device,
        "local_rank": args.local_rank,
    }

    if args.dataset.lower() == "fiq" or args.dataset.lower() == "fc":
        if args.local_rank == 0:
            print(f"Finetune BLIP2 model on {args.dataset}")
        train_compose(**training_hyper_params)
    elif args.dataset.lower() == "200k" or args.dataset.lower() == "shoes":
        if args.local_rank == 0:
            print(f"Finetune BLIP2 model on {args.dataset}")
        train_single(**training_hyper_params)

    elif args.dataset.lower() == "fiq_origin":
        if args.local_rank == 0:
            print(f"Finetune BLIP2 model on {args.dataset}")
        train_fiq(**training_hyper_params)

    elif args.dataset.lower() == "fc-st" or args.dataset.lower() == "200k-st":
        if args.local_rank == 0:
            print(f"Finetune BLIP2 model on {args.dataset}")
        train_st(**training_hyper_params)
