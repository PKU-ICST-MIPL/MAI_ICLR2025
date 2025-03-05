from operator import itemgetter
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json

from utils import collate_fn, device, concatenate_numbered_descriptions, read_features_fusion, read_features, add_prompt_cap


def compute_blip_compose_multi(relative_val_dataset, blip_model, index_feats,
                               index_names, txt_processors, dataset_name):
    pred_sim, target_names, reference_names, captions_all = generate_blip_compose_multi(
        blip_model, relative_val_dataset, index_names, index_feats, txt_processors, dataset_name)

    distances = 1 - pred_sim
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    labels = torch.tensor(sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at20 = (torch.sum(labels[:, :20]) / len(labels)).item() * 100
    return recall_at1, recall_at5, recall_at10, recall_at20


def get_visualize_names(relative_val_dataset, blip_model, index_feats, index_names, txt_processors, dataset_name):
    pred_sim, target_names, reference_names, captions_all = generate_blip_compose_multi(
        blip_model, relative_val_dataset, index_names, index_feats, txt_processors, dataset_name)
    distances = 1 - pred_sim
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]
    return target_names, reference_names, captions_all, sorted_index_names


def generate_blip_compose_multi(blip_model, relative_val_dataset, index_names,
                                index_features, txt_processors, dataset_name):
    relative_val_loader = DataLoader(dataset=relative_val_dataset,
                                     batch_size=64,
                                     num_workers=8,
                                     pin_memory=True,
                                     collate_fn=collate_fn,
                                     shuffle=False)

    name_to_feat = dict(zip(index_names, index_features[-1]))
    target_names_list = []
    reference_names_all = []
    distance = []
    captions_all = []

    for ref_name, tar1_name, tar2_name, batch_target_names, mod1, mod2, mod3, cap0, cap1, cap2, cap3 in tqdm(
            relative_val_loader, desc="Val"):
        input_mod1 = concatenate_numbered_descriptions(mod1)
        input_mod1 = [txt_processors["eval"](caption) for caption in input_mod1]
        input_mod2 = concatenate_numbered_descriptions(mod2)
        input_mod2 = [txt_processors["eval"](caption) for caption in input_mod2]
        input_mod3 = concatenate_numbered_descriptions(mod3)
        input_mod3 = [txt_processors["eval"](caption) for caption in input_mod3]

        input_cap0 = [txt_processors["eval"](caption) for caption in cap0]
        input_cap1 = [txt_processors["eval"](caption) for caption in cap1]
        input_cap2 = [txt_processors["eval"](caption) for caption in cap2]

        if dataset_name == "fc":
            concat_cap = add_prompt_cap(cap0, cap1)
            input_concat_cap = [txt_processors["eval"](caption) for caption in concat_cap]
        else:
            concat_cap = add_prompt_cap(cap0, cap1, cap2)
            input_concat_cap = [txt_processors["eval"](caption) for caption in concat_cap]

        with torch.no_grad():
            if len(input_mod3) == 1:
                ref_feats_raw = itemgetter(*ref_name)(name_to_feat).unsqueeze(0)
                tar1_feats_raw = itemgetter(*tar1_name)(name_to_feat).unsqueeze(0)
                tar2_feats_raw = itemgetter(*tar2_name)(name_to_feat).unsqueeze(0)
            else:
                ref_feats_raw = torch.stack(itemgetter(*ref_name)(name_to_feat))
                tar1_feats_raw = torch.stack(itemgetter(*tar1_name)(name_to_feat))
                tar2_feats_raw = torch.stack(itemgetter(*tar2_name)(name_to_feat))

            ref_feats_raw = ref_feats_raw.to(blip_model.device)
            tar1_feats_raw = tar1_feats_raw.to(blip_model.device)
            tar2_feats_raw = tar2_feats_raw.to(blip_model.device)

            fus_token0 = blip_model.inference(
                {
                    "img_embeds": ref_feats_raw,
                    "ref_cap": input_cap0
                },
                mode="token")
            fus_token1 = blip_model.inference(
                {
                    "img_embeds": tar1_feats_raw,
                    "ref_cap": input_cap1,
                    "fusion": fus_token0
                },
                mode="token")
            if dataset_name == "fc":
                fus_token1 = fus_token1.to(blip_model.device)
                batch_distance = blip_model.inference(
                    {
                        "target_feats": index_features[0],
                        "modified_text": input_mod3,
                        "ref_cap": input_concat_cap,
                        "fusion": fus_token1
                    },
                    mode="multi")
            else:
                fus_token2 = blip_model.inference(
                    {
                        "img_embeds": tar2_feats_raw,
                        "ref_cap": input_cap2,
                        "fusion": fus_token1
                    },
                    mode="token")
                batch_distance = blip_model.inference(
                    {
                        "target_feats": index_features[0],
                        "modified_text": input_mod3,
                        "ref_cap": input_concat_cap,
                        "fusion": fus_token2
                    },
                    mode="multi")

            distance.append(batch_distance)
            captions_all += input_mod3

        target_names_list.extend(batch_target_names)
        reference_names_all.extend(tar2_name)

    distance = torch.vstack(distance).cpu()
    return distance, target_names_list, reference_names_all, captions_all


def compute_blip_single_multi(
    relative_val_dataset,
    blip_model,
    index_feats,
    index_names,
    txt_processors,
    dataset_name,
):
    pred_sim, target_names, reference_names, captions_all = generate_blip_single_multi(
        blip_model, relative_val_dataset, index_names, index_feats,
        txt_processors, dataset_name)

    distances = 1 - pred_sim
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(
            index_names)).reshape(len(target_names), -1))
    assert torch.equal(
        torch.sum(labels, dim=-1).int(),
        torch.ones(len(target_names)).int())

    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at20 = (torch.sum(labels[:, :20]) / len(labels)).item() * 100
    return recall_at1, recall_at5, recall_at10, recall_at20


def generate_blip_single_multi(
    blip_model,
    relative_val_dataset,
    index_names,
    index_features,
    txt_processors,
    dataset_name,
):
    relative_val_loader = DataLoader(
        dataset=relative_val_dataset,
        batch_size=64,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
        shuffle=False,
    )

    name_to_feat = dict(zip(index_names, index_features[-1]))
    target_names_list = []
    reference_names_all = []
    distance = []
    captions_all = []

    for (
        reference_name,
        target_name,
        reference_cap,
        target_cap,
        ref_name,
        tar1_name,
        tar2_name,
        tar3_name,
        mod1,
        mod2,
        mod3,
        cap0,
        cap1,
        cap2,
        cap3,
    ) in tqdm(relative_val_loader, desc="Val"):
        input_mod1 = concatenate_numbered_descriptions(mod1)
        input_mod1 = [txt_processors["eval"](caption) for caption in input_mod1]
        input_mod2 = concatenate_numbered_descriptions(mod2)
        input_mod2 = [
            txt_processors["eval"](caption) for caption in input_mod2
        ]
        input_mod3 = concatenate_numbered_descriptions(mod3)
        input_mod3 = [
            txt_processors["eval"](caption) for caption in input_mod3
        ]

        input_cap0 = [txt_processors["eval"](caption) for caption in cap0]
        input_cap1 = [txt_processors["eval"](caption) for caption in cap1]
        input_cap2 = [txt_processors["eval"](caption) for caption in cap2]
        input_last_cap = [txt_processors["eval"](caption) for caption in reference_cap]

        with torch.no_grad():
            if len(input_mod3) == 1:
                ref_feats_raw = itemgetter(*ref_name)(name_to_feat).unsqueeze(0)
                tar1_feats_raw = itemgetter(*tar1_name)(name_to_feat).unsqueeze(0)
                tar2_feats_raw = itemgetter(
                    *tar2_name)(name_to_feat).unsqueeze(0)
                last_ref_raw = itemgetter(
                    *reference_name)(name_to_feat).unsqueeze(0)
            else:
                ref_feats_raw = torch.stack(
                    itemgetter(*ref_name)(name_to_feat))
                tar1_feats_raw = torch.stack(
                    itemgetter(*tar1_name)(name_to_feat))
                tar2_feats_raw = torch.stack(
                    itemgetter(*tar2_name)(name_to_feat))
                last_ref_raw = torch.stack(
                    itemgetter(*reference_name)(name_to_feat))

            ref_feats_raw = ref_feats_raw.to(blip_model.device)
            tar1_feats_raw = tar1_feats_raw.to(blip_model.device)
            tar2_feats_raw = tar2_feats_raw.to(blip_model.device)
            last_ref_raw = last_ref_raw.to(blip_model.device)

            fus_token0 = blip_model.inference(
                {
                    "img_embeds": ref_feats_raw,
                    "ref_cap": input_cap0
                },
                mode="token")
            fus_token1 = blip_model.inference(
                {
                    "img_embeds": tar1_feats_raw,
                    "ref_cap": input_cap1,
                    "fusion": fus_token0
                },
                mode="token")
            batch_distance = blip_model.inference(
                {
                    "img_embeds": last_ref_raw,
                    "ref_cap": input_last_cap,
                    "fusion": fus_token1,
                    "target_feats": index_features[0],
                    'text_input': input_mod3
                },
                mode="single")

            distance.append(batch_distance)
            captions_all += input_mod3

        target_names_list.extend(target_name)
        reference_names_all.extend(reference_name)

    distance = torch.vstack(distance).cpu()
    return distance, target_names_list, reference_names_all, captions_all


def compute_blip_st(relative_val_dataset, blip_model, index_feats, index_names, txt_processors):
    pred_sim, target_names, reference_names, captions_all = generate_blip_st(
        blip_model, relative_val_dataset, index_names, index_feats, txt_processors
    )

    distances = 1 - pred_sim
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    labels = torch.tensor(
        sorted_index_names
        == np.repeat(np.array(target_names), len(index_names)).reshape(
            len(target_names), -1
        )
    )
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at20 = (torch.sum(labels[:, :20]) / len(labels)).item() * 100
    return recall_at1, recall_at5, recall_at10, recall_at20


def generate_blip_st(
    blip_model, relative_val_dataset, index_names, index_features, txt_processors
):
    relative_val_loader = DataLoader(
        dataset=relative_val_dataset,
        batch_size=16,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
        shuffle=False,
    )
    name_to_feat = dict(zip(index_names, index_features[-1]))
    target_names_list = []
    reference_names_all = []
    distance = []
    captions_all = []
    # for (ref_name, tar_name, mod, ref_cap, _) in relative_val_loader:
    for ref_name, tar_name, mod, ref_cap, _ in tqdm(relative_val_loader, desc="Val"):
        input_mods = concatenate_numbered_descriptions(mod)
        input_mods = [txt_processors["eval"](caption) for caption in input_mods]
        input_ref_cap = [txt_processors["eval"](caption) for caption in ref_cap]

        with torch.no_grad():
            if len(input_mods) == 1:
                ref_feats_raw = itemgetter(*ref_name)(name_to_feat).unsqueeze(0)
            else:
                ref_feats_raw = torch.stack(itemgetter(*ref_name)(name_to_feat))
            ref_feats_raw = ref_feats_raw.to(blip_model.device)
            batch_distance = blip_model.inference(
                {
                    "img_embeds": ref_feats_raw,
                    "ref_cap": input_ref_cap,
                    "fusion": None,
                    "text_input": input_mods,
                    "target_feats": index_features[0],  # TODO
                },
                mode="single",
            )
            distance.append(batch_distance)
            captions_all += input_mods
        target_names_list.extend(tar_name)
        reference_names_all.extend(ref_name)
    distance = torch.vstack(distance).cpu()
    return distance, target_names_list, reference_names_all, captions_all


def compute_blip_MTFiq(relative_val_dataset, blip_model, index_feats, index_names, txt_processors):
    pred_sim, target_names, reference_names, captions_all = generate_blip_MTFiq(
        blip_model, relative_val_dataset, index_names, index_feats, txt_processors
    )
    distances = 1 - pred_sim
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    labels = torch.tensor(
        sorted_index_names
        == np.repeat(np.array(target_names), len(index_names)).reshape(
            len(target_names), -1
        )
    )
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at8 = (torch.sum(labels[:, :8]) / len(labels)).item() * 100
    return recall_at5, recall_at8


def generate_blip_MTFiq(blip_model, relative_val_dataset, index_names, index_features, txt_processors):
    relative_val_loader = DataLoader(
        dataset=relative_val_dataset,
        batch_size=16,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
        shuffle=False,
    )
    name_to_feat = dict(zip(index_names, index_features[-1]))
    target_names_list = []
    reference_names_all = []
    distance = []
    captions_all = []
    for (
        reference_names_list,
        tar_name,
        modifiers_list,
        reference_captions_list,
    ) in tqdm(relative_val_loader, desc="Val"):

        input_captions = modifiers_list[-1]
        ref_name = reference_names_list[-1]
        input_mods = [txt_processors["eval"](caption) for caption in input_captions]
        ref_cap = reference_captions_list[-1]
        input_ref_cap = [txt_processors["eval"](caption) for caption in ref_cap]

        with torch.no_grad():
            if len(input_mods) == 1:
                ref_feats_raw = itemgetter(*ref_name)(name_to_feat).unsqueeze(0)
            else:
                ref_feats_raw = torch.stack(itemgetter(*ref_name)(name_to_feat))
            ref_feats_raw = ref_feats_raw.to(blip_model.device)
            batch_distance = blip_model.inference(
                {
                    "img_embeds": ref_feats_raw,
                    "ref_cap": input_ref_cap,
                    "fusion": None,
                    "text_input": input_mods,
                    "target_feats": index_features[0],
                },
                mode="single",
            )
            distance.append(batch_distance)
            captions_all += input_mods
        target_names_list.extend(tar_name)
        reference_names_all.extend(ref_name)
    distance = torch.vstack(distance).cpu()
    return distance, target_names_list, reference_names_all, captions_all


def compute_blip_fiq(
    relative_val_dataset,
    blip_model,
    index_feats,
    index_names,
    txt_processors,
    local_rank,
):
    pred_sim, target_names, reference_names, captions_all = generate_blip_fiq(
        blip_model,
        relative_val_dataset,
        index_names,
        index_feats,
        txt_processors,
        local_rank,
    )

    distances = 1 - pred_sim
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    labels = torch.tensor(sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    return recall_at10, recall_at50


def generate_blip_fiq(
    blip_model,
    relative_val_dataset,
    index_names,
    index_features,
    txt_processors,
    local_rank,
):
    relative_val_loader = DataLoader(
        dataset=relative_val_dataset,
        batch_size=4,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
        shuffle=False,
    )

    name_to_feat = dict(zip(index_names, index_features[-1]))
    target_names_list = []
    reference_names_all = []
    distance = []
    captions_all = []

    for ref_name, tar_name, mod, ref_cap, _ in tqdm(
        relative_val_loader, desc="Val", disable=(local_rank != 0)
    ):
        flattened_captions: list = np.array(mod).T.flatten().tolist()
        input_mods = [
            f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}"
            for i in range(0, len(flattened_captions), 2)
        ]
        input_mods = [txt_processors["eval"](caption) for caption in input_mods]
        input_ref_cap = [txt_processors["eval"](caption) for caption in ref_cap]

        with torch.no_grad():
            if len(input_mods) == 1:
                ref_feats_raw = itemgetter(*ref_name)(name_to_feat).unsqueeze(0)
            else:
                ref_feats_raw = torch.stack(itemgetter(*ref_name)(name_to_feat))
            ref_feats_raw = ref_feats_raw.to(blip_model.device)
            batch_distance = blip_model.inference(
                {
                    "img_embeds": ref_feats_raw,
                    "ref_cap": input_ref_cap,
                    "fusion": None,
                    "target_feats": index_features[0], # TODO
                    'text_input': input_mods
                },
                mode="fiq") # TODO 选择
            distance.append(batch_distance)
            captions_all += input_mods
        target_names_list.extend(tar_name)
        reference_names_all.extend(ref_name)
    distance = torch.vstack(distance).cpu()
    return distance, target_names_list, reference_names_all, captions_all


def compute_val_metrics_clip_compose(relative_val_dataset, clip_model,
                                     index_features, index_names,
                                     combining_function, tokenizer):
    predicted_features, target_names_list = generate_val_predictions_clip_compose(
        clip_model,
        relative_val_dataset,
        combining_function,
        index_names,
        index_features,
        tokenizer,
    )
    index_features = F.normalize(index_features, dim=-1).float().cpu()
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]
    labels = torch.tensor(sorted_index_names == np.repeat(
        np.array(target_names_list), len(index_names)).reshape(
            len(target_names_list), -1))
    assert torch.equal(
        torch.sum(labels, dim=-1).int(),
        torch.ones(len(target_names_list)).int())
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at20 = (torch.sum(labels[:, :20]) / len(labels)).item() * 100
    return recall_at1, recall_at5, recall_at10, recall_at20


def generate_val_predictions_clip_compose(clip_model, relative_val_dataset, combining_function, index_names, index_features, tokenizer):
    relative_val_loader = DataLoader(dataset=relative_val_dataset,
                                     batch_size=32,
                                     shuffle=False,
                                     num_workers=8,
                                     pin_memory=True,
                                     collate_fn=collate_fn)
    name_to_feat = dict(zip(index_names, index_features))
    predicted_features = torch.empty((0, clip_model.visual.output_dim))
    target_names_list = []

    for ref_name, tar1_name, tar2_name, batch_target_names, mod1, mod2, mod3, cap0, cap1, cap2, cap3 in tqdm(
            relative_val_loader, desc="Val"):
        reference_names = tar2_name
        input_captions = concatenate_numbered_descriptions(mod3)
        text_inputs = tokenizer(input_captions,
                                context_length=77).to(device,
                                                      non_blocking=True)

        with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs).cpu()
            if text_features.shape[0] == 1:
                reference_image_features = itemgetter(
                    *reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(
                    itemgetter(*reference_names)(name_to_feat))

            batch_predicted_features = combining_function(
                reference_image_features, text_features)
            batch_predicted_features = batch_predicted_features.cpu()
        predicted_features = torch.vstack(
            (predicted_features, F.normalize(batch_predicted_features,
                                             dim=-1)))
        target_names_list.extend(batch_target_names)

    return predicted_features, target_names_list


def compute_val_metrics_clip_single(relative_val_dataset, clip_model,
                                    index_features, index_names,
                                    combining_function, tokenizer):
    predicted_features, target_names_list = generate_val_predictions_clip_single(
        clip_model, relative_val_dataset, combining_function, index_names,
        index_features, tokenizer)
    index_features = F.normalize(index_features, dim=-1).float().cpu()
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]
    labels = torch.tensor(sorted_index_names == np.repeat(
        np.array(target_names_list), len(index_names)).reshape(
            len(target_names_list), -1))
    assert torch.equal(
        torch.sum(labels, dim=-1).int(),
        torch.ones(len(target_names_list)).int())
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at20 = (torch.sum(labels[:, :20]) / len(labels)).item() * 100
    return recall_at1, recall_at5, recall_at10, recall_at20


def generate_val_predictions_clip_single(clip_model, relative_val_dataset,
                                         combining_function, index_names,
                                         index_features, tokenizer):
    relative_val_loader = DataLoader(dataset=relative_val_dataset,
                                     batch_size=32,
                                     shuffle=False,
                                     num_workers=8,
                                     pin_memory=True,
                                     collate_fn=collate_fn)
    name_to_feat = dict(zip(index_names, index_features))
    predicted_features = torch.empty((0, clip_model.visual.output_dim))
    target_names_list = []

    for (reference_names, batch_target_names, reference_cap, target_cap,
         ref_img, tar1_img, tar2_img, tar3_img, mod1, mod2, mod3, cap0, cap1,
         cap2, cap3) in tqdm(relative_val_loader, desc="Val"):
        input_captions = concatenate_numbered_descriptions(mod3)
        text_inputs = tokenizer(input_captions, context_length=77).to(device, non_blocking=True)

        with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs).cpu()
            if text_features.shape[0] == 1:
                reference_image_features = itemgetter(
                    *reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(
                    itemgetter(*reference_names)(name_to_feat))

            batch_predicted_features = combining_function(
                reference_image_features, text_features)
            batch_predicted_features = batch_predicted_features.cpu()
        predicted_features = torch.vstack(
            (predicted_features, F.normalize(batch_predicted_features,
                                             dim=-1)))
        target_names_list.extend(batch_target_names)

    return predicted_features, target_names_list


def compute_val_metrics_clip_MT(
    relative_val_dataset,
    clip_model,
    index_features,
    index_names,
    combining_function,
    tokenizer,
):
    predicted_features, target_names_list = generate_val_clip_MT(
        clip_model,
        relative_val_dataset,
        combining_function,
        index_names,
        index_features,
        tokenizer,
    )
    index_features = F.normalize(index_features, dim=-1).float().cpu()
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]
    labels = torch.tensor(
        sorted_index_names
        == np.repeat(np.array(target_names_list), len(index_names)).reshape(
            len(target_names_list), -1
        )
    )
    assert torch.equal(
        torch.sum(labels, dim=-1).int(), torch.ones(len(target_names_list)).int()
    )
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at8 = (torch.sum(labels[:, :8]) / len(labels)).item() * 100
    return recall_at5, recall_at8


def generate_val_clip_MT(
    clip_model,
    relative_val_dataset,
    combining_function,
    index_names,
    index_features,
    tokenizer,
):
    relative_val_loader = DataLoader(
        dataset=relative_val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    name_to_feat = dict(zip(index_names, index_features))
    predicted_features = torch.empty((0, clip_model.visual.output_dim))
    target_names_list = []

    for (
        reference_names_list,
        batch_target_names,
        modifiers_list,
        reference_captions_list,
    ) in tqdm(relative_val_loader, desc="Val"):

        mods = modifiers_list[-1]
        ref_names = reference_names_list[-1]
        input_mods = tokenizer(mods, context_length=77).to(device, non_blocking=True)
        ref_caps = reference_captions_list[-1]
        input_ref_cap = tokenizer(ref_caps, context_length=77).to(device, non_blocking=True)

        with torch.no_grad():
            mod_features = clip_model.encode_text(input_mods).cpu()
            if mod_features.shape[0] == 1:
                reference_image_features = itemgetter(*ref_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*ref_names)(name_to_feat))

            batch_predicted_features = combining_function(reference_image_features, mod_features)
            batch_predicted_features = batch_predicted_features.cpu()
        predicted_features = torch.vstack(
            (predicted_features, F.normalize(batch_predicted_features, dim=-1))
        )
        target_names_list.extend(batch_target_names)

    return predicted_features, target_names_list
