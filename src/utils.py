import multiprocessing
import random
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
import os
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import re


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# 设置随机数种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def select_caps(modifiers):
    selected_modifiers = []
    for modifier in modifiers:
        pattern = r'\d+\.\s*([^0-9]+)'  # 使用正则表达式匹配数字序号和子句
        matches = re.findall(pattern, modifier)
        if len(matches) >= 2:
            # TODO 注意选择的句子数量
            random_sentences = random.sample(matches, 2)
            cleaned_sentences = [sentence.replace('.', '') for sentence in random_sentences]
            selected = " and ".join(cleaned_sentences)
        else:
            selected = modifier
        selected = selected.replace("  ", " ")
        selected_modifiers.append(selected)
    return selected_modifiers


def add_prompt_cap(cap1=None, cap2=None, cap3=None, prompt=None):
    if cap2 is None:
        return [f"{prompt}: " + a for a in cap1]
    elif cap1 is not None and cap2 is not None and cap3 is None:
        return ["FIRST: " + a + " SECOND: " + b for a, b in zip(cap1, cap2)]
    elif cap1 is not None and cap2 is not None and cap3 is not None:
        return ["FIRST: " + a + " SECOND: " + b + " THIRD: " + c for a, b, c in zip(cap1, cap2, cap3)]


def select_caps_compose(modifiers):
    selected_modifiers = []
    for modifier in modifiers:
        pattern = r'\d+\.\s*([^0-9]+)'  # 使用正则表达式匹配数字序号和子句
        matches = re.findall(pattern, modifier)
        if len(matches) >= 2:  # TODO 注意选择的句子数量
            if "4." in modifier:
                random_sentences = [random.choice(matches[:len(matches) - 1]), matches[-1]]
            else:
                random_sentences = random.sample(matches, 2)
            cleaned_sentences = [sentence.replace('.', '') for sentence in random_sentences]
            selected = " and ".join(cleaned_sentences)
        else:
            selected = modifier
        selected = selected.replace("  ", " ")
        selected_modifiers.append(selected)
    return selected_modifiers


def concatenate_numbered_descriptions(modifiers):
    cat_mods = []
    for text in modifiers:
        # 使用正则表达式提取所有序号后的内容
        descriptions = re.findall(r'\d+\.\s(.+?)(?=\d+\.\s|\Z)', text)
        # TODO 不要太长的modified text
        if len(descriptions) >= 2:
            descriptions = [descriptions[0], descriptions[-1]]

        concatenated_description = ' and '.join(descriptions)  # 使用and连接所有提取的描述
        concatenated_description = concatenated_description.replace("  ", " ")
        cat_mods.append(concatenated_description)
    return cat_mods


def extract_index_features(dataset, clip_model):
    feature_dim = clip_model.visual.output_dim
    classic_val_loader = DataLoader(
        dataset=dataset,
        batch_size=32,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True,
        collate_fn=collate_fn,
    )
    # index_features = torch.empty((0, feature_dim)).to(device, non_blocking=True)
    index_features = torch.empty((0, feature_dim))
    index_names = []
    for names, images, caption in tqdm(classic_val_loader, desc="Index"):
        # for names, images in classic_val_loader:
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            batch_features = clip_model.encode_image(images).cpu()
            index_features = torch.vstack((index_features, batch_features))
            index_names.extend(names)
    return index_features, index_names


def extract_index_blip_features(dataset, blip_model, cal_type="cpu"):
    classic_val_loader = DataLoader(
        dataset=dataset,
        batch_size=512,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    index_features = []
    index_features_raw = []
    index_names = []

    for names, images in tqdm(classic_val_loader, desc="Index"):
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            image_features, image_embeds_frozen = blip_model.extract_target_features(images)
            if cal_type == "cpu":
                image_features, image_embeds_frozen = image_features.cpu(), image_embeds_frozen.cpu()
            index_features.append(image_features)
            index_features_raw.append(image_embeds_frozen)
            index_names.extend(names)

    index_features = torch.vstack(index_features)
    index_features_raw = torch.vstack(index_features_raw)
    return (index_features, index_features_raw), index_names


def extract_index_blip_fusion_features(dataset, blip_model, txt_processors):
    classic_val_loader = DataLoader(
        dataset=dataset,
        batch_size=256,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    index_fusion_features = []
    index_features_raw = []
    index_names = []

    for names, images, captions in tqdm(classic_val_loader, desc="Index"):
        images = images.to(device, non_blocking=True)
        input_captions = [txt_processors["eval"](caption) for caption in captions]
        with torch.no_grad():
            _, image_embeds_frozen = blip_model.extract_target_features(images)
            index_fusion_feats = blip_model.inference(
                {"target_embeds": image_embeds_frozen, "tar_cap": input_captions},
                mode="target",
            )
            index_fusion_feats, image_embeds_frozen = index_fusion_feats.cpu(), image_embeds_frozen.cpu()
            index_fusion_features.append(index_fusion_feats)
            index_features_raw.append(image_embeds_frozen)
            index_names.extend(names)

    index_fusion_features = torch.vstack(index_fusion_features)
    index_features_raw = torch.vstack(index_features_raw)
    return (index_fusion_features, index_features_raw), index_names


def extract_index_fiq(dataset, blip_model, txt_processors, local_rank):
    classic_val_loader = DataLoader(dataset=dataset, batch_size=256, num_workers=8, pin_memory=True,
                                    collate_fn=collate_fn)
    index_features_raw = []
    index_names = []
    tar_feats = []

    for names, images, _ in tqdm(classic_val_loader, desc="Index", disable=(local_rank != 0)):
    # for names, images, captions in classic_val_loader:
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            tar_feat, image_embeds_frozen = blip_model.extract_target_features(images)
            image_embeds_frozen = image_embeds_frozen.cpu()
            tar_feat = tar_feat.cpu()
            index_features_raw.append(image_embeds_frozen)
            tar_feats.append(tar_feat)
            index_names.extend(names)

    index_features_raw = torch.vstack(index_features_raw)
    tar_feats = torch.vstack(tar_feats)
    return (tar_feats, index_features_raw), index_names


def extract_index_blip_features_ddp(classic_val_loader, blip_model, save_path_prefix, path2code_dict):
    index_names = []
    # for names, images in tqdm(classic_val_loader, desc="Index"):
    for names, images in classic_val_loader:
        index_names.extend(names)
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            image_features, image_embeds_frozen = blip_model.extract_target_features(images)
            for name, feature, embed in zip(names, image_features, image_embeds_frozen):
                save_code = path2code_dict[name]
                torch.save(feature.detach().cpu(), os.path.join(save_path_prefix, f'{save_code}_feature.pth'))
                torch.save(embed.detach().cpu(), os.path.join(save_path_prefix, f'{save_code}_embed_frozen.pth'))
    # return index_names


def read_features(index_names, prefix_path, path2code_dict, mode="both"):
    if mode == "both":
        index_features = []
        index_features_raw = []
        for name in index_names:
            save_code = path2code_dict[name]
            feature_path = os.path.join(prefix_path, f'{save_code}_feature.pth')
            feature = torch.load(feature_path)
            index_features.append(feature)
            raw_feature_path = os.path.join(prefix_path, f'{save_code}_embed_frozen.pth')
            feature_raw = torch.load(raw_feature_path)
            index_features_raw.append(feature_raw)
        index_features = torch.stack(index_features)
        index_features_raw = torch.stack(index_features_raw)
        return index_features, index_features_raw

    elif mode == "index":
        index_features = []
        for name in index_names:
            save_code = path2code_dict[name]
            feature_path = os.path.join(prefix_path, f'{save_code}_feature.pth')
            feature = torch.load(feature_path)
            index_features.append(feature)
        index_features = torch.stack(index_features)
        return index_features


def read_features_fusion(names, prefix_path, path2code_dict):
    features = []
    for name in names:
        save_code = path2code_dict[name]
        feature_path = os.path.join(prefix_path, f'{save_code}_fusion.pth')
        feature = torch.load(feature_path)
        features.append(feature)
    features = torch.stack(features)
    return features


def extract_index_fuse_features(dataset, fuse_model):
    """
    Extract FashionIQ or CIRR index features
    :param dataset: FashionIQ or CIRR dataset in 'classic' mode
    :param clip_model: CLIP model
    :return: a tensor of features and a list of images
    """
    classic_val_loader = DataLoader(
        dataset=dataset,
        batch_size=32,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    index_features = []
    index_names = []
    for names, images in tqdm(classic_val_loader):
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            image_features = fuse_model.retrieval_transformer.encode_image(images)
            index_features.append(image_features)
            index_names.extend(names)

    index_features = torch.vstack(index_features)

    return (index_features), index_names


def element_wise_sum(image_features, text_features, weight=0.5):
    """
    Normalized element-wise sum of image features and text features
    :param image_features: non-normalized image features
    :param text_features: non-normalized text features
    :return: normalized element-wise sum of image and text features
    """
    return F.normalize(image_features * weight + text_features * (1 - weight), dim=-1)


def generate_randomized_fiq_caption(flattened_captions: List[str]) -> List[str]:
    """
    Function which randomize the FashionIQ training captions in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1
    (d) cap2
    :param flattened_captions: the list of caption to randomize, note that the length of such list is 2*batch_size since
     to each triplet are associated two captions
    :return: the randomized caption list (with length = batch_size)
    """
    captions = []
    for i in range(0, len(flattened_captions), 2):
        random_num = random.random()
        if random_num < 0.25:
            captions.append(
                f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}")
        elif 0.25 < random_num < 0.5:
            captions.append(
                f"{flattened_captions[i + 1].strip('.?, ').capitalize()} and {flattened_captions[i].strip('.?, ')}")
        elif 0.5 < random_num < 0.75:
            captions.append(f"{flattened_captions[i].strip('.?, ').capitalize()}")
        else:
            captions.append(f"{flattened_captions[i + 1].strip('.?, ').capitalize()}")
    return captions


def collate_fn(batch: list):
    """
    Discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def update_train_running_results(train_running_results: dict, loss: torch.tensor, images_in_batch: int):
    """
    Update `train_running_results` dict during training
    :param train_running_results: logging training dict
    :param loss: computed loss for batch
    :param images_in_batch: num images in the batch
    """
    train_running_results['accumulated_train_loss'] += loss.to('cpu',
                                                               non_blocking=True).detach().item() * images_in_batch
    train_running_results["images_in_epoch"] += images_in_batch


def set_train_bar_description(train_bar, epoch: int, num_epochs: int, train_running_results: dict):
    """
    Update tqdm train bar during training
    :param train_bar: tqdm training bar
    :param epoch: current epoch
    :param num_epochs: numbers of epochs
    :param train_running_results: logging training dict
    """
    train_bar.set_description(
        desc=f"[{epoch}/{num_epochs}] "
             f"train loss: {train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch']:.3f} "
    )


def update_train_running_results_dict(train_running_results: dict, loss_dict: dict, images_in_batch: int):
    """
    Update `train_running_results` dict during training
    :param train_running_results: logging training dict
    :param loss: computed loss for batch
    :param images_in_batch: num images in the batch
    """
    for key in loss_dict.keys():
        if key not in train_running_results:
            train_running_results[key] = 0
        train_running_results[key] += loss_dict[key].to('cpu', non_blocking=True).detach().item() * images_in_batch

    train_running_results["images_in_epoch"] += images_in_batch


def set_train_bar_description_dict(train_bar, epoch: int, num_epochs: int, train_running_results: dict):
    """
    Update tqdm train bar during training
    :param train_bar: tqdm training bar
    :param epoch: current epoch
    :param num_epochs: numbers of epochs
    :param train_running_results: logging training dict
    """
    images_in_epoch = train_running_results['images_in_epoch']
    bar_content = ''
    for key in train_running_results:
        if key != 'images_in_epoch':
            bar_content += f'{key}: {train_running_results[key] / images_in_epoch:.3f}, '
    train_bar.set_description(
        desc=f"[{epoch}/{num_epochs}] "
             f"{bar_content}"
    )


def save_model(model_path: str, cur_epoch: int, model_to_save: nn.Module):
    folder_path = os.path.dirname(model_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    model_name = model_to_save.__class__.__name__
    torch.save({
        'epoch': cur_epoch,
        model_name: model_to_save.state_dict(),
    }, model_path)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_list_to_txt(data_list, file_path):
    with open(file_path, 'w') as f:
        for item in data_list:
            if isinstance(item, str):
                f.write(item + '\n')
            elif isinstance(item, list):
                f.write(' '.join(map(str, item)) + '\n')


def read_txt(input_file_path, mode="single"):
    data_list = []
    with open(input_file_path, 'r') as infile:
        lines = infile.readlines()  # 读取所有行
        for line in lines:
            # 将每一行的内容分割成列表
            if mode == "single":
                elements = line.strip()
            else:
                elements = line.strip().split()
            # 只保留前10个元素
            data_list.append(elements)
    return data_list
