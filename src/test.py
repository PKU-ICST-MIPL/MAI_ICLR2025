import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

from argparse import ArgumentParser
import torch

from lavis.models import load_model_and_preprocess
from dataset import (
    targetpad_transform,
    ComposeDataset,
    SingleDataset,
    STDataset,
    ExistingFIQ,
)
from utils import setup_seed, device, extract_index_blip_fusion_features
from validate import (
    compute_blip_compose_multi,
    compute_blip_single_multi,
    compute_blip_st,
    compute_blip_MTFiq
)


def test_compose(blip_model_name, model_path, dataset_name, device):
    blip_model, _, txt_processors = load_model_and_preprocess(
        name=blip_model_name, model_type="pretrain", is_eval=False, device=device
    )
    checkpoint_path = model_path
    checkpoint = torch.load(checkpoint_path, map_location=device)
    blip_model.load_state_dict(checkpoint[blip_model.__class__.__name__], strict=False)
    input_dim = 224
    preprocess = targetpad_transform(1.25, input_dim)
    relative_val_dataset = ComposeDataset("val", "relative", preprocess, dataset_name)
    classic_val_dataset = ComposeDataset('val', 'classic', preprocess, dataset_name)

    with torch.no_grad():
        blip_model.eval()
        index_feats, index_names = extract_index_blip_fusion_features(classic_val_dataset, blip_model, txt_processors)

        recall_at1, recall_at5, recall_at10, recall_at20 = compute_blip_compose_multi(
            relative_val_dataset, blip_model, index_feats, index_names, txt_processors, dataset_name
        )
        r_average = (recall_at1 + recall_at5 + recall_at10 + recall_at20) / 4
        print("R@1:", recall_at1, "  R@5: ", recall_at5, "  R@10:", recall_at10, "  R@20: ", recall_at20)
        print("Mean Now: ", r_average, "*" * 30)


def test_single(blip_model_name, model_path, dataset_name, device):
    blip_model, _, txt_processors = load_model_and_preprocess(
        name=blip_model_name, model_type="pretrain", is_eval=False, device=device
    )
    checkpoint_path = model_path
    checkpoint = torch.load(checkpoint_path, map_location=device)
    blip_model.load_state_dict(checkpoint[blip_model.__class__.__name__], strict=False)
    input_dim = 224
    preprocess = targetpad_transform(1.25, input_dim)
    relative_val_dataset = SingleDataset('val', 'relative', preprocess, dataset_name)
    classic_val_dataset = SingleDataset('val', 'classic', preprocess, dataset_name)

    with torch.no_grad():
        blip_model.eval()
        index_feats, index_names = extract_index_blip_fusion_features(classic_val_dataset, blip_model, txt_processors)

        recall_at1, recall_at5, recall_at10, recall_at20 = compute_blip_single_multi(
            relative_val_dataset, blip_model, index_feats, index_names, txt_processors, dataset_name
        )
        r_average = (recall_at1 + recall_at5 + recall_at10 + recall_at20) / 4
        print("R@1:", recall_at1, "  R@5: ", recall_at5, "  R@10:", recall_at10, "  R@20: ", recall_at20)
        print("Mean Now: ", r_average, "*" * 30)


def test_st(blip_model_name, model_path, dataset_name, device):
    blip_model, _, txt_processors = load_model_and_preprocess(
        name=blip_model_name, model_type="pretrain", is_eval=False, device=device
    )
    checkpoint_path = model_path
    checkpoint = torch.load(checkpoint_path, map_location=device)
    blip_model.load_state_dict(checkpoint[blip_model.__class__.__name__], strict=False)
    input_dim = 224
    preprocess = targetpad_transform(1.25, input_dim)
    relative_val_dataset = STDataset('val', 'relative', preprocess, dataset_name)
    classic_val_dataset = STDataset('val', 'classic', preprocess, dataset_name)

    with torch.no_grad():
        blip_model.eval()
        index_feats, index_names = extract_index_blip_fusion_features(
            classic_val_dataset, blip_model, txt_processors
        )

        recall_at1, recall_at5, recall_at10, recall_at20 = compute_blip_st(
            relative_val_dataset, blip_model, index_feats, index_names, txt_processors
        )
        r_average = (recall_at1 + recall_at5 + recall_at10 + recall_at20) / 4
        print("R@1:", recall_at1, "  R@5: ", recall_at5, "  R@10:", recall_at10, "  R@20: ", recall_at20)
        print("Mean Now: ", r_average, "*" * 30)


def test_existing(blip_model_name, model_path, device):
    blip_model, _, txt_processors = load_model_and_preprocess(
        name=blip_model_name, model_type="pretrain", is_eval=False, device=device
    )
    checkpoint_path = model_path
    checkpoint = torch.load(checkpoint_path, map_location=device)
    blip_model.load_state_dict(checkpoint[blip_model.__class__.__name__], strict=False)
    input_dim = 224
    preprocess = targetpad_transform(1.25, input_dim)
    turn_number = 3
    relative_val_dataset = ExistingFIQ('val', ['all'], 'relative', preprocess, turn_number)
    classic_val_dataset = ExistingFIQ('val', ['all'], 'classic', preprocess)

    with torch.no_grad():
        blip_model.eval()
        index_feats, index_names = extract_index_blip_fusion_features(
            classic_val_dataset, blip_model, txt_processors
        )
        recall_at5, recall_at8 = compute_blip_MTFiq(
            relative_val_dataset, blip_model, index_feats, index_names, txt_processors
        )
        r_average = (recall_at5 + recall_at8) / 2
        print(f"R@5: {recall_at5}")
        print(f"R@8: {recall_at8}")
        print("Mean Now: ", r_average, "*" * 30)


if __name__ == '__main__':
    setup_seed(42)

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='mt-fiq')  # TODO
    parser.add_argument("--blip-model-name", default="blip2_cir_align_prompt", type=str)
    parser.add_argument("--epoch-num", default="10", type=str)  # TODO
    parser.add_argument("--date", default="05-16-09", type=str)  # TODO
    args = parser.parse_args()
    # args.model_path = f"/mnt/longvideo/chenyanzhe/Multiturn/ckpt/blip_{args.dataset.lower()}/{args.date}/epoch{args.epoch_num}.pth"
    args.model_path = f"/mnt/longvideo/chenyanzhe/Multiturn/ckpt/blip_fiq/05-11-13/epoch10.pth"

    print("Model:", args.model_path)
    print(f"Testing {args.dataset} dataset")
    if args.dataset.lower() == 'fiq' or args.dataset.lower() == 'fc':
        test_compose(args.blip_model_name, args.model_path, args.dataset, device)
    elif args.dataset.lower() == 'fc-st' or args.dataset.lower() == '200k-st':
        test_st(args.blip_model_name, args.model_path, args.dataset, device)
    elif args.dataset.lower() == 'mt-fiq':
        test_existing(args.blip_model_name, args.model_path, device)
    else:
        test_single(args.blip_model_name, args.model_path, args.dataset, device)
