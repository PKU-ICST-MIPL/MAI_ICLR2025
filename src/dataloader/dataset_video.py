import ast
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dataloader.data_utils import FrameLoader, id2int, pre_caption, write_txt

Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombWarning


class WebVidCoVRDataset(Dataset):

    def __init__(
        self,
        transform,
        annotation: str,  # 生成的视频annotation地址, "annotation/webvid-covr"
        vid_dir: str,
        split: str,
        mode: str = "relative",
        max_words: int = 30,
        iterate: str = "pth2",
        vid_query_method: str = "middle",
        vid_frames: int = 1,
    ) -> None:
        super().__init__()

        self.transform = transform
        self.annotation_pth = annotation
        assert Path(annotation).exists(), f"Annotation file {annotation} does not exist"
        self.df = pd.read_csv(annotation)

        self.vid_dir = Path(vid_dir)
        assert self.vid_dir.exists(), f"Image directory {self.vid_dir} does not exist"
        vid_pths = self.vid_dir.glob("*/*.mp4")
        id2vidpth = {
            vid_pth.parent.stem + "/" + vid_pth.stem: vid_pth for vid_pth in vid_pths
        }
        assert len(id2vidpth) > 0, f"No videos found in {vid_dir}"

        assert split in [
            "train",
            "val",
            "test",
        ], f"Invalid split: {split}, must be one of train, val, or test"
        self.split = split
        self.mode = mode

        self.df["path1"] = self.df["pth1"].apply(lambda x: id2vidpth.get(x, None))  # type: ignore
        # ? 注意甄别
        self.df["path2"] = self.df["pth2"].apply(lambda x: id2vidpth.get(x, None))  # type: ignore
        # Count unique missing paths
        missing_pth1 = self.df[self.df["path1"].isna()]["pth1"].unique().tolist()
        missing_pth1.sort()
        total_pth1 = self.df["pth1"].nunique()

        missing_pth2 = self.df[self.df["path2"].isna()]["pth2"].unique().tolist()
        missing_pth2.sort()
        total_pth2 = self.df["pth2"].nunique()

        if len(missing_pth1) > 0:
            print(
                f"Missing {len(missing_pth1)} pth1's ({len(missing_pth1)/total_pth1 * 100:.1f}%), saving them to missing_pth1-{split}.txt"
            )
            if split == "test":
                raise ValueError(
                    f"Missing {len(missing_pth1)} pth1's ({len(missing_pth1)/total_pth1 * 100:.1f}%) in test split"
                )
            write_txt(missing_pth1, f"missing_pth1-{split}.txt")
        if len(missing_pth2) > 0:
            print(
                f"Missing {len(missing_pth2)} pth2's ({len(missing_pth2)/total_pth2 * 100:.1f}%), saving them to missing_pth2-{split}.txt"
            )
            if split == "test":
                raise ValueError(
                    f"Missing {len(missing_pth2)} pth2's ({len(missing_pth2)/total_pth2 * 100:.1f}%) in test split"
                )
            write_txt(missing_pth2, f"missing_pth2-{split}.txt")

        # Remove missing paths
        self.df = self.df[self.df["path1"].notna()]
        self.df = self.df[self.df["path2"].notna()]

        self.max_words = max_words

        if iterate in ["idx", "triplets"]:
            iterate = "idx"
            self.df["idx"] = self.df.index
        self.iterate = iterate
        self.target_txts = self.df[iterate].unique()
        assert iterate in self.df.columns, f"{iterate} not in {Path(annotation).stem}"
        self.df.sort_values(iterate, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.df["int1"] = self.df["pth1"].apply(lambda x: id2int(x, sub="0"))
        self.pairid2ref = self.df["int1"].to_dict()
        assert (
            self.df["int1"].nunique() == self.df["pth1"].nunique()
        ), "int1 is not unique"
        # int2id is a dict with key: int1, value: pth1
        self.int2id = self.df.groupby("int1")["pth1"].apply(set).to_dict()
        self.int2id = {k: list(v)[0] for k, v in self.int2id.items()}
        self.df.set_index(iterate, inplace=True)
        self.df[iterate] = self.df.index

        if split == "test":
            assert (
                len(self.target_txts) == self.df.shape[0]
            ), "Test split should have one caption per row"

        assert vid_query_method in [
            "middle",
            "random",
            "sample",
        ], f"Invalid vid_query_method: {vid_query_method}, must be one of middle, random, or sample"

        self.frame_loader = FrameLoader(
            transform=self.transform, method=vid_query_method, frames_video=vid_frames
        )

    def __len__(self) -> int:
        return len(self.target_txts)

    def __getitem__(self, index):
        target_txt = self.target_txts[index]
        ann = self.df.loc[target_txt]
        if ann.ndim > 1:
            ann = ann.sample()
            ann = ann.iloc[0]

        reference_pth = str(ann["path1"])
        reference_frames = self.frame_loader(reference_pth)  # 加载视频帧

        modifier = pre_caption(ann["edit"], self.max_words)

        target_pth = str(ann["path2"])
        target_frames = self.frame_loader(target_pth)  # 加载视频帧

        # description = str(ann["caption"])
        ref_webvid_caption = str(ann["txt1"])
        tar_webvid_caption = str(ann["txt2"])

        if self.mode == "classic":
            return target_pth, target_frames, tar_webvid_caption

        if self.split == "train":
            return (
                reference_frames,
                target_frames,
                modifier,
                ref_webvid_caption,
                tar_webvid_caption,
            )
        else:
            return (
                reference_pth,
                target_pth,
                modifier,
                ref_webvid_caption,
                tar_webvid_caption,
                reference_frames
            )
