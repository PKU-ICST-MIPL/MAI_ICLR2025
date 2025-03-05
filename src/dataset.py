import json
import os
import PIL
from torch.utils.data import DataLoader
import PIL.Image
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class TargetPad:
    def __init__(self, target_ratio: float, size: int):
        self.size = size
        self.target_ratio = target_ratio

    def __call__(self, image):
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:  # check if the ratio is above or below the target ratio
            return image
        scaled_max_wh = max(w, h) / self.target_ratio  # rescale the pad to match the target ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, "constant")


def targetpad_transform(target_ratio: float, dim: int):
    return Compose(
        [
            TargetPad(target_ratio, dim),
            Resize(dim),
            CenterCrop(dim),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


class SingleDataset(Dataset):
    def __init__(self, split, mode, preprocess, dataset_name):
        self.mode = mode
        self.split = split
        self.json_file_root = "/mnt/longvideo/chenyanzhe/Multiturn/data/modifiers"
        self.preprocess = preprocess
        self.dataset_name = dataset_name.lower()

        if split == "train":
            with open(os.path.join(self.json_file_root, f"{self.dataset_name}_train_cap.json")) as f:
                self.triplets = json.load(f)
        else:
            with open(os.path.join(self.json_file_root, f"{self.dataset_name}_test_cap.json")) as f:
                self.triplets = json.load(f)

        # get the image names
        self.image_names: list = []
        with open(os.path.join(self.json_file_root, f"{self.dataset_name}_index_names.txt")) as f:
            self.image_names = f.readlines()
            self.image_names = [line.strip() for line in self.image_names]

    def __getitem__(self, index):
        try:
            if self.mode == "relative":
                ref_path = self.triplets[index]["ref"]
                tar1_path = self.triplets[index]["tar1"]
                tar2_path = self.triplets[index]["tar2"]
                tar3_path = self.triplets[index]["tar3"]
                mod1 = self.triplets[index]["mod1"]
                mod2 = self.triplets[index]["mod2"]
                mod3 = self.triplets[index]["mod3"]
                cap0 = self.triplets[index]["cap0"]
                cap1 = self.triplets[index]["cap1"]
                cap2 = self.triplets[index]["cap2"]
                cap3 = self.triplets[index]["cap3"]
                ref_img = self.preprocess(PIL.Image.open(ref_path))
                tar1_img = self.preprocess(PIL.Image.open(tar1_path))
                tar2_img = self.preprocess(PIL.Image.open(tar2_path))
                tar3_img = self.preprocess(PIL.Image.open(tar3_path))
                if self.split == "train":
                    if "FIRST" in mod3:
                        last_ref_img = ref_img
                        last_cap = cap0
                    elif "SECOND" in mod3:
                        last_ref_img = tar1_img
                        last_cap = cap1
                    else:
                        last_ref_img = tar2_img
                        last_cap = cap2
                    return (
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
                    )

                elif self.split == "val":
                    if "FIRST" in mod3:
                        reference_name = ref_path
                        reference_cap = cap0
                    elif "SECOND" in mod3:
                        reference_name = tar1_path
                        reference_cap = cap1
                    else:
                        reference_name = tar2_path
                        reference_cap = cap2
                    target_name = tar3_path
                    target_cap = cap3
                    return (
                        reference_name,
                        target_name,
                        reference_cap,
                        target_cap,
                        ref_path,
                        tar1_path,
                        tar2_path,
                        tar3_path,
                        mod1,
                        mod2,
                        mod3,
                        cap0,
                        cap1,
                        cap2,
                        cap3,
                    )

            elif self.mode == "classic":
                image_path = self.image_names[index]
                image_name = image_path
                image = self.preprocess(PIL.Image.open(image_path))
                if self.dataset_name == "200k":
                    name = image_path.rsplit("/", 1)[-1].split("_")[0]
                    file_path = os.path.join(
                        "/mnt/longvideo/chenyanzhe/fashion-caption/image_captions_200k",
                        name + ".txt",
                    )
                    with open(file_path, "r") as file:
                        caption = file.read()
                else:  # shoes
                    txt_name = image_path.rsplit("/", 1)[-1].replace("jpg", "txt")
                    txt_file_path = os.path.join(
                        "/mnt/longvideo/chenyanzhe/datasets/shoes_dataset/blip2_captions_combine/",
                        txt_name,
                    )
                    with open(txt_file_path, "r") as file:
                        caption = file.read()
                return image_name, image, caption

            else:
                raise ValueError("mode should be in ['relative', 'classic']")
        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == "relative":
            return len(self.triplets)
        elif self.mode == "classic":
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


class ComposeDataset(Dataset):
    def __init__(self, split: str, mode: str, preprocess, dataset_name):
        self.mode = mode
        self.split = split
        self.json_file_root = "/mnt/longvideo/chenyanzhe/Multiturn/data/modifiers"
        self.preprocess = preprocess
        self.dataset_name = dataset_name.lower()

        if self.dataset_name == "fiq":
            if split == "train":
                with open(os.path.join(self.json_file_root, "Fiq_train_cap.json")) as f:
                    self.triplets = json.load(f)
            else:
                with open(os.path.join(self.json_file_root, "Fiq_val_cap.json")) as f:
                    self.triplets = json.load(f)
            self.image_names: list = []
            with open(os.path.join(self.json_file_root, "Fiq_index_names.txt")) as f:
                self.image_names = f.readlines()
                self.image_names = [line.strip() for line in self.image_names]
                # 注意 Fiq 的 index names 都是序号，需要加上前后缀
                self.image_names = [
                    "/mnt/longvideo/chenyanzhe/Multiturn/data/fashion-iq/images/" + name + ".png"
                    for name in self.image_names
                ]
        elif self.dataset_name == "fc":
            if split == "train":
                with open(os.path.join(self.json_file_root, "FC_train_cap.json")) as f:
                    self.triplets = json.load(f)
            else:
                with open(os.path.join(self.json_file_root, "FC_test_cap.json")) as f:
                    self.triplets = json.load(f)
            self.image_names = []
            with open(os.path.join(self.json_file_root, "FC_index_names.txt")) as f:
                self.image_names = f.readlines()
                self.image_names = [line.strip() for line in self.image_names]
            with open(os.path.join(self.json_file_root, "FC-caption.json")) as f:
                self.name_cap_dict = json.load(f)

        else:
            raise ValueError("Dataset Name ERROR!")

    def __getitem__(self, index):
        try:
            if self.mode == "relative":
                ref_path = self.triplets[index]["ref"]
                tar1_path = self.triplets[index]["tar1"]
                tar2_path = self.triplets[index]["tar2"]
                tar3_path = self.triplets[index]["tar3"]
                mod1 = self.triplets[index]["mod1"]
                mod2 = self.triplets[index]["mod2"]
                mod3 = self.triplets[index]["mod3"]
                cap0 = self.triplets[index]["cap0"]
                cap1 = self.triplets[index]["cap1"]
                cap2 = self.triplets[index]["cap2"]
                cap3 = self.triplets[index]["cap3"]
                ref_img = self.preprocess(PIL.Image.open(ref_path))
                tar1_img = self.preprocess(PIL.Image.open(tar1_path))
                tar2_img = self.preprocess(PIL.Image.open(tar2_path))
                tar3_img = self.preprocess(PIL.Image.open(tar3_path))

                if self.split == "train":
                    return (
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
                    )

                elif self.split == "val":
                    ref_name, tar1_name, tar2_name, tar3_name = (
                        ref_path,
                        tar1_path,
                        tar2_path,
                        tar3_path,
                    )
                    return (
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
                    )

            elif self.mode == "classic":
                if self.dataset_name.lower() == "fiq":
                    image_name = self.image_names[index]
                    image = self.preprocess(PIL.Image.open(image_name))
                    if ".png" in image_name:
                        name_id = image_name.split("/")[-1].split(".png")[0]
                    else:
                        name_id = image_name
                    path = os.path.join(
                        "/mnt/longvideo/chenyanzhe/Multiturn/data/fashion-iq/blip2_captions_combine",
                        name_id + ".txt",
                    )
                    with open(path, "r") as file:
                        caption = file.read()
                    return image_name, image, caption

                elif self.dataset_name.lower() == "fc":
                    image_path = self.image_names[index]
                    image_name = image_path
                    image = self.preprocess(PIL.Image.open(image_path))
                    name_id = image_name[image_name.rfind("/", 0, image_name.rfind("/")) + 1 : image_name.rfind("/")]
                    caption = self.name_cap_dict[name_id]
                    return image_name, image, caption

                else:
                    raise ValueError("Dataset Name ERROR!")

            else:
                raise ValueError("mode should be in ['relative', 'classic']")
        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == "relative":
            return len(self.triplets)
        elif self.mode == "classic":
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


class STDataset(Dataset):
    def __init__(self, split: str, mode: str, preprocess, dataset_name):
        self.mode = mode
        self.split = split
        self.json_file_root = "/mnt/longvideo/chenyanzhe/Multiturn/data/modifiers"
        self.preprocess = preprocess
        self.dataset_name = dataset_name.lower()

        if self.dataset_name == "200k-st":
            if split == "train":
                with open(os.path.join(self.json_file_root, "200k_train_cap.json")) as f:
                    self.triplets = json.load(f)
            else:
                with open(os.path.join(self.json_file_root, "200k_test_cap.json")) as f:
                    self.triplets = json.load(f)
            self.image_names: list = []
            with open(os.path.join(self.json_file_root, "200k_index_names.txt")) as f:
                self.image_names = f.readlines()
                self.image_names = [line.strip() for line in self.image_names]
        elif self.dataset_name == "fc-st":
            if split == "train":
                with open(os.path.join(self.json_file_root, "FC_train_cap.json")) as f:
                    self.triplets = json.load(f)
            else:
                with open(os.path.join(self.json_file_root, "FC_test_cap.json")) as f:
                    self.triplets = json.load(f)
            self.image_names = []
            with open(os.path.join(self.json_file_root, "FC_index_names.txt")) as f:
                self.image_names = f.readlines()
                self.image_names = [line.strip() for line in self.image_names]
            with open(os.path.join(self.json_file_root, "FC-caption.json")) as f:
                self.name_cap_dict = json.load(f)

        else:
            raise ValueError("Dataset Name ERROR!")

    def __getitem__(self, index):
        try:
            if self.mode == "relative":
                ref_path = self.triplets[index]["ref"]
                tar1_path = self.triplets[index]["tar1"]
                mod1 = self.triplets[index]["mod1"]
                cap0 = self.triplets[index]["cap0"]
                cap1 = self.triplets[index]["cap1"]
                ref_img = self.preprocess(PIL.Image.open(ref_path))
                tar1_img = self.preprocess(PIL.Image.open(tar1_path))

                if self.split == "train":
                    return ref_img, tar1_img, mod1, cap0, cap1

                elif self.split == "val":
                    ref_name, tar1_name = ref_path, tar1_path
                    return ref_name, tar1_name, mod1, cap0, cap1

            elif self.mode == "classic":
                image_path = self.image_names[index]
                image_name = image_path
                image = self.preprocess(PIL.Image.open(image_path))
                if self.dataset_name.lower() == "200k-st":
                    name = image_path.rsplit("/", 1)[-1].split("_")[0]
                    file_path = os.path.join(
                        "/mnt/longvideo/chenyanzhe/fashion-caption/image_captions_200k",
                        name + ".txt",
                    )
                    with open(file_path, "r") as file:
                        caption = file.read()
                    return image_name, image, caption
                elif self.dataset_name.lower() == "fc-st":
                    name_id = image_name[image_name.rfind("/", 0, image_name.rfind("/")) + 1 : image_name.rfind("/")]
                    caption = self.name_cap_dict[name_id]
                    return image_name, image, caption
                else:
                    raise ValueError("Dataset Name ERROR!")
            else:
                raise ValueError("mode should be in ['relative', 'classic']")
        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == "relative":
            return len(self.triplets)
        elif self.mode == "classic":
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


def get_fiq_caption(name):
    if ".png" in name:
        name = name.split("/")[-1].split(".png")[0]
    path = os.path.join(
        "/mnt/longvideo/chenyanzhe/Multiturn/data/fashion-iq/blip2_captions_combine",
        name + ".txt",
    )
    with open(path, "r") as file:
        caption = file.read()
    return caption


class FashionIQDataset(Dataset):
    def __init__(self, split, dress_types, mode, preprocess: callable):
        self.mode = mode
        self.dress_types = dress_types
        self.split = split

        if mode not in ["relative", "classic"]:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ["test", "train", "val"]:
            raise ValueError("split should be in ['test', 'train', 'val']")
        for dress_type in dress_types:
            if dress_type not in ["dress", "shirt", "toptee"]:
                raise ValueError("dress_type should be in ['dress', 'shirt', 'toptee']")

        self.preprocess = preprocess
        self.base_path = "/mnt/longvideo/chenyanzhe/Multiturn"

        self.triplets = []
        for dress_type in dress_types:
            with open(f"{self.base_path}/fashionIQ_dataset/captions/cap.{dress_type}.{split}.json") as f:
                self.triplets.extend(json.load(f))

        self.image_names: list = []
        for dress_type in dress_types:
            with open(f"{self.base_path}/fashionIQ_dataset/image_splits/split.{dress_type}.{split}.json") as f:
                self.image_names.extend(json.load(f))

    def __getitem__(self, index):
        try:
            if self.mode == "relative":
                modifier = self.triplets[index]["captions"]
                reference_name = self.triplets[index]["candidate"]
                target_name = self.triplets[index]["target"]
                reference_caption = get_fiq_caption(reference_name)
                target_caption = get_fiq_caption(target_name)

                if self.split == "train":
                    reference_image_path = f"{self.base_path}/fashionIQ_dataset/images/{reference_name}.png"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    target_image_path = f"{self.base_path}/fashionIQ_dataset/images/{target_name}.png"
                    target_image = self.preprocess(PIL.Image.open(target_image_path))
                    return (
                        reference_image,
                        target_image,
                        modifier,
                        reference_caption,
                        target_caption,
                    )

                elif self.split == "val":
                    return (
                        reference_name,
                        target_name,
                        modifier,
                        reference_caption,
                        target_caption,
                    )

            elif self.mode == "classic":
                image_name = self.image_names[index]
                image_path = f"{self.base_path}/fashionIQ_dataset/images/{image_name}.png"
                image = self.preprocess(PIL.Image.open(image_path))
                image_caption = get_fiq_caption(image_name)
                return image_name, image, image_caption

            else:
                raise ValueError("mode should be in ['relative', 'classic']")
        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == "relative":
            return len(self.triplets)
        elif self.mode == "classic":
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


class ExistingFIQ(Dataset):
    def __init__(self, split, dress_types, mode, preprocess, turn_number: int = 0):
        self.mode = mode
        self.dress_types = dress_types
        self.split = split

        if mode not in ["relative", "classic"]:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ["test", "train", "val"]:
            raise ValueError("split should be in ['test', 'train', 'val']")
        for type in dress_types:
            if type not in ["dress", "shirt", "toptee", "all"]:
                raise ValueError("type should be in ['dress', 'shirt', 'toptee', 'all']")

        self.preprocess = preprocess
        self.prefix_path = "/mnt/longvideo/chenyanzhe/Multiturn/data/fashion-iq"

        if not (mode == "classic" and dress_types == ["all"]):
            self.triplets = []
            for type in dress_types:
                with open("{}/json_files/{}.{}_{}.json".format(self.prefix_path, type, split, turn_number)) as f:
                    self.triplets.extend(json.load(f))

        self.image_names: list = []
        for type in dress_types:
            with open("{}/json_files/split.{}.{}.json".format(self.prefix_path, type, split)) as f:
                self.image_names.extend(json.load(f))

        print(f"FashionIQ {split} - type:{dress_types}  mode:{mode} turn_number:{turn_number} initialized")

    def __getitem__(self, index):
        try:
            if self.mode == "relative":
                target_name = self.triplets[index]["target"][1]
                target_caption = get_fiq_caption(target_name)
                reference_names_list = [_[-1] for _ in self.triplets[index]["reference"]]
                reference_captions_list = [get_fiq_caption(name) for name in reference_names_list]
                modifiers_list = [_[1] for _ in self.triplets[index]["reference"]]
                modifiers_list = [" and ".join(_).lower() for _ in modifiers_list]

                if self.split == "train":
                    reference_images_list = []
                    for i in range(len(reference_names_list)):
                        reference_name = reference_names_list[i]
                        reference_path = "{}/images/{}.png".format(self.prefix_path, reference_name)
                        reference_image = self.preprocess(PIL.Image.open(reference_path))
                        reference_images_list.append(reference_image)
                    target_path = "{}/images/{}.png".format(self.prefix_path, target_name)
                    target_image = self.preprocess(PIL.Image.open(target_path))
                    return (
                        reference_images_list,
                        target_image,
                        modifiers_list,
                        reference_captions_list,
                        target_caption,
                    )
                else:  # 'val'
                    return (
                        reference_names_list,
                        target_name,
                        modifiers_list,
                        reference_captions_list,
                    )

            elif self.mode == "classic":
                image_name = self.image_names[index]
                image_path = "{}/images/{}.png".format(self.prefix_path, image_name)
                image = self.preprocess(PIL.Image.open(image_path))
                image_caption = get_fiq_caption(image_name)
                return image_name, image, image_caption

            else:
                raise ValueError("mode should be in ['relative', 'classic']")

        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == "relative":
            return len(self.triplets)
        elif self.mode == "classic":
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


if __name__ == "__main__":
    pre = targetpad_transform(1.25, 224)
    relative_train_dataset = ComposeDataset("train", "relative", pre, "fc")
    relative_train_loader = DataLoader(
        dataset=relative_train_dataset,
        batch_size=32,
        drop_last=True,
        num_workers=8,
        pin_memory=False,
        collate_fn=collate_fn,
        shuffle=True,
    )
    for idx, (ref_img, tar1_img, tar2_img, tar3_img, mod1, mod2, mod3) in enumerate(relative_train_loader):
        print(ref_img)
        break
