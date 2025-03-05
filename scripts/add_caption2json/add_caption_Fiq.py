import json
from tqdm import tqdm
import os

# TODO
json_file_root = "/mnt/longvideo/chenyanzhe/Multiturn/data/modifiers"
with open(os.path.join(json_file_root, "Fiq_val.json")) as f:
    triplets = json.load(f)


def get_Fiq_caption(name):
    if ".png" in name:
        name = name.split('/')[-1].split(".png")[0]
    path = os.path.join('/mnt/longvideo/chenyanzhe/Multiturn/data/fashion-iq/blip2_captions_combine', name + '.txt')
    with open(path, 'r') as file:
        file_contents = file.read()
    return file_contents


new_triplets = []
wrong_num = 0
prefix_path = "/mnt/longvideo/chenyanzhe/Multiturn/data/fashion-iq/images/"
for item in tqdm(triplets, total=len(triplets)):
    try:
        ref_cap = get_Fiq_caption(item["ref"])
        tar1_cap = get_Fiq_caption(item["tar1"])
        tar2_cap = get_Fiq_caption(item["tar2"])
        tar3_cap = get_Fiq_caption(item["tar3"])
        if ref_cap is None or tar1_cap is None or tar2_cap is None or tar3_cap is None:
            wrong_num += 1
            continue
        new_item = {
            "ref": prefix_path + item["ref"] + ".png",
            "tar1": prefix_path + item["tar1"] + ".png",
            "tar2": prefix_path + item["tar2"] + ".png",
            "tar3": prefix_path + item["tar3"] + ".png",
            "mod1": item["mod1"],
            "mod2": item["mod2"],
            "mod3": item["mod3"],
            "cap0": ref_cap,
            "cap1": tar1_cap,
            "cap2": tar2_cap,
            "cap3": tar3_cap
        }
        new_triplets.append(new_item)
    except:
        wrong_num += 1

# TODO 存储文件路径
save_file = "/mnt/longvideo/chenyanzhe/Multiturn/data/modifiers/Fiq_val_cap.json"
with open(save_file, "w") as json_file:
    json.dump(new_triplets, json_file, indent=4)
print(f"数据已保存到 {save_file}")
print(f"Wrong: {wrong_num}")
