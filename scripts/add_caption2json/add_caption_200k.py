import json
from tqdm import tqdm
import os
import random

# TODO
json_file_root = "/mnt/longvideo/chenyanzhe/Multiturn/data/modifiers"
with open(os.path.join(json_file_root, "200k_test.json")) as f:
    triplets = json.load(f)


def get_200k_caption(image_path):
    name = image_path.rsplit('/', 1)[-1].split('_')[0]
    file_path = os.path.join('/mnt/longvideo/chenyanzhe/fashion-caption/image_captions_200k', name + '.txt')
    with open(file_path, 'r') as file:
        file_contents = file.read()
    return file_contents


random.shuffle(triplets)
triplets = triplets[:int(len(triplets) / 3)]
new_triplets = []
wrong_num = 0
for item in tqdm(triplets, total=len(triplets)):
    try:
        ref_cap = get_200k_caption(item["ref"])
        tar1_cap = get_200k_caption(item["tar1"])
        tar2_cap = get_200k_caption(item["tar2"])
        tar3_cap = get_200k_caption(item["tar3"])
        if ref_cap is None or tar1_cap is None or tar2_cap is None or tar3_cap is None:
            wrong_num += 1
            continue
        new_item = {
            "ref": item["ref"],
            "tar1": item["tar1"],
            "tar2": item["tar2"],
            "tar3": item["tar3"],
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
save_file = "/mnt/longvideo/chenyanzhe/Multiturn/data/modifiers/200k_test_cap.json"
with open(save_file, "w") as json_file:
    json.dump(new_triplets, json_file, indent=4)
print(f"数据已保存到 {save_file}")
print(f"Wrong: {wrong_num}")
