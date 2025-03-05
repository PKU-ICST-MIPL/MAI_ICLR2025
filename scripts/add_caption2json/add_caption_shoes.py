import json
from tqdm import tqdm
import os
import random

# TODO
json_file_root = "/mnt/longvideo/chenyanzhe/Multiturn/data/modifiers"
file_name = "shoes_test.json"
with open(os.path.join(json_file_root, file_name)) as f:
    triplets = json.load(f)


def get_shoes_caption(image_path):
    txt_name = image_path.rsplit('/', 1)[-1].replace("jpg", "txt")
    txt_file_path = os.path.join(
        "/mnt/longvideo/chenyanzhe/datasets/shoes_dataset/blip2_captions_combine/", txt_name)
    with open(txt_file_path, 'r') as file:
        caption = file.read()
    return caption


random.shuffle(triplets)
new_triplets = []
wrong_num = 0
for item in tqdm(triplets, total=len(triplets)):
    try:
        ref_cap = get_shoes_caption(item["ref"])
        tar1_cap = get_shoes_caption(item["tar1"])
        tar2_cap = get_shoes_caption(item["tar2"])
        tar3_cap = get_shoes_caption(item["tar3"])
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

new_file_prefix = file_name.replace(".json", "")
save_file = f"/mnt/longvideo/chenyanzhe/Multiturn/data/modifiers/{new_file_prefix}_cap.json"
with open(save_file, "w") as json_file:
    json.dump(new_triplets, json_file, indent=4)
print(f"数据已保存到 {save_file}")
print(f"Wrong: {wrong_num}")
