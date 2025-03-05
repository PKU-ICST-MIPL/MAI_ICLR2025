import json
from tqdm import tqdm
import os

json_file_root = "/mnt/longvideo/chenyanzhe/Multiturn/data/modifiers"
with open(os.path.join(json_file_root, "FC_test.json")) as f:
    triplets = json.load(f)

# TODO
with open(os.path.join("/mnt/longvideo/chenyanzhe/Multiturn/data/pairs", "FC-caption.json")) as f:
    name_cap_dict = json.load(f)


def get_num(path_string):
    return path_string[path_string.rfind('/', 0, path_string.rfind('/')) + 1: path_string.rfind('/')]


new_triplets = []
wrong_num = 0
for item in tqdm(triplets, total=len(triplets)):
    try:
        ref_cap = name_cap_dict[get_num(item["ref"])]
        tar1_cap = name_cap_dict[get_num(item["tar1"])]
        tar2_cap = name_cap_dict[get_num(item["tar2"])]
        tar3_cap = name_cap_dict[get_num(item["tar3"])]
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
save_file = "/mnt/longvideo/chenyanzhe/Multiturn/data/modifiers/FC_test_cap.json"
with open(save_file, "w") as json_file:
    json.dump(new_triplets, json_file, indent=4)
print(f"数据已保存到 {save_file}")
print(f"Wrong: {wrong_num}")
