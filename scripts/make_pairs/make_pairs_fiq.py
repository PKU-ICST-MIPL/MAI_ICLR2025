import json
from tqdm import tqdm
import random
import os


# 找有一致attribute的target3
def find_match(ref_name, tar1_name, tar2_name, name_attr_dict, names):
    names_list = names.copy()
    random.shuffle(names_list)
    ref_attr, tar1_attr, tar2_attr = name_attr_dict.get(ref_name), name_attr_dict.get(tar1_name), name_attr_dict.get(
        tar2_name)
    for name in names_list:
        if name not in [ref_name, tar1_name, tar2_name]:
            if not name in name_attr_dict:
                continue
            attr = name_attr_dict[name]
            if attr is None or ref_attr is None or tar1_attr is None:
                continue
            if set(attr) & set(ref_attr) and set(attr) & set(tar1_attr):
                return name
    return None


def get_intersect(list1, list2):
    intersection = set(list1) & set(list2)
    if intersection:
        return ' and '.join(intersection)
    else:
        return None


def get_caption(name):
    file_path = os.path.join('/mnt/longvideo/chenyanzhe/Multiturn/data/fashion-iq/blip2_captions_combine',
                             name + '.txt')
    with open(file_path, 'r') as file:
        file_contents = file.read()
    return file_contents


if __name__ == "__main__":
    # TODO
    SPLIT = "val"
    product_types = ["dress", "toptee", "shirt"]
    attr_file_path = f"/mnt/longvideo/chenyanzhe/Multiturn/data/fashion-iq/attributes/merged.json"
    with open(attr_file_path, "r") as attr_f:
        name_attr_dict = json.load(attr_f)

    all_lines: list = []
    for product_type in product_types:
        split_file_path = f"/mnt/longvideo/chenyanzhe/Multiturn/data/fashion-iq/image_splits/split.{product_type}.{SPLIT}.json"
        with open(split_file_path, "r") as split_f:
            file_names: list = json.load(split_f)

        for idx, ref_name in tqdm(enumerate(file_names), total=len(file_names)):
            for _ in range(5):  # 每一行对应制造10批数据
                tar1_name, tar2_name = random.sample(file_names, 2)

                if not (tar1_name in name_attr_dict and tar2_name in name_attr_dict):
                    continue

                ref_attr, tar1_attr = name_attr_dict.get(ref_name), name_attr_dict.get(tar1_name)
                if ref_attr is None or tar1_attr is None:
                    continue

                tar3_name = find_match(ref_name, tar1_name, tar2_name, name_attr_dict, file_names)
                if tar3_name is None:
                    continue
                tar2_attr, tar3_attr = name_attr_dict.get(tar2_name), name_attr_dict.get(tar3_name)
                simi1, simi2, simi3 = get_intersect(ref_attr, tar3_attr), get_intersect(
                    tar1_attr, tar3_attr), get_intersect(tar2_attr, tar3_attr)

                ref_cap, tar1_cap, tar2_cap, tar3_cap = get_caption(ref_name), get_caption(tar1_name), get_caption(
                    tar2_name), get_caption(tar3_name)

                new_line = f"{ref_name}\t{tar1_name}\t{tar2_name}\t{tar3_name}\t{ref_cap}\t{tar1_cap}\t{tar2_cap}\t{tar3_cap}\t{simi1}\t{simi2}\t{simi3}\n"
                all_lines.append(new_line)

    random.shuffle(all_lines)
    # TODO
    file_name = f'/mnt/longvideo/chenyanzhe/Multiturn/data/pairs/Fiq-pairs-{SPLIT}.txt'
    with open(file_name, 'w') as file:
        for line in all_lines:
            file.write(line)
    # 输出成功完成的消息
    print(f'文件{file_name}写入完成。')
    print("行数:", len(all_lines))
