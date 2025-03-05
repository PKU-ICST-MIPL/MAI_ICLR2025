# -- coding: utf-8 --
import json
from tqdm import tqdm
import random

with open("/mnt/longvideo/chenyanzhe/fashion-caption/DatasetFC/meta_all_129927.json", "r") as json_file:
    all_data = json.load(json_file)
with open("/mnt/longvideo/chenyanzhe/Multiturn/data/pairs/FC-id-info.json", "r") as json_file:
    id_info_dict = json.load(json_file)
with open("/mnt/longvideo/chenyanzhe/Multiturn/data/pairs/FC-category-id.json", "r") as json_file:
    category_id_dict = json.load(json_file)


def get_color_image(data):
    color_info = random.choice(data)
    try:
        index = random.randint(0, len(color_info) - 2)
        color = color_info["color"]
        link = color_info[str(index)]
        image_name = link.split('sr3/')[-1].split("?")[0]
    except:
        color, image_name = None, None
    return color, image_name


def change_cap(text, color):
    if text.startswith("a"):
        index = text.find(" ")
        if index != -1:
            new_text = text[:index] + " " + color + text[index:]
        else:
            new_text = color + " " + text
    else:
        new_text = color + " " + text
    return new_text


def find_similar_list(ref_id, tar1_id, tar2_id, non_zero_numbers, same_cate_id_list, id_info_dict):
    random.shuffle(same_cate_id_list)
    for id in same_cate_id_list:
        if id not in [ref_id, tar1_id, tar2_id]:
            attrid = id_info_dict.get(str(id))["attrid"]
            count = sum(1 for num in non_zero_numbers if num in attrid)
            # 如果至少有两个相同的数字，则返回该列表
            if count >= 3:
                return str(id)
    return None


def clean_title_des(ref0_title, tar1_title, tar2_title, tar3_title, ref0_des, tar1_des, tar2_des, tar3_des):
    strings = [ref0_title, tar1_title, tar2_title, tar3_title, ref0_des, tar1_des, tar2_des, tar3_des]
    cleaned_strings = []
    for s in strings:
        cleaned_string = s.lower().replace("\n", "").replace("\t", "")
        cleaned_strings.append(cleaned_string)
    return cleaned_strings[0], cleaned_strings[1], cleaned_strings[2], cleaned_strings[3], cleaned_strings[4], \
        cleaned_strings[5], cleaned_strings[6], cleaned_strings[7]


count_missing = 0
all_lines: list = []
for idx, item in tqdm(enumerate(all_data), total=len(all_data)):
    id = item["id"]
    category = item["category"]
    images = item["images"]
    title = item["title"]
    description = item["description"]
    attrid = item["attrid"]

    color, image = get_color_image(images)
    if image is None:
        continue

    ref_id, ref_title, ref_des, ref_attrid, ref_image = str(id), change_cap(
        title, color), change_cap(description, color), attrid, image

    same_cate_id_list = category_id_dict[category]
    tar1_id, tar2_id = random.sample(same_cate_id_list, 2)
    tar1_id, tar2_id = str(tar1_id), str(tar2_id)

    tar1_info, tar2_info = id_info_dict.get(tar1_id), id_info_dict.get(tar2_id)
    tar1_color, tar1_images = get_color_image(tar1_info["images"])
    if tar1_images is None:
        continue
    tar1_title, tar1_des, tar1_attrid = tar1_info["title"], tar1_info["description"], tar1_info["attrid"]
    tar1_title, tar1_des = change_cap(tar1_title, tar1_color), change_cap(tar1_des, tar1_color)

    tar2_color, tar2_images = get_color_image(tar2_info["images"])
    if tar2_images is None:
        continue
    tar2_title, tar2_des, tar2_attrid = tar2_info["title"], tar2_info["description"], tar2_info["attrid"]
    tar2_title, tar2_des = change_cap(tar2_title, tar2_color), change_cap(tar2_des, tar2_color)

    non_zero_numbers = [num for num in ref_attrid + tar1_attrid if num != 0]
    tar3_id = find_similar_list(ref_id, tar1_id, tar2_id, non_zero_numbers, same_cate_id_list, id_info_dict)
    if tar3_id is None:
        tar3_id = str(random.sample(same_cate_id_list, 1)[0])

    tar3_info = id_info_dict.get(str(tar3_id))
    tar3_color, tar3_images = get_color_image(tar3_info["images"])
    if tar3_images is None:
        continue
    tar3_title, tar3_des, tar3_attrid = tar3_info["title"], tar3_info["description"], tar3_info["attrid"]
    tar3_title, tar3_des = change_cap(tar3_title, tar3_color), change_cap(tar3_des, tar3_color)

    ref_title, tar1_title, tar2_title, tar3_title, ref_des, tar1_des, tar2_des, tar3_des = clean_title_des(
        ref_title, tar1_title, tar2_title, tar3_title, ref_des, tar1_des, tar2_des, tar3_des)

    line = f"{ref_id}/{ref_image}\t{tar1_id}/{tar1_images}\t{tar2_id}/{tar2_images}\t{tar3_id}/{tar3_images}\t{ref_title}\t{tar1_title}\t{tar2_title}\t{tar3_title}\t{ref_des}\t{tar1_des}\t{tar2_des}\t{tar3_des}\n"

    try:
        (ref0_path, tar1_path, tar2_path, tar3_path, ref_title, tar1_title, tar2_title, tar3_title
         , ref_des, tar1_des, tar2_des, tar3_des) = line.strip().split('\t')
        all_lines.append(line)
    except:
        count_missing += 1

random.shuffle(all_lines)

# TODO
file_name = '/mnt/longvideo/chenyanzhe/Multiturn/data/pairs/FC-pair-path.txt'
with open(file_name, 'w') as file:
    for line in all_lines:
        file.write(line)
# 输出成功完成的消息
print(f'文件{file_name}写入完成。')

print("行数:", len(all_lines))
