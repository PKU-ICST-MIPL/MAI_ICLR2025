import json
import os
from tqdm import tqdm

# 读取文本文件，逐行解析每一行的内容
with open('/mnt/longvideo/chenyanzhe/Multiturn/data/pairs/FC-pair-path.txt', 'r') as file:
    lines = file.readlines()

# 创建一个空字典来存储path和title的对应关系
path_to_title = {}
prefix = "/mnt/longvideo/chenyanzhe/Multiturn/data/DatasetFC/fashion_images/"
# 逐行处理文本内容
for line in tqdm(lines):
    ref_path, tar1_path, tar2_path, tar3_path, ref_title, tar1_title, tar2_title, tar3_title, ref0_des, tar1_des, tar2_des, tar3_des = line.strip().split(
        '\t')
    ref_path = ref_path.split("/")[0]
    tar1_path = tar1_path.split("/")[0]
    tar2_path = tar2_path.split("/")[0]
    tar3_path = tar3_path.split("/")[0]

    # 将path和title添加到字典中
    path_to_title[ref_path] = ref_title
    path_to_title[tar1_path] = tar1_title
    path_to_title[tar2_path] = tar2_title
    path_to_title[tar3_path] = tar3_title

# 将字典存储为JSON文件
with open('/mnt/longvideo/chenyanzhe/Multiturn/data/pairs/FC-caption.json', 'w', encoding='utf-8') as json_file:
    json.dump(path_to_title, json_file, indent=4)
