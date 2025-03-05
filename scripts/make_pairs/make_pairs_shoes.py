import os
import random
from tqdm import tqdm


def get_image_paths(root_dir):
    image_paths = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.jpg'):
                image_paths.append(os.path.join(root, file))
    return image_paths


def get_caption(img_path):
    txt_name = img_path.rsplit('/', 1)[-1].replace("jpg", "txt")
    txt_file_path = os.path.join("/mnt/longvideo/chenyanzhe/datasets/shoes_dataset/blip2_captions_combine/", txt_name)
    with open(txt_file_path, 'r') as file:
        file_contents = file.read()
    return file_contents


all_lines = []
for type in ["womens_athletic_shoes", "womens_boots", "womens_clogs", "womens_flats", "womens_high_heels",
             "womens_pumps", "womens_rain_boots", "womens_sneakers", "womens_stiletto", "womens_wedding_shoes"]:
    # TODO 设置根目录
    root_directory = '/mnt/longvideo/chenyanzhe/datasets/shoes_dataset/women/{}'.format(type)

    for first_level_dir in os.listdir(root_directory):  # 0, 1
        first_level_dir_path = os.path.join(root_directory, first_level_dir)
        if os.path.isdir(first_level_dir_path):
            for third_level_dir in tqdm(os.listdir(first_level_dir_path)):
                ref_image_path = os.path.join(first_level_dir_path, third_level_dir)
                if not ref_image_path.endswith('.jpg'):
                    continue

                images_list = get_image_paths(root_directory)
                tar1_img_path, tar2_img_path, tar3_img_path = random.sample(images_list, 3)

                all_lines.append(
                    f'{ref_image_path}\t{tar1_img_path}\t{tar2_img_path}\t{tar3_img_path}\t{get_caption(ref_image_path)}\t{get_caption(tar1_img_path)}\t{get_caption(tar2_img_path)}\t{get_caption(tar3_img_path)}\n')

random.shuffle(all_lines)
with open('/mnt/longvideo/chenyanzhe/Multiturn/data/pairs/shoes-pairs.txt', 'w') as file:
    for line in all_lines:
        file.write(line)

print('/mnt/longvideo/chenyanzhe/Multiturn/data/pairs/shoes-pairs.txt存储完毕')
