import os
import random
from tqdm import tqdm

ref_path, tar_path1, tar_path2, tar_path3 = [], [], [], []
for type in ["dresses", "jackets", "pants", "skirts", "tops"]:
    # 设置根目录
    root_directory = '/mnt/longvideo/chenyanzhe/fashion-caption/women/{}'.format(type)

    # 遍历所有一级目录
    for first_level_dir in os.listdir(root_directory):
        first_level_dir_path = os.path.join(root_directory, first_level_dir)
        if os.path.isdir(first_level_dir_path):
            # 遍历一级目录下的所有三级目录
            for third_level_dir in tqdm(os.listdir(first_level_dir_path)):
                third_level_dir_path = os.path.join(first_level_dir_path, third_level_dir)
                second_path = third_level_dir_path.rsplit('/', 1)[0]
                # 检查是否为目录
                if os.path.isdir(third_level_dir_path):
                    # 查找三级目录下所有 .jpeg 文件
                    jpeg_files = [f for f in os.listdir(third_level_dir_path) if f.lower().endswith('.jpeg')]
                    entries = os.listdir(second_path)
                    possible_folders = [entry for entry in entries if os.path.isdir(os.path.join(second_path, entry))]
                    for jpeg_file in jpeg_files:
                        selected_file_path = os.path.join(third_level_dir_path, jpeg_file)

                        target_folder = os.path.join(second_path, random.choice(possible_folders))
                        target_images = os.listdir(target_folder)
                        target_image = random.choice(target_images)
                        target_image_path = os.path.join(target_folder, target_image)

                        target_folder2 = os.path.join(second_path, random.choice(possible_folders))
                        target_images2 = os.listdir(target_folder2)
                        target_image2 = random.choice(target_images2)
                        target_image_path2 = os.path.join(target_folder2, target_image2)

                        target_folder3 = os.path.join(second_path, random.choice(possible_folders))
                        target_images3 = os.listdir(target_folder3)
                        target_image3 = random.choice(target_images3)
                        target_image_path3 = os.path.join(target_folder3, target_image3)

                        # print(f"随机选择的文件: {selected_file_path}")
                        # print(f"匹配的文件: {target_image_path}")
                        ref_path.append(selected_file_path)
                        tar_path1.append(target_image_path)
                        tar_path2.append(target_image_path2)
                        tar_path3.append(target_image_path3)

with open('/mnt/longvideo/chenyanzhe/Multiturn/data/pairs/200k-pair-path.txt', 'w') as file:
    for item1, item2, item3, item4 in zip(ref_path, tar_path1, tar_path2, tar_path3):
        line = f'{item1}\t{item2}\t{item3}\t{item4}\n'  # 添加换行符以分隔每一行
        file.write(line)

# 输出成功完成的消息
print('文件写入完成。')
