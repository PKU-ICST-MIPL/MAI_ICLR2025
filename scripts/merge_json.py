import os
import json

# TODO 指定路径和文件前缀
directory = "/mnt/longvideo/chenyanzhe/Multiturn/data/modifiers"
prefix = "Fiq_val"

merged_data = []
# 遍历指定路径下的所有文件
for filename in os.listdir(directory):
    if filename.startswith(prefix) and filename.endswith(".json"):
        print(filename)
        with open(os.path.join(directory, filename), 'r') as file:
            data = json.load(file)
            merged_data.extend(data)

# 将合并后的数据写入新的 JSON 文件
save_file_root = "/mnt/longvideo/chenyanzhe/Multiturn/data/modifiers"
save_file_path = os.path.join(save_file_root, f"{prefix}.json")
with open(save_file_path, 'w') as outfile:
    json.dump(merged_data, outfile, indent=4)
print(f"{save_file_path}存储完毕")
