import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import argparse
import re
import json


def get_caption(name_caption_dict, image_path):
    name = image_path.rsplit('/', 1)[-1].split('_')[0]
    file_path = os.path.join('/mnt/longvideo/chenyanzhe/fashion-caption/image_captions_200k', name + '.txt')
    with open(file_path, 'r') as file:
        file_contents = file.read()
    return file_contents


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=str, help="GPU card id")
    args = parser.parse_args()
    COUNT = int(args.count)
    print("COUNT id:", COUNT)
    TOTAL_NUM = 11  # TODO 注意多少个进程

    # TODO pair文件路径
    with open('/mnt/longvideo/chenyanzhe/Multiturn/data/pairs/200k-pair-path-shuffle.txt', 'r') as file:
        lines = file.readlines()
    json_file = '/mnt/longvideo/chenyanzhe/Multiturn/data/pairs/200k-name-caption.json'
    with open(json_file, 'r') as f:
        name_caption_dict = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = "/mnt/longvideo/chenyanzhe/fashion-caption/Xwin"
    model = AutoModelForCausalLM.from_pretrained("Xwin-LM/Xwin-LM-13B-V0.2", torch_dtype=torch.float16,
                                                 cache_dir=cache_dir)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("Xwin-LM/Xwin-LM-13B-V0.2", cache_dir=cache_dir)

    new_triplets = []
    # TODO 划分部分
    lines = lines[int(len(lines) / 3):int(len(lines) / 2)]

    for idx, line in tqdm(enumerate(lines), total=len(lines)):
        if not (idx % TOTAL_NUM == (COUNT - 1)):
            continue
        ref0_path, tar1_path, tar2_path, tar3_path = line.strip().split('\t')

        paths = [ref0_path, tar1_path, tar2_path, tar3_path]
        modifiers = []
        for i in range(len(paths) - 2):  # 注意最后一轮
            ref_path, tar_path = paths[i], paths[i + 1]
            reference_caption, target_caption = get_caption(name_caption_dict, ref_path), get_caption(
                name_caption_dict, tar_path)
            prompt = '''The reference image depicts "{}", the target image depicts "{}". Describe the distinctiveness of the target image briefly, with three numbered short phrases. Considering aspects include: product type, color, size, pattern, style, etc. Use the comparative form when appropriate. Do not mention the reference image. Keep answers as simple as possible. Answer:'''.format(
                reference_caption, target_caption)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            samples = model.generate(**inputs, max_new_tokens=128, temperature=0.7)
            output = tokenizer.decode(samples[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            modifier = re.sub(r'\n', ' ', output)
            if modifier != "":
                clean_prefix = ["Please", "To achieve", "To modi", "Additionally", "To trans", ]
                for prefix in clean_prefix:
                    if prefix in modifier:
                        please_index = modifier.index(prefix)
                        modifier = modifier[:please_index]  # 删除 prefix 后的内容
                pattern = r'\d+\.\s*([^0-9]+)'
                matches = re.findall(pattern, modifier)
                cleaned_mod_list = []
                if len(matches) >= 2:
                    for sentence in matches:
                        sentence = sentence.replace("The target image ", "")
                        sentence = sentence.replace("shows", "is")
                        sentence = sentence.replace("features", "is")
                        cleaned_mod_list.append(sentence.split(",")[0])
                final_modifier = ""
                for idx, string in enumerate(cleaned_mod_list, start=1):
                    final_modifier += f"{idx}. {string} "
            else:
                if ', ' in target_caption:
                    final_modifier = "1. is a " + target_caption.split(', ')[0].lower()
                else:
                    final_modifier = "1. is a " + target_caption.lower()
            modifiers.append(final_modifier)

        # 最后一组
        random_idx = random.choice([0, 1])
        ref_path, tar_path = paths[random_idx], paths[-1]
        reference_caption, target_caption = get_caption(name_caption_dict, ref_path), get_caption(
            name_caption_dict, tar_path)
        prompt = '''The reference image depicts "{}", the target image depicts "{}". Describe the distinctiveness of the target image briefly, with three numbered short phrases. Considering aspects include: product type, color, size, pattern, style, etc. Use the comparative form when appropriate. Do not mention the reference image. Keep answers as simple as possible. Answer:'''.format(
            reference_caption, target_caption)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        samples = model.generate(**inputs, max_new_tokens=128, temperature=0.7)
        output = tokenizer.decode(samples[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        modifier = re.sub(r'\n', ' ', output)
        if modifier != "":
            clean_prefix = ["Please", "To achieve", "To modi", "Additionally", "To trans", ]
            for prefix in clean_prefix:
                if prefix in modifier:
                    please_index = modifier.index(prefix)
                    modifier = modifier[:please_index]  # 删除 prefix 后的内容
            pattern = r'\d+\.\s*([^0-9]+)'
            matches = re.findall(pattern, modifier)
            cleaned_mod_list = []
            if len(matches) >= 2:
                for sentence in matches:
                    sentence = sentence.replace("The target image ", "")
                    sentence = sentence.replace("shows", "is")
                    sentence = sentence.replace("features", "is")
                    cleaned_mod_list.append(sentence.split(",")[0])
            final_turn_modifier = ""
            for idx, string in enumerate(cleaned_mod_list, start=1):
                final_turn_modifier += f"{idx}. {string} "
        else:
            if ', ' in target_caption:
                final_turn_modifier = "1. is a " + target_caption.split(', ')[0].lower()
            else:
                final_turn_modifier = "1. is a " + target_caption.lower()

        choose_str = ["FIRST", "SECOND"]
        prefix_sentences = [
            "Compared to this one I prefer the {}, and ".format(choose_str[random_idx]),
            "I would rather choose the {}, and ".format(choose_str[random_idx]),
            "I prefer the {}, and ".format(choose_str[random_idx])
        ]
        final_turn_modifier = random.choice(prefix_sentences) + final_turn_modifier

        new_item = {
            "ref": ref0_path,
            "tar1": tar1_path,
            "tar2": tar2_path,
            "tar3": tar3_path,
            "mod1": modifiers[0],
            "mod2": modifiers[1],
            "mod3": final_turn_modifier
        }
        print(new_item)
        new_triplets.append(new_item)

    # TODO 存储文件路径
    save_file = "/mnt/longvideo/chenyanzhe/Multiturn/data/modifiers/200k_part3_{}.json".format(COUNT)
    with open(save_file, "w") as json_file:
        json.dump(new_triplets, json_file, indent=4)
    print(f"数据已保存到 {save_file}")
