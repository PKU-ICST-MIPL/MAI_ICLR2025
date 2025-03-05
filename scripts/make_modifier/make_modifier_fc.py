import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import re
import json


def get_same_attr(list_A, list_B):
    same_list = [word for word in list_B if word in list_A and word != "nah"]
    if same_list:
        return " and ".join([word for word in list_B if word in list_A and word != "nah"])
    else:
        return None


def get_composed_parts(id_attr_dict, ref0_path, tar1_path, tar3_path):
    ref0_attr = id_attr_dict.get(str(ref0_path.split('/')[0]))
    tar1_attr = id_attr_dict.get(str(tar1_path.split('/')[0]))
    tar3_attr = id_attr_dict.get(str(tar3_path.split('/')[0]))
    keep_first = get_same_attr(ref0_attr, tar3_attr)
    keep_second = get_same_attr(tar1_attr, tar3_attr)
    random_word = random.choice(["design", "feature", "design"])
    if keep_first is None:
        return "4. keep the SECOND's {} {}".format(keep_second, random_word)
    else:
        if keep_second is None:
            return "4. keep the FIRST's {} {}".format(keep_first, random_word)
        else:
            return "4. keep the FIRST's {} {} and keep the SECOND's {} {}".format(
                keep_first, random_word, keep_second, random_word)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=str, help="GPU card id")
    args = parser.parse_args()
    COUNT = int(args.count)
    print("COUNT id:", COUNT)

    TOTAL_NUM = 11  # TODO 注意一共多少个并行进程

    # TODO pair文件路径
    with open('/mnt/longvideo/chenyanzhe/Multiturn/data/pairs/FC-pair-path.txt', 'r') as file:
        lines = file.readlines()
    with open("/mnt/longvideo/chenyanzhe/Multiturn/data/pairs/FC-id-attr.json", "r") as json_file:
        id_attr_dict = json.load(json_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = "/mnt/longvideo/chenyanzhe/fashion-caption/Xwin"
    model = AutoModelForCausalLM.from_pretrained("Xwin-LM/Xwin-LM-13B-V0.2", torch_dtype=torch.float16,
                                                 cache_dir=cache_dir)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("Xwin-LM/Xwin-LM-13B-V0.2", cache_dir=cache_dir)

    new_triplets = []
    # TODO 注意划分部分
    # lines = lines[:int(len(lines) / 5)]
    lines = lines[int(len(lines) / 5):]

    for idx, line in tqdm(enumerate(lines), total=len(lines)):
        if not (idx % TOTAL_NUM == (COUNT - 1)):
            continue
        ref0_path, tar1_path, tar2_path, tar3_path, ref0_title, tar1_title, tar2_title, tar3_title, ref0_des, tar1_des, tar2_des, tar3_des = line.strip().split(
            '\t')

        paths = [ref0_path, tar1_path, tar2_path, tar3_path]
        titles = [ref0_title, tar1_title, tar2_title, tar3_title]
        modifiers = []
        # 最后一个保留空
        for i in range(len(paths) - 2):
            ref_path, tar_path = paths[i], paths[i + 1]
            reference_caption, target_caption = titles[i], titles[i + 1]
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
                final_modifier = "1. is a " + target_caption.lower()
            modifiers.append(final_modifier)

        prompt_composed = '''The reference image depicts "{}" and "{}", the target image depicts "{}". Describe the distinctiveness of the target image briefly, with three numbered short phrases. Considering aspects include: product type, color, size, pattern, style, etc. Use the comparative form when appropriate. Do not mention the reference image. Keep answers as simple as possible. Answer:'''.format(
            ref0_title, tar1_title, tar3_title)
        inputs = tokenizer(prompt_composed, return_tensors="pt").to(device)
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
            differ_modifier = ""
            for idx, string in enumerate(cleaned_mod_list, start=1):
                differ_modifier += f"{idx}. {string} "
        else:
            differ_modifier = "1. is a " + tar3_title.lower()
        simi_modifier = get_composed_parts(id_attr_dict, ref0_path, tar1_path, tar3_path)
        final_composed_modifier = " ".join([differ_modifier, simi_modifier])

        new_item = {
            "ref": ref0_path,
            "tar1": tar1_path,
            "tar2": tar2_path,
            "tar3": tar3_path,
            "mod1": modifiers[0],
            "mod2": modifiers[1],
            "mod3": final_composed_modifier
        }
        print(new_item)
        new_triplets.append(new_item)

    # TODO 存储文件路径
    save_file = "/mnt/longvideo/chenyanzhe/Multiturn/data/modifiers/FC_part3_{}.json".format(COUNT)
    with open(save_file, "w") as json_file:
        json.dump(new_triplets, json_file, indent=4)
    print(f"数据已保存到 {save_file}")
