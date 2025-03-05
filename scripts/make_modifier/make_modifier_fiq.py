import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import re
import json


def get_composed_parts(simi1, simi2, simi3):
    random_word = random.choice(["design", "feature", "design"])
    if simi3 == "None" or simi3 is None:
        return "4. keep the FIRST's {} {} and the SECOND's {} {}".format(
            simi1, random_word, simi2, random_word)
    else:
        return "4. keep the FIRST's {} {}, the SECOND's {} {} and the THIRD's {} {}".format(
            simi1, random_word, simi2, random_word, simi3, random_word)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=str, help="GPU card id")
    args = parser.parse_args()
    COUNT = int(args.count)
    print("COUNT id:", COUNT)

    TOTAL_NUM = 11  # TODO 注意一共多少个并行进程

    # TODO pair文件路径
    with open('/mnt/longvideo/chenyanzhe/Multiturn/data/pairs/Fiq-pairs-val.txt', 'r') as file:
        lines = file.readlines()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir = "/mnt/longvideo/chenyanzhe/fashion-caption/Xwin"
    model = AutoModelForCausalLM.from_pretrained("Xwin-LM/Xwin-LM-13B-V0.2", torch_dtype=torch.float16,
                                                 cache_dir=cache_dir)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("Xwin-LM/Xwin-LM-13B-V0.2", cache_dir=cache_dir)

    new_triplets = []
    # TODO 注意划分部分
    # lines = lines
    lines = lines[:int(len(lines) / 3)]
    for idx, line in tqdm(enumerate(lines), total=len(lines)):
        if not (idx % TOTAL_NUM == (COUNT - 1)):
            continue
        ref0_name, tar1_name, tar2_name, tar3_name, ref0_cap, tar1_cap, tar2_cap, tar3_cap, simi1, simi2, simi3 = line.strip().split(
            '\t')
        names = [ref0_name, tar1_name, tar2_name, tar3_name]
        captions = [ref0_cap, tar1_cap, tar2_cap, tar3_cap]
        modifiers = []
        for i in range(len(names) - 2):
            ref_name, tar_name = names[i], names[i + 1]
            ref_cap, tar_cap = captions[i], captions[i + 1]
            prompt = '''The reference image depicts "{}", the target image depicts "{}". Describe the distinctiveness of the target image briefly, with three numbered short phrases. Considering aspects include: product type, color, size, pattern, style, etc. Use the comparative form when appropriate. Do not mention the reference image. Keep answers as simple as possible. Answer:'''.format(
                ref_cap, tar_cap)
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
                final_modifier = "1. is a " + tar_cap.lower()
            modifiers.append(final_modifier)

        prompt_composed = '''The reference images depicts "{}","{}" and "{}", the target image depicts "{}". Describe the distinctiveness of the target image briefly, with three numbered short phrases. Considering aspects include: product type, color, size, pattern, style, etc. Use the comparative form when appropriate. Do not mention the reference image. Keep answers as simple as possible. Answer:'''.format(
            ref0_cap, tar1_cap, tar2_cap, tar3_cap)
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
            differ_modifier = "1. is a " + tar3_cap.lower()
        simi_modifier = get_composed_parts(simi1, simi2, simi3)
        final_composed_modifier = " ".join([differ_modifier, simi_modifier])

        new_item = {
            "ref": ref0_name,
            "tar1": tar1_name,
            "tar2": tar2_name,
            "tar3": tar3_name,
            "mod1": modifiers[0],
            "mod2": modifiers[1],
            "mod3": final_composed_modifier
        }
        print(new_item)
        new_triplets.append(new_item)

    # TODO 存储文件路径
    save_file = "/mnt/longvideo/chenyanzhe/Multiturn/data/modifiers/Fiq_val_{}.json".format(COUNT)
    with open(save_file, "w") as json_file:
        json.dump(new_triplets, json_file, indent=4)
    print(f"数据已保存到 {save_file}")
