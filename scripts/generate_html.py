import os
import sys
import re
import glob, cv2
import time
import numpy as np
from tqdm import tqdm
import random
import base64
from PIL import Image


def read_txt(input_file_path, mode="single"):
    data_list = []
    with open(input_file_path, 'r') as infile:
        lines = infile.readlines()  # 读取所有行
        for line in lines:
            # 将每一行的内容分割成列表
            if mode == "single":
                elements = line.strip()
            else:
                elements = line.strip().split()
            # 只保留前10个元素
            data_list.append(elements)
    return data_list


def vis_ann(ref_list, tar_list, cap_list, ind_list, html_file, lines):
    prefix = "/mnt/longvideo/chenyanzhe/Multiturn/data/DatasetFC/fashion_images/"

    html_file_fp = open(html_file, 'w')
    html_file_fp.write('<html>\n<body>\n')
    html_file_fp.write('<meta charset="utf-8">\n')

    html_file_fp.write('<p>\n')
    html_file_fp.write('<table border="0" align="center">\n')
    
    for i, (ref_path, tar_path, cap, idx_list) in tqdm(enumerate(zip(ref_list, tar_list, cap_list, ind_list))):
        if i < lines:
            continue
        if i > lines * 2:
            break
        html_file_fp.write('<tr>\n')
        html_file_fp.write('<tr><td height="10" colspan="100"><hr></td></tr>\n')
        
        img = cv2.imread(ref_path)
        img_h, img_w, _ = img.shape
        imgdata = cv2.imencode('.jpg', img)[1].tobytes()
        html_file_fp.write(
            """
            <td bgcolor=%s align='center'>
                <img width="300" height="400" src='data:image/jpeg;base64, %s'>
                <br> ref name: %s
                <br> modified text: %s
            </td>
            """ % ('white', base64.b64encode(imgdata).decode(), ref_path.replace(prefix, ""), cap)
        )

        # tar
        img = cv2.imread(tar_path)
        img_h, img_w, _ = img.shape
        imgdata = cv2.imencode('.jpg', img)[1].tobytes()
        html_file_fp.write(
            """
            <td bgcolor=%s align='center'>
                <img width="300" height="400" src='data:image/jpeg;base64, %s'>
                <br> target name: %s
            </td>
            """ % ('green', base64.b64encode(imgdata).decode(), tar_path.replace(prefix, ""))
        )

        for img_path in idx_list:
            # if name == "":
            #     continue
            img = cv2.imread(img_path)
            img_h, img_w, _ = img.shape
            imgdata = cv2.imencode('.jpg', img)[1].tobytes()
            html_file_fp.write(
                """
                <td bgcolor=%s align='center'>
                    <img width="300" height="400" src='data:image/jpeg;base64, %s'>
                    <br> name: %s
                </td>
                """ % ('white', base64.b64encode(imgdata).decode(), img_path.replace(prefix, ""))
            )

        html_file_fp.write('</tr>\n')
        html_file_fp.write('<tr><td height="10" colspan="100"><hr></td></tr>\n')

    html_file_fp.write('</table>\n')
    html_file_fp.write('</p>\n')
    html_file_fp.write('</body>\n</html>')


if __name__ == '__main__':
    ref_list = read_txt("/mnt/longvideo/chenyanzhe/Multiturn/visualize/ref.txt")
    tar_list = read_txt("/mnt/longvideo/chenyanzhe/Multiturn/visualize/tar.txt")
    cap_list = read_txt("/mnt/longvideo/chenyanzhe/Multiturn/visualize/cap.txt")
    ind_list = read_txt("/mnt/longvideo/chenyanzhe/Multiturn/visualize/ind_short.txt", mode="list")

    output_path = "/mnt/longvideo/chenyanzhe/Multiturn/visualize/FC-show-0519.html"
    vis_ann(ref_list, tar_list, cap_list, ind_list, output_path, lines=100)
    print("存储完毕！")
