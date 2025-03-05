import os
import cv2
from tqdm import tqdm
import base64
import json

if __name__ == '__main__':
    img_root = "/mnt/longvideo/chenyanzhe/fashion-caption/DatasetFC/fashion_images"
    # TODO 存储名称
    html_file = "/mnt/longvideo/chenyanzhe/Multiturn/visualize/fc_modifiers_1.html"
    html_file_fp = open(html_file, 'w')
    html_file_fp.write('<html>\n<body>\n')
    html_file_fp.write('<meta charset="utf-8">\n')
    html_file_fp.write('<p>\n')
    html_file_fp.write('<table border="0" align="center">\n')

    triplets = []
    # TODO modified text文件名称
    with open("/mnt/longvideo/chenyanzhe/Multiturn/data/modifiers/FC_composed_1.json", "r") as f:
        triplets.extend(json.load(f))

    for idx, item in tqdm(enumerate(triplets), total=len(triplets)):
        if idx > 50:
            break
        html_file_fp.write('<tr>\n')
        html_file_fp.write('<tr><td height="10" colspan="100"><hr></td></tr>\n')

        ref_name = item["ref"]
        tar1_name = item["tar1"]
        tar2_name = item["tar2"]
        tar3_name = item["tar3"]
        mod1 = item["mod1"]
        mod2 = item["mod2"]
        mod3 = item["mod3"]
        try:
            # ref -> tar1
            img_path = os.path.join(img_root, ref_name)
            img = cv2.imread(img_path)
            img_h, img_w, _ = img.shape
            imgdata = cv2.imencode('.jpeg', img)[1].tobytes()
            html_file_fp.write(
                """
                <td bgcolor=%s align='center'>
                    <img width="300" height="400" src='data:image/jpeg;base64, %s'>
                    <br> name: %s
                    <br> modifier: %s
                </td>
                """ % ('white', base64.b64encode(imgdata).decode(), ref_name, mod1))

            # tar1 -> tar2
            img_path = os.path.join(img_root, tar1_name)
            img = cv2.imread(img_path)
            img_h, img_w, _ = img.shape
            imgdata = cv2.imencode('.jpeg', img)[1].tobytes()
            html_file_fp.write(
                """
                <td bgcolor=%s align='center'>
                    <img width="300" height="400" src='data:image/jpeg;base64, %s'>
                    <br> name: %s
                    <br> modifier: %s
                </td>
                """ % ('white', base64.b64encode(imgdata).decode(), tar1_name, mod2))

            # tar2 -> tar3
            img_path = os.path.join(img_root, tar2_name)
            img = cv2.imread(img_path)
            img_h, img_w, _ = img.shape
            imgdata = cv2.imencode('.jpeg', img)[1].tobytes()
            html_file_fp.write(
                """
                <td bgcolor=%s align='center'>
                    <img width="300" height="400" src='data:image/jpeg;base64, %s'>
                    <br> name: %s
                    <br> modifier: %s
                </td>
                """ % ('white', base64.b64encode(imgdata).decode(), tar2_name, mod3))

            # tar3
            img_path = os.path.join(img_root, tar3_name)
            img = cv2.imread(img_path)
            img_h, img_w, _ = img.shape
            imgdata = cv2.imencode('.jpeg', img)[1].tobytes()
            html_file_fp.write(
                """
                <td bgcolor=%s align='center'>
                    <img width="300" height="400" src='data:image/jpeg;base64, %s'>
                    <br> name: %s
                </td>
                """ % ('green', base64.b64encode(imgdata).decode(), tar3_name))
        except:
            continue

        html_file_fp.write('</tr>\n')
        html_file_fp.write('<tr><td height="10" colspan="100"><hr></td></tr>\n')

    html_file_fp.write('</table>\n')
    html_file_fp.write('</p>\n')
    html_file_fp.write('</body>\n</html>')

    print(f"{html_file}存储完毕")
