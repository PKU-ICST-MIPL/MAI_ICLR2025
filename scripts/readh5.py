import h5py
import csv
from PIL import Image
from tqdm import tqdm

csvfile_path = './validate'
path_behind = 0
headline = ['index', 'description', 'category']
file_h5 = h5py.File('fashiongen_256_256_validation.h5', 'r')

for i in tqdm(range(0, len(file_h5['index']) - 1)):
    index = file_h5['index'][i][0]
    try:
        category = str(file_h5['input_category'][i][0], 'UTF-8')
        description = str(file_h5['input_description'][i][0], 'UTF-8')
    except Exception as e:
        continue
    img = Image.fromarray(file_h5['input_image'][i])
    img.save('./images/' + str(index) + '.jpg')
    if i % 50000 == 0:
        path_behind += 1
        csvfile = open(csvfile_path + str(path_behind) + '.csv', 'w', newline='')
        writer = csv.DictWriter(csvfile, fieldnames=headline)
        writer.writeheader()

    newData = {'index': str(index), 'description': description, 'category': category}
    writer.writerow(newData)

csvfile.close()

csvfile_path = './validate'
path_behind = 0
headline = ['index', 'description', 'category']
file_h5 = h5py.File('fashiongen_256_256_validation.h5', 'r')

for i in tqdm(range(0, len(file_h5['index']) - 1)):
    index = file_h5['index'][i][0]
    try:
        category = str(file_h5['input_category'][i][0], 'UTF-8')
        description = str(file_h5['input_description'][i][0], 'UTF-8')
    except Exception as e:
        continue
    img = Image.fromarray(file_h5['input_image'][i])
    img.save('./images/' + str(index) + '.jpg')
    if i % 50000 == 0:
        path_behind += 1
        csvfile = open(csvfile_path + str(path_behind) + '.csv', 'w', newline='')
        writer = csv.DictWriter(csvfile, fieldnames=headline)
        writer.writeheader()

    newData = {'index': str(index), 'description': description, 'category': category}
    writer.writerow(newData)

csvfile.close()
