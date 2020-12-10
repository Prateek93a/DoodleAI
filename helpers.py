from PIL import Image
import numpy as np
from pathlib import Path


def load_categories(source_path):
    try:
        categories = []

        with open(source_path, 'r') as file:
            for line in file.readlines():
                categories.append(line.strip())

        return categories

    except:
        print('An error occured while loading categories.')


def download_npy(categories, dest_path):
    try:
        Path('{}'.format(dest_path)).mkdir(parents=True, exist_ok=True)
        base = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'

        for c in categories:
            cls_url = c.replace('_', '%20')
            path = base+cls_url+'.npy'
            print(path)
            urlretrieve(path, '{}/{}.npy'.format(dest_path, c))

    except:
        print('An error occured while downloading.')


def convert_npy_to_png(categories, source_path, dest_path):
    try:
        Path('{}/train'.format(dest_path)).mkdir(parents=True, exist_ok=True)
        Path('{}/test'.format(dest_path)).mkdir(parents=True, exist_ok=True)

        for c in categories:
            Path('{}/train/{}'.format(dest_path, c)
                 ).mkdir(parents=True, exist_ok=True)
            Path('{}/test/{}'.format(dest_path, c)
                 ).mkdir(parents=True, exist_ok=True)

            c_img = np.load('npy_files/'+c+'.npy')
            train_set = (c_img[:50000])[np.random.choice(
                c_img[:50000].shape[0], size=12000, replace=False)]
            test_set = (c_img[50000:])[np.random.choice(
                c_img[50000:].shape[0], size=8000, replace=False)]

            for i, img in enumerate(train_set):
                img = img.reshape((28, 28))
                img = Image.fromarray(img, 'L')
                img.save('./Data/train/{}/{}_train_{}.png'.format(c, c, i))

            for i, img in enumerate(test_set):
                img = img.reshape((28, 28))
                img = Image.fromarray(img, 'L')
                img.save('./Data/test/{}/{}_test_{}.png'.format(c, c, i))

    except:
        print('An error occured while converting files.')
