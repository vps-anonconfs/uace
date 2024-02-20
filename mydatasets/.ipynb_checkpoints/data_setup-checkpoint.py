# based on the scripts:
# https://github.com/princetonvisualai/OverlookedFactors/blob/master/data_setup.py
# https://huggingface.co/spaces/mfrashad/CharacterGAN/blob/d04f9b292dd6044addb90330daaaa956c36ee343/netdissect/broden.py

"""
1. Downloads broden dataset, which contains PASCAL and ADE mydatasets
2. Parses the downloaded dataset and writes to separate PASCAL and ADE to separate pkl files 
"""
import csv
import pandas as pd
import numpy as np
import os
import pickle
from PIL import Image
from sklearn.model_selection import train_test_split
from urllib.request import urlopen
import zipfile

def ensure_broden_downloaded(directory, resolution, broden_version=1):
    assert resolution in [224, 227, 384]
    baseurl = 'http://netdissect.csail.mit.edu/data/'
    dirname = 'broden%d_%d' % (broden_version, resolution)
    if os.path.isfile(os.path.join(directory, dirname, 'index.csv')):
        return # Already downloaded
    zipfilename = 'broden1_%d.zip' % resolution
    download_dir = os.path.join(directory, 'download')
    os.makedirs(download_dir, exist_ok=True)
    full_zipfilename = os.path.join(download_dir, zipfilename)
    if not os.path.exists(full_zipfilename):
        url = '%s/%s' % (baseurl, zipfilename)
        print('Downloading %s' % url)
        data = urlopen(url)
        with open(full_zipfilename, 'wb') as f:
            f.write(data.read())
    print('Unzipping %s' % zipfilename)
    with zipfile.ZipFile(full_zipfilename, 'r') as zip_ref:
        zip_ref.extractall(directory)
    assert os.path.isfile(os.path.join(directory, dirname, 'index.csv'))

    
def parse_dataset(dataset_name, data_fldr):
    image_df = pd.read_csv(f"{data_fldr}/index.csv")
    images = []
    labels = pd.Series([])
    for idx in image_df.index:
        if (image_df['image'][idx]).split('/')[0] != dataset_name:
            continue
        full_image_name = data_fldr + "/images/{}".format(image_df['image'][idx])

        images.append(full_image_name)
        labels[full_image_name] = []

        for cat in ['object', 'part']:
            if image_df[cat].notnull()[idx]:
                for x in image_df[cat][idx].split(';'):    
                    img_labels = Image.open(data_fldr + "/images/{}".format(x))
                    numpy_val = np.array(img_labels)[:, :, 0]+ 256* np.array(img_labels)[:, :, 1]
                    code_val = [i for i in np.sort(np.unique(numpy_val))[1:]]
                    labels[full_image_name] += code_val

    images_train, images_valtest = train_test_split(images, test_size=0.4, random_state=42)
    images_val, images_test = train_test_split(images_valtest, test_size=0.5, random_state=42)

    return images_train, images_val, images_test, labels

    
    
if __name__ == '__main__':
    fldr = '/scratch/vp421/data'
    # ensure_broden_downloaded(fldr, 224)
    
    data_fldr = f"{fldr}/broden1_224/"
    images_train, images_val, images_test, labels = parse_dataset('ade20k', data_fldr)
    with open(f'{fldr}/ade20k_imagelabels.pkl', 'wb+') as handle:
        pickle.dump({'train': images_train, 'val':images_val, 'test':images_test, 'labels':labels}, handle)

    images_train, images_val, images_test, labels = parse_dataset('pascal', data_fldr)
    with open(f'{fldr}/pascal_imagelabels.pkl', 'wb+') as handle:
        pickle.dump({'train': images_train, 'val':images_val, 'test':images_test, 'labels':labels}, handle)
