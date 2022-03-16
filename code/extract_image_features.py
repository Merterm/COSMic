"""
#########################################################
extract_image_features.py

Description: Uses ResNet50V2 to extract image features from the images

Dependencies:
    - tensorflow
    - keras
    - requests

Author: Mert Inan
Date: 12 Jan 2021

Usage without loaded images:
python extract_image_features.py 0

Usage with loaded images:
python extract_image_features.py 1

#########################################################
"""
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from halo import Halo
import numpy as np
import os
import sys
from PIL import Image
import requests
import io
import json

'''
#_#_#_#_#_#_#_#_#_#_#_#_#_# PARAMETERS #_#_#_#_#_#_#_#_#_#_#_#
'''
filename = "data/arranged_gen_caption_ratings.tsv"
'''
#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#_#
'''

# 1) Instantiate the model
model = ResNet50V2(weights='imagenet', include_top=False, pooling="avg")

# 2) Loop through the reference file and get the image URLs
urls = {}
with open(filename) as f:
    f.readline()
    for line in f.readlines():
        split_line = line.split('\t')
        broken = int(split_line[17])
        id = split_line[16]
        if (not broken) and (split_line[11]) and ('Action' not in id):
            url = split_line[3]
            urls[id] = url
print("length",len(urls))

if int(sys.argv[1]):
    x = np.load('output/images.npy')
    print("Loaded the saved image array.")
else:
    print('No saved array, downloading images')
    # 3) Download the image from the URL and then put it in the input array and delete
    x = np.zeros((len(urls), 224,224,3))
    with Halo(text='Fetching Image URLs', spinner='monkey'):
        for i, (id, url) in enumerate(urls.items()):
            try:
                r = requests.get(url, stream=True, timeout=20)
                img = Image.open(io.BytesIO(r.content))
                img = img.resize((224,224),Image.ANTIALIAS)
                img_arr = image.img_to_array(img)
                x[i] = img_arr
            except Exception as e:
                print('\t', 'After ', i, ' images, an error:', e)

    np.save('output/images', x)

print()
print('Preprocessing...')
print('x before preporc',x.shape)
x = preprocess_input(x)
print('x after preproc', x.shape)
print('Extracting Features...')
features = model.predict(x)
print('features',features.shape)
features = [features[i].squeeze() for i in range(features.shape[0])]

# 4) Save the features
print('Saving the Features...')
np.save("output/img_feats", features, allow_pickle=True)
# feats = {}
# for i, (id, url) in enumerate(urls.items()):
#     feats[id] = [round(float(x), 6) for x in features[i].flat]
# json.dump(feats, open("output/img_feats.json", 'w'))
