#!/bin/bash

cd /ihome/malikhani/mei13/projects/BLEURT_google/

source /ihome/malikhani/mei13/miniconda3/etc/profile.d/conda.sh
conda activate keras

python code/extract_image_features.py 0
