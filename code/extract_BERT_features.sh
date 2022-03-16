#!/bin/bash

source /ihome/malikhani/mei13/miniconda3/etc/profile.d/conda.sh
conda activate python2

# Extract features from the generated captions
echo 'Extracting BERT CLS features from generated captions'

python bert/extract_CLS_features.py \
  --input_file=data/gen_captions.txt \
  --output_file=output/gen_feat.jsonl \
  --vocab_file=bert/vocab.txt \
  --bert_config_file=bert/bert_config.json \
  --init_checkpoint=bert/bert_model.ckpt \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=128 \
  --batch_size=8

# Extract features from the generated captions
echo 'Extracting BERT CLS features from reference captions'

python bert/extract_CLS_features.py \
  --input_file=data/ref.txt \
  --output_file=output/ref_feat.jsonl \
  --vocab_file=bert/vocab.txt \
  --bert_config_file=bert/bert_config.json \
  --init_checkpoint=bert/bert_model.ckpt \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=128 \
  --batch_size=8
