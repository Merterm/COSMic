#!/bin/bash

# Rearrange the data
echo "Rearranging annotation data"
python code/arrange_cc_annotations.py "data/gen_caption_ratings.tsv" "data/cc_annotation.tsv" "data/arranged_cc_annotation.tsv" "data/arranged_gen_caption_ratings.tsv"

# Get the generated discourse one-hot labels
echo "Getting the generated caption discourse labels"
python code/extract_discourse_labels_from_annotations.py "data/arranged_gen_caption_ratings.tsv" "output/gen_labels"

# Get the ground truth discourse one-hot labels
echo "Getting the ground truth caption discourse labels"
python code/extract_gt_discourse_labels.py "data/data-both-04-08-cleaned.tsv" "data/arranged_gen_caption_ratings.tsv" "output/ref_labels"

# Get the ratings
echo "Getting the generated caption ratings"
python code/extract_ratings_from_annotations_v2.py "data/arranged_cc_annotation.tsv" "output/ratings"

# Make generated and reference caption text files
echo "Making generated and reference captions"
python code/extract_sentences_from_annotations.py "data/arranged_gen_caption_ratings.tsv" "data/gen_captions.txt" "data/ref.txt"

# Extract BERT features
echo "Running BERT feature extractions"
./code/extract_BERT_features.sh

# Extract Image features
echo "Running BERT feature extractions"
./code/extract_image_features.sh
