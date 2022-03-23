# Data

Here we present the data that we have annotated. RaCCoon.tsv contains the training data, and testset contains test captions for different generation models while COIN.tsv contains the ground truth captions.

## RaCCoon
We have collected ratings on the quality of different image descriptions with coherence labels for a subset of 1000 images from the Conceptual Captions (CC)training dataset (Ng et al., 2020). With this paper, we are publishing this dataset as a benchmark for evaluation metrics that are coherence-aware. The set-up of the data collection is as follows: CC images are input into a caption-generation model created by Alikhani et al. (2020). This model generates coherence-aware descriptions for input
images in 4 different coherence classes of Meta, Visible, Subjective, Story. These 4,000 image/caption pairs are then presented to human annotators who are asked to select the correct
coherence label for each pair:
• Meta: the caption talks about when, where, and how the picture is taken. Meta-talk in Schiffrin (1980)
• Visible: the caption is true just by looking at the picture. Restatement relation in Prasad et al. (2008a).
• Subjective: the captions is the matter of opinion. Evaluation relation in Hobbs (1985).
• Story: text and image work like story and illustration. Occasion relation in Hobbs (1985).
After the annotator selects a specific coherence label from the above, we ask them to rate the quality of the captions, given the label, on a scale of 1 to
5. We use these annotations as training data for our coherence-aware captioning metric, COSMic. We call this data we annotated RaCCoon (Ratings for
Conceptual Caption).

## COIN
OpenImages Ground Truth Captions To create an out of domain test set we asked our annotators to write Visible captions for 1,000 images from the OpenImages dataset (Kuznetsova et al.,
2020a). We call this dataset COIN (Corpus of OpenImages with Natural descriptions). 
