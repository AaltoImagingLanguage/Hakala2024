# Hakala2024
This repository contains the code and data utilized in the paper:  "Subword representations successfully decode brain responses to morphologically complex written words"  by Tero Hakala, Tiina Lindh-Knuutila, Annika Hultén, Minna Lehtonen, and Riitta Salmelin Department of Neuroscience and Biomedical Engineering, Aalto University, Finland


21.5.2024


###

This should work at least with the following version:

Python 3.10.13 
sklearn.__version__ '1.3.2'
scipy.__version__ '1.11.3'


###
quickstart

To test whether everything works, you can try running the following command in the morppirepo/zeroshot directory.

python run_zero_shot_py3.py -d cosine --verbose --vocab ../vekt/trigrams/vocab.tsv --wordvec ../vekt/trigrams/vectors.tsv --morphemes ../vekt/trigrams/morppisanat_170_trigrams.txt --nosave 1 --output testrun_trigrams.txt ../megdata/megdata_filt40_check_noresamp_8.mat


This command should calculate the decoding accuracy for word vectors constructed from 3-gram segment vectors, using the MEG time window spanning 350-450ms, and store the resulting accuracy in the file: testrun_trigrams.txt

This accuracy should be around 0.6424.


###

Contents by directory:

zeroshot

run_zero_shot_py3.py
This code is for decoding word vectors from evoked brain activation. It serves as a frontend to the ridge-regression function in the scikit-learn library.

The script takes as input the complete word vectors and MEG data vectors.

Alternatively, instead of complete word vectors, it can take a list of segmented words and the individual segment vectors. It then constructs the word vectors by summing the corresponding segment vectors for each word.

The script can perform permutation shuffling either at the word level (shuffling word labels) or at the segment level (shuffling segment labels before constructing the word vectors, as described in the paper).


###

megdata

Evoked potentials for each word in the experiment, averaged over subjects.

306 Channels. 
Data is sampled at 1000 Hz
Low-pass filtered at 40Hz.

Cleaned from artifacts (limit 3000fT/cm for gradiometers).
ICA components of ocular artifacts are removed.

Data with incorrect behavioral responses are removed before
averaging over subjects.


MNE Evokeds files (averaged over subjects for each word):
evokeds_averages.fif

event tiggercodes and corresponding stimulus words (only real words are used):
triggercodes.csv


In this file, the stimulus onset is at point 238 from the beginning (200ms baseline + 38 ms delay due to the projector refresh rate until the stimulus is visible)


Data is split into 100ms timewindows (overlap 50ms)
MEG datavectors used in the decoding:

filename                             time window

megdata_filt40_check_noresamp_1.mat  0 - 100 ms
megdata_filt40_check_noresamp_2.mat  50 - 150
megdata_filt40_check_noresamp_3.mat  100 - 200
megdata_filt40_check_noresamp_4.mat  150 - 250
megdata_filt40_check_noresamp_5.mat  200 - 300
megdata_filt40_check_noresamp_6.mat  250 - 350
megdata_filt40_check_noresamp_7.mat  300 - 400
megdata_filt40_check_noresamp_8.mat  350 - 450
megdata_filt40_check_noresamp_9.mat  400 - 500
megdata_filt40_check_noresamp_10.mat 450 - 550
megdata_filt40_check_noresamp_11.mat 500 - 600
megdata_filt40_check_noresamp_12.mat 550 - 650
megdata_filt40_check_noresamp_13.mat 600 - 700
megdata_filt40_check_noresamp_14.mat 650 - 750
megdata_filt40_check_noresamp_15.mat 700 - 800

##

wordvectors

The segment vectors were constructed using the gensim word2vec skip-gram model with the Finnish internet corpus in collaboration with TurkuNLP group. (Luotolahti, J., Kanerva, J., Laippala, V., Pyysalo, S., & Ginter, F. (2015). Towards
universal web parsebanks. Proceedings of the Third International Conference on Dependency Linguistics (Depling 2015), 211–220.)

Before running word2vec, each word in the corpus was segmented into word segments according to respective segmentation schemes.  Surface corresponds to whole words (no segmentation).

The subdirectories contain word/segment vectors for various segmentations.

surface		Whole words
unigrams 	1-grams
bigrams 	2-grams
trigrams 	3-grams
morfessor 	Segmentations by Morfessor, a statistical model of morphology
ling  		Linguistic segmentation by commercial Linsoft utility
random 		Random segmentation
morfessor_modified     Morfessor segmentation on modified corpus (experiment words removed from the corpus)
ling_modified	       Linguistic segmentation on modified corpus (experiment words removed from the corpus)


The morfessor, ling, and surface directories are further divided into w1 to w7 subdirectories. These correspond to specific skip-gram window lengths that were used to construct the vectors (e.g., w7: 7 segments before to 7 segments after the target). 

Each directory contains the following files.

morppisanat_170_bigrams.txt  List of segmented words (in this case, segmented into 2-grams)
vocab.tsv	 	     List of individual segments (these come from word2vec)
vectors.tsv		     Vectors for the segments in vocab.tsv (these come from word2vec)
bigrams_170_w7_sum.mat 	     Complete word vectors constructed by summing corresponding segments for each word (This file is not needed anymore as the zeroshot script can also construct the word vectors from the segments)



##

Experimental stimuli

The screen images of the words shown during the experiment.

##

supplementary

Figures of containing the hierarchical clustering (complete linkage, cosine) of the different word vectors in pdf-format. 
