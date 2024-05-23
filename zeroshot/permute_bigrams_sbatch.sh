#!/bin/bash

# run decoding with 2000 permutations on morpheme level on triton cluster

# Make sure to request only the resources you really need to avoid cueing
#SBATCH --time=15:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH -n 1
#SBATCH -c 16
# Do 2000 random iterations. 
#SBATCH --array=1-2000

# Location of the input file. Make sure the path and filename are correct!
INPUT_PATH=/m/nbe/work/thakala/vekt

INPUT_FILE=$INPUT_PATH/megdata/megdata_filt40_check_noresamp_8.mat

# Location of the semantic norms file. Make sure the path and filename are correct!
# this is not needed if we construct vectors online from vocab and vectors
NORM_FILE=$INPUT_PATH/bigrams/bigrams_170_w7_sum.mat

# Use these files to construct word vectors from segment vectors
VDIR=$INPUT_PATH/bigrams
VOCAB_FILE=$VDIR/vocab.tsv
VECT_FILE=$VDIR/vectors.tsv
MORPHEME_FILE=$VDIR/morppisanat_170_bigrams.txt

# The directory in which to place the results. Make sure the path is correct!
OUTPUT_PATH=$INPUT_PATH/results/permute_bigrams/

# Make sure the output path exists
mkdir -p $OUTPUT_PATH

# Construct the names of the output files of this run
OUTPUT_FILE=$OUTPUT_PATH/results_bigrams_w7_sum_"$SLURM_ARRAY_TASK_ID".txt
LOG_FILE=$OUTPUT_PATH/res_bigrams_w7_sum_"$SLURM_ARRAY_TASK_ID".log

# On triton, uncomment this to load the Python environment. On taito, you
# presumably installed Python yourself.
module load anaconda

# Run the analysis!
srun -o $LOG_FILE python /m/nbe/work/thakala/zeroshot/run_zero_shot_py3.py -d 'cosine' --verbose --vocab $VOCAB_FILE --wordvec $VECT_FILE --morphemes $MORPHEME_FILE --permutemorphs 1 --nosave 1 --output $OUTPUT_FILE $INPUT_FILE
