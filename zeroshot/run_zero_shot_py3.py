import argparse
import numpy as np
from scipy.io import loadmat, savemat
from zeroshotdecoding import zero_shot_decoding

# norm_file = '/neuro/data/redness/semantic_features/AaltoNorms/aalto85/Aalto85_sorted20160204.mat'
# the norm_file contains a struct with the data in sorted.mat and the item labels in sorted.word
# and sorted.wordsNoscand (without scandinavian letters, this version is used in the input filenames)

# Handle command line arguments
parser = argparse.ArgumentParser(description='Run zero-shot learning on a single subject.')
parser.add_argument('input_file',nargs='?', type=str, default = None,
                    help='The file that contains the subject data; should be a .mat file.')
parser.add_argument('-i', '--subject-id', metavar='N', type=int, default=1,
                    help='The subject-id (as a number). This number is recorded in the output .mat file. Defaults to 1.')
parser.add_argument('-o', '--output', metavar='filename', type=str, default='./results.mat',
                    help='The file to write the results to; should end in .mat. Defaults to ./results.mat.')
parser.add_argument('--norms', metavar='filename', type=str, default=None,
                    help='The file that contains the norm data.')
parser.add_argument('--control', metavar='filename', type=str, default=None,
                    help='The file that contains the control variables; should end in .mat. Defaults to None, which disables adding control variables.')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Whether to show a progress bar')
parser.add_argument('-b', '--break-after', metavar='N', type=int, default=-1,
                    help='Break after N iterations (useful for testing)')
parser.add_argument('-m', '--morphemes', metavar='filename', type=str, default=None,
                    help='list of segmented words. one word per line, spaces between segments')
parser.add_argument('-w', '--wordvec', metavar='filename', type=str, default=None,
                    help='word vectors for segments')
parser.add_argument('-t', '--vocab', metavar='filename', type=str, default=None,
                    help='vocab that corresponds the wordvec vectors')                  
parser.add_argument('-pm', '--permutemorphs', metavar='N', type=int, default=0,
                    help='permute morpheme vectors')                  
parser.add_argument('-pw', '--permutewords', metavar='N', type=int, default=0,
                    help='permute words')                  
parser.add_argument('-ns', '--nosave', metavar='N', type=int, default=0,
                    help='dont save everything, save only the accuracy')                  


parser.add_argument('-d', '--distance-metric', type=str, default='sqeuclidean',
                    help=("The distance metric to use. Any distance implemented in SciPy's "
                          "spatial.distance module is supported. See the docstring of "
                          "scipy.spatial.distance.pdict for the exhaustive list of possitble "
                          "metrics. Here are some of the more useful ones: "
                          "'euclidean' - Euclidean distance "
                          "'sqeuclidean' - Squared euclidean distance (the default) "
                          "'correlation' - Pearson correlation "
                          "'cosine' - Cosine similarity "))
args = parser.parse_args()
verbose = args.verbose
if args.break_after > 0:
    break_after = args.break_after
else:
    break_after = None


if(args.norms is None):
    # use vocab and vectors tsv to build the norms for segmented words
    print('morphemefile:', args.morphemes)
    print('vectorsfile:', args.wordvec)
    print('vocabfile:', args.vocab)
    
    
    vocwords = []
    with open(args.vocab, 'r', encoding='utf-8') as file:
        for line in file:
            # Splitting the line by tab and taking the first element (word)
            word = line.split('\t')[0]
            vocwords.append(word)
    
    arrays = []
    with open(args.wordvec, 'r', encoding='utf-8') as file:
        for line in file:
            # Converting each line into a list of floats
            numbers = [float(num) for num in line.split()]
            arrays.append(numbers)

    # Converting the list of lists into a 2D numpy array
    vocvects = np.array(arrays)
    
    # permute vocvects if we want to random permute morphemes 
    if args.permutemorphs ==1:
        permuted_array = np.random.permutation(vocvects)
        vocvects = permuted_array.copy()
        print("permuted morphemes randomly")
    
    morph_dict = {}
    with open(args.morphemes, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            key = ''.join(parts)
            morph_dict[key] = parts
    
    
    norm_words=[]
    
    n = len(morph_dict.keys())
    nv = vocvects.shape[1]
    y = np.empty((n, nv))
    
    ind=0
    for key in morph_dict:
        norm_words.append(key)
        print(key)
        segments = morph_dict[key]
        segvects=[]
        for seg in segments:
            print(seg)
            index = [i for i, word in enumerate(vocwords) if seg == word]
            #print('indices:', index)
    
            segvect = vocvects[index]
            segvects.append(segvect)
        sumvects = sum(segvects)
        y[ind] = sumvects
        ind = ind+1


       
print('Subject:', args.subject_id)
print('Input:', args.input_file) 
if args.control is not None:
    print('Control:', args.control)
print('Output:', args.output)


# Load semantic norms from a mat file with the struct sorted.mat

if(args.norms is not None):
    print('Norms:', args.norms)
    m = loadmat(args.norms)
    y = m['sorted']['mat'][0][0]

    
    if 'wordsNoscand' in m['sorted'].dtype.names:
        norm_words = [x[0][0] for x in m['sorted']['wordsNoscand'][0][0]]
    else:
    #    norm_words = [x[0][0] for x in m['sorted']['wordNoscand'][0][0]]
        norm_words = [(x[0][0]) for x in m['sorted']['word'][0][0]]

n_targets = y.shape[1]


# Load MEG data
# the mat-file contains a matrix named megdata with the following dimensions: items x time x vertices/channels 
# m = loadmat('/data/thakala/morffimeg/megdata_filt20.mat')

m = loadmat(args.input_file)
n_stimuli, n_times, n_vertices = m['megdata'].shape
X = m['megdata'].reshape(n_stimuli, n_times * n_vertices)
target_word_labels = [w.strip() for w in m['target_word_labels']]

# Make norms and MEG data be in the same order
# and remove words that are missing either megdata or norm..  (th)

megidx=-1
X1=[]
y1=[]
norm_words1=[]
for w in target_word_labels:
    megidx = megidx+1
    if w in norm_words:
        ii = norm_words.index(w)
        X1.append(X[megidx,:])
        y1.append(y[ii])        
        norm_words1.append(w)
X = np.array(X1)
y = np.array(y1)
norm_words = norm_words1
target_word_labels = norm_words

print('number of words', len(norm_words))

# permutation shuffle
if(args.permutewords ==1):
    np.random.shuffle(X) 



# Load control variables if needed
if args.control is not None:
    m_c = loadmat(args.control)
    control = m_c['sorted']['mat'][0][0]
    n_control_variables = control.shape[1]

    # Make control variables be in the correct order
    control_words = [x[0][0] for x in m_c['sorted']['word'][0][0]]
    control_words = [control_words[i] for i in order]
    control = control[order]

    # Regress out the effects of the control variables on the semantic norms
    control = np.hstack((control, np.ones((n_stimuli, 1))))
    control_weights = np.linalg.lstsq(control, y)[0]
    y_control = y - control.dot(control_weights)
    control_weights = control_weights[:-1, :]
else:
    control_weights = np.zeros((1, n_targets))

# Run the zero-shot decoding
pairwise_accuracies, model, target_scores, predicted_y, patterns = zero_shot_decoding(
    X, y, verbose=verbose, break_after=break_after, metric=args.distance_metric,
)

print(model.alpha_)

results_all = {
    'pairwise_accuracies': pairwise_accuracies,
    'weights': model.coef_,
    'control_weights': control_weights,
    'feat_scores': target_scores,
    'subject': args.subject_id,
    'inputfile': args.input_file,
    'normfile': args.norms,
    'alphas': model.alpha_,
    'target_word_labels': target_word_labels,
    'predicted_y': predicted_y,
}
results = {
    'pairwise_accuracies': pairwise_accuracies,
    'inputfile': args.input_file,
    'normfile': args.norms,
     'target_word_labels': target_word_labels,
}

if 'category_labels' in m:
    results['category_labels'] = m['category_labels']
    results_all['category_labels'] = m['category_labels']

acc = pairwise_accuracies
nn=[]
nn = [e for e in acc.flatten() if e==1]
oikein = len(nn)
perc = len(nn)/((acc.shape[1]*(acc.shape[1]-1))/2)
print(perc)


if(args.nosave ==1):
    with open(args.output, 'w') as file:
        file.write(str(perc))
else: 
    savemat(args.output, results_all)
