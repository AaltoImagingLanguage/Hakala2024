# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 14:37:14 2017

quick analyze results.mat 

@author: thakala /fixes tlk 2023

"""

from __future__ import division
import argparse
from scipy.io import loadmat, savemat


parser = argparse.ArgumentParser(description='Check percentage of correct predictions in results.mat.')

parser.add_argument('input_file', type=str, default='results.mat',
                    help='The file that contains the results data; should be a .mat file.')

args = parser.parse_args()
m = loadmat(args.input_file)

acc = m['pairwise_accuracies']

nn=[]
nn = [e for e in acc.flatten() if e==1]
oikein = len(nn)
perc = len(nn)/((acc.shape[1]*(acc.shape[1]-1))/2)
print(perc)

