"""
Input: a data file (csv or txt) containing N data points, each has M-1 independent features and label.
Data is represented as a table of size NxM. Two columns are separated by comma (',') character

Output: a clustering file (csv) containing K clusters in K lines. Each line i contains the indices (0-based) of the
data points belong to cluster i
"""

import numpy as np 
import argparse
import os

parser = argparse.ArgumentParser(description="Read data and generate ground truth clustering")
parser.add_argument("-f", "--filename", metavar="file_name", type=str, nargs='?', help="data file name")
parser.add_argument("-c", "--columnLabel", type=int, nargs='?', help="index of the label column")

args = parser.parse_args()
#print args

if args.filename is None:
	raise Exception("Filename unspecified")
if args.columnLabel is None:
	raise Exception("Label column unspecified")

try:
	data = np.loadtxt(args.filename, delimiter=",")
	#print data.shape
	labels = data[:, args.columnLabel]
	labels = labels.astype(int)
	labels = np.unique(labels)
	#print labels

	filename = os.path.basename(args.filename)
	outFilename = filename.split('.')[0] + "_groundtruth.csv"

	with open(outFilename, 'w') as fh:
		for label in labels:
			indices = np.argwhere(data[:, args.columnLabel] == label)
			indices = indices.reshape(1, -1)
			np.savetxt(fh, indices, fmt='%d', delimiter=', ')

	print( "Groundtruth saved to {0}".format(outFilename) )
except Exception as e:
	raise



