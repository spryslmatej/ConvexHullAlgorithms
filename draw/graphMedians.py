import matplotlib.pyplot as plt
import pandas as pd
import csv
import numpy as np
import seaborn as sns
import sys
import os

from statistics import median
from os.path import isfile, join
from os import listdir

def getFilePathsInFolder(folder: str) -> list:
    onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
    return onlyfiles

def getFolderPathsInFolder(folder: str) -> list:
    onlyFolders = [f for f in listdir(folder) if os.path.isdir(join(folder, f))]
    return onlyFolders

def loadValues(paths):
    results = []
    for path in paths:
        with open(path, newline='') as csvfile:
            values = []
            spamreader = csv.reader(csvfile, delimiter='\n')
            for row in spamreader:
                values.append(float(row[0]))
                
            algoName = path.split('/')[-1]
            results.append([algoName, values])
    return results

def plot(results, measureName, out_folder):
    plt.figure(figsize=(7.5, 1.1))

    bars = plt.barh(a, b, height=0.9)
    bars[0].set_color('indianred')
    bars[1].set_color('mediumseagreen')
    bars[2].set_color('cornflowerblue')

    plt.xlabel('Time [ms]')

    plt.tight_layout(pad=0)
    plt.autoscale()

    plt.savefig(out_folder + '/' + measureName + ".pdf", bbox_inches='tight')
    # plt.show()
    # plt.clf()
    plt.close()

def calculateMedians(results, a, b):
    for i in results:
        val = median(i[1])
        a.append(i[0])
        b.append(val)


# PROGRAM

# Load test cases
folder = sys.argv[1]
out_folder = sys.argv[2]
testCases=getFolderPathsInFolder(folder)

for testCase in testCases:
    # Load results
    paths = getFilePathsInFolder(folder + '/' + testCase + '/res')
    paths.sort()

    fullPaths = []

    for it in paths:
        fullPaths.append(folder + '/' + testCase + '/res/' + it)
    
    results = loadValues(fullPaths)

    # Calculate medians
    a=[]
    b=[]
    calculateMedians(results, a, b)

    # Plot
    plot(results, testCase, out_folder)
    
