import numpy as np
import scipy
import utils
import time
import argparse
import os
import sys
import shutil
import datetime
import matplotlib
import matplotlib.pyplot as plt

import pickle

def PlotSomeLoss(outputPath, iterations, first=0):
    tmp = os.path.join(outputPath, "loss.npy")
    with open (tmp, 'rb') as fp:
        loss = pickle.load(fp)

    myRange = range(first, iterations)
    diff = len(loss) - len(myRange)
    fig = plt.figure()
    plt.plot(myRange, loss[diff:])
    plt.ylabel('Cost')
    plt.xlabel('Iteration')
    tmp = os.path.join(outputPath, "loss.png")
    fig.savefig(tmp)

def main():
    # Define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("-t", "--iterations", help="Max number of iterations to run", type=int)
    parser.add_argument("-f", "--first", help="First iteration to consider", type=int)
    parser.add_argument("-e", "--last", help="Last iteration to consider", type=int)
    args = parser.parse_args()

    if args.iterations is None or args.iterations <= 0:
        tmp = os.path.join(args.output, "iterations.npy")
        with open (tmp, 'rb') as fp:
            iterations = pickle.load(fp)
    else:
        iterations = args.iterations

    if args.first is not None:
        first = args.first
    else:
        first = 0

    PlotSomeLoss(args.output, iterations, first)



if __name__ == "__main__":
    main()
