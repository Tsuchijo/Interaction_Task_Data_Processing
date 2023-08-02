## Script to automatically process trials from the raw data files when they are uploaded
# !/usr/bin/env python3

# Import libraries
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import re
import math
import argparse

## Main function
# Takes as args the path to the raw data file and the path to the processed data file
def main(args):
    #TODO add main functionality
    pass

if __name__ == 'main':
    parser = argparse.ArgumentParser(description='Process raw csv files to create processed data files used to segment videos')
    parser.add_argument('raw_data_path', type=str, help='Path to raw data directories')
    parser.add_argument('processed_data_path', type=str, help='Path to processed data file')
    args = parser.parse_args()
    main(args)
