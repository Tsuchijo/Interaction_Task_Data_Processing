import os 
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import datetime
import re
import math
import cv2
import scipy.stats as stats
import tdt
import sys


class helper_functions:
    def subtract_time_series(x, index1, index2):
        time_str1 = str(x[index1])
        time_str2 = str(x[index2])
        if time_str2 == 'nan':
            return None
        secs1 = float(str(x[index1])[12:14])
        secs2 = float(str(x[index2])[12:14])
        millis1 = float(str(x[index1])[14:])
        millis2 = float(str(x[index2])[14:])
        hours1 = float(str(x[index1])[8:10])
        hours2 = float(str(x[index2])[8:10])
        mins1 = float(str(x[index1])[10:12])
        mins2 = float(str(x[index2])[10:12])
        # divide millis until there is nothing to the right of the decimal point
        while millis1 > 1:
            millis1 /= 10
        while millis2 > 1:
            millis2 /= 10

        time1 = datetime.datetime(2020, 1, 1, int(hours1), int(mins1), int(secs1), int(millis1 * 1000))
        time2 = datetime.datetime(2020, 1, 1, int(hours2), int(mins2), int(secs2), int(millis2 * 1000))
        diff = (time2 - time1).total_seconds()

        if diff > 100:
            print(diff)
            return None
        else:
            return diff
        
    def process_time_string(time_string):
        time_string = str(time_string)
        if time_string == 'nan':
            return None
        if len(time_string) > 14:
            millis = float(time_string[14:])
        else:
            millis = 0
        secs = int(time_string[12:14])
        mins = int(time_string[10:12])
        hours = int(time_string[8:10])
        day = int(time_string[6:8])
        month = int(time_string[4:6])
        year = int(time_string[0:4])
        # divide millis until there is nothing to the right of the decimal point
        while millis >= 1:
            millis /= 10
        return datetime.datetime(year, month, day, hours, mins, secs, int(millis * 1000))

    def scrub_video_seconds(vid_capture, seconds):
        fps = vid_capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(vid_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        current_pos_seconds = frame_count / fps
        time_to_skip = current_pos_seconds - seconds
        for i in range(int(time_to_skip * fps)):
            vid_capture.read()
        return vid_capture

    # Using OpenCV, takes a reference image of the interaction area and the current frame
    # using the reference image finds a perspective transform to transform and crop the current frame to the reference frame using SIFT algorithm
    def find_area_and_transform(frame, reference):
        # create SIFT object
        sift = cv2.SIFT_create()
        # find keypoints and descriptors for reference and current frame
        kp1, des1 = sift.detectAndCompute(reference,None)
        kp2, des2 = sift.detectAndCompute(frame,None)
        # create BFMatcher object
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        knn_matches = matcher.knnMatch(des1, des2, 2)
        good = []
        for m,n in knn_matches:
            if m.distance < 0.75*n.distance:
                good.append(m)

        #-- Localize the object
        obj = np.empty((len(good),2), dtype=np.float32)
        scene = np.empty((len(good),2), dtype=np.float32)
        for i in range(len(good)):
            #-- Get the keypoints from the good matches
            obj[i,0] = kp1[good[i].queryIdx].pt[0]
            obj[i,1] = kp1[good[i].queryIdx].pt[1]
            scene[i,0] = kp2[good[i].trainIdx].pt[0]
            scene[i,1] = kp2[good[i].trainIdx].pt[1]
        # find the perspective transform
        try:
            M, mask = cv2.findHomography(scene, obj, cv2.RANSAC,5.0)
            return M, len(good)
        except:
            return None, 0

    # iterate through list of paths and get the best matching SIFT transformation
    def find_matching_transformation(scene , paths):
        references_images = [cv2.imread(path) for path in paths]
        transformations = []
        num_matches = []
        for reference in references_images:
            transformation, num_match = helper_functions.find_area_and_transform(scene, reference)
            transformations.append(transformation)
            num_matches.append(num_match)
        return transformations[num_matches.index(max(num_matches))], max(num_matches), paths[num_matches.index(max(num_matches))]
    
    def save_video(video, path):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, 25, (video.shape[2], video.shape[1]))
        # check if output folder exists if not create it
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        for frame in video:
            # convert frame to BGR for writing from grayscale 
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            out.write(frame)
        out.release()

    def detect_triggers(data, threshold=0.0005):
        smoothed_data = np.convolve(data, np.ones(1000)/1000, mode='same')
        diff_data = np.diff(smoothed_data)
        diff_data = np.convolve(diff_data, np.ones(1000)/1000, mode='same')
        #plt.plot(diff_data[:1000000])
        recording = False
        triggered_data = []
        trial_indices = []
        for i in range(len(diff_data)):
            if diff_data[i] > threshold and not recording:
                recording = True
                trial_indices.append(i)
            elif diff_data[i] < -threshold and recording:
                recording = False
                
            elif recording and diff_data[i] < threshold and diff_data[i] > -threshold:
                triggered_data.append(data[i])
        return triggered_data, trial_indices

    ## Given a block of tdt data and a list of time deltas from the start of the trial, return a list of trials
    # params:
    #   stream_data_block: tdt data block object
    #   delta_time: list of time deltas from the start of the trial, floating point seconds
    def stream_to_trials(stream_data_block, delta_time, threshold):
        fs = stream_data_block.fs
        stream = stream_data_block.data
        _, trigger_indices = helper_functions.detect_triggers(stream, threshold=threshold)
        print(len(trigger_indices))
        print(len(delta_time))
        if len(trigger_indices) == len(delta_time):
            skip_index = 1
        else:
            skip_index = 1
        try:
            trials = [ stream[trigger_indices[skip_index]:][int (time * fs + fs) : int ((time + 31) * fs)] for time in delta_time]
        # catch index error
        except:
            print(delta_time[-1] * fs)
            print(trigger_indices)
            print(len(stream[trigger_indices[1]:]))
        return trials


    ## given a date, return an dict of photometry data, with each stream from the block broken up into trials basedon the csv
    # params:
    #   date: string of date in format 'YYYY-MM-DD'
    #   returns: dict {
    #    '465A': array(# trials, 30 secs * fs)
    #    '405A': array(# trials, 30 secs * fs)
    #    '465C': array(# trials, 30 secs * fs)
    #    '405C': array(# trials, 30 secs * fs)
    # }
    def get_trial_photometry_data(key, dict):
        date_format = "%Y-%m-%d %H:%M:%S.%f"
        date_list = pd.read_csv(dict[key]['trial'])['cue_times'].to_list()
        data_block = tdt.read_block(dict[key]['photometry'])
        delta_time = [(datetime.datetime.strptime(date, date_format) - datetime.datetime.strptime(date_list[0], date_format)).total_seconds() for date in date_list]
        return {
            "465A" : helper_functions.stream_to_trials(data_block.streams['_465A'], delta_time, threshold=0.2),
            "405A" : helper_functions.stream_to_trials(data_block.streams['_405A'], delta_time, threshold=0.01),
            '465C' : helper_functions.stream_to_trials(data_block.streams['_465C'], delta_time, threshold=0.2),
            '405C' : helper_functions.stream_to_trials(data_block.streams['_405C'], delta_time, threshold=0.01),
            'fs' : data_block.streams['_465A'].fs
        }


        
    def deltaFF(data, t, method='exp_fit'):
        # if method == 'poly':
        #     reg = np.polyfit(dat2, dat1, 1)
        #     a, b = reg
        #     controlFit = a*dat2 + b
        #     dff = np.divide(np.subtract(dat1, controlFit), controlFit)
        # elif method == 'subtract':
        #     dff1 = np.divide(np.subtract(dat1, np.mean(dat1)), np.mean(dat1))
        #     dff2 = np.divide(np.subtract(dat2, np.mean(dat2)), np.mean(dat2))
        #     dff = dff1 - dff2
        # elif method == 'exp_fit':
        #
        guess_a, guess_b, guess_c = np.max(data), -0.05, np.min(data)
        guess = [guess_a, guess_b, guess_c]
        exp_decay = lambda x, A, b, y0: A * np.exp(x * b) + y0
        params, cov = helper_functions.curve_fit(exp_decay, t, data, p0=guess, maxfev=5000)
        A, b, y0 = params
        best_fit = lambda x: A * np.exp(b * x) + y0
        dff = data#-best_fit(t) + 100  # add DC offset so mean of corrected signal is positive
        dff = (dff - np.mean(dff))/np.mean(dff)
        return stats.zscore(dff), best_fit


    def get_movement_success_trials(key, experiment_data, movement_extract_path):
        try:
            video_titles = os.listdir(movement_extract_path + str(experiment_data[key]['trial'].split('/')[-1][:14]))
        except:
            print('No video found for ' + key)
            return None
        move_trials = set([int(x.split('_')[2]) for x in video_titles])
        
        csv = pd.read_csv(experiment_data[key]['trial'])
        success_trials = set([int(x) for x, success in enumerate(csv['successful_trials'].to_list()) if success])
        return (success_trials.intersection(move_trials))
    