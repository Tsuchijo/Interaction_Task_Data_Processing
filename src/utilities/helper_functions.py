import os 
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import datetime
import re
import math
import cv2

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
            transformation, num_match = find_area_and_transform(scene, reference)
            transformations.append(transformation)
            num_matches.append(num_match)
        return transformations[num_matches.index(max(num_matches))], max(num_matches)