#!/home/murph_4090ws/Documents/Arjun_data/.conda/bin/python

# Import libraries
import os 
import sys
sys.path.append(os.path.abspath('../'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import re
import math
import argparse
import utilities.helper_functions as hf
import config
import cv2


def write_successful_trials(video, df, offset, path, reference_path):
    # iterate through df and for each successfull trial get the start time in seconds
    reference_image = cv2.imread(reference_path[0])
    transformation, num_match = hf.find_matching_transformation(video.read()[1], reference_path)
    # create video writer
    fps = video.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    for index, row in df.iterrows():
        if row['successful_trials']:
            transformation_new, num_match_new = hf.find_matching_transformation(video.read()[1], reference_path)
            if num_match_new > num_match or num_match_new > 30:
                transformation = transformation_new
                num_match = num_match_new
            start_time_str = row['time_since_start'].split(' ')[2]
            start_time = start_time_str.split(':')
            start_time = float(start_time[0])*3600 + float(start_time[1])*60 + float(start_time[2]) + offset
            end_time = start_time + 15
            # set the video to the start time
            fps = video.get(cv2.CAP_PROP_FPS)
            current_time = float(video.get(cv2.CAP_PROP_POS_FRAMES)) / fps
            time_offset = start_time - current_time
            for i in range(int(time_offset * video.get(cv2.CAP_PROP_FPS))):
                video.read()

            # find the mice names
            name1 = row['names'][:2]
            name2 = row['names'][2:]
            # write the video to a new file until the end time
            # check if folder exists, if not create it
            if not os.path.exists(path):
                os.makedirs(path)
            out = cv2.VideoWriter(path + '_trial_' +str(index) + '.mp4', fourcc, fps, (reference_image.shape[1], reference_image.shape[0]))

            while True:
                success, image = video.read()
                if success == False or float(video.get(cv2.CAP_PROP_POS_FRAMES))/fps > end_time:
                    break
                # add a countdown to the video 
                transformed = cv2.warpPerspective(image, transformation, (reference_image.shape[1], reference_image.shape[0]))
                out.write(transformed)
                if cv2.waitKey(12) & 0xFF == ord('q'):
                    break
            out.release()


## Runs through all the raw data files and generates a csv file for each trial with information on successes
def generate_csv_files():
    path = config.remote_trial_path
    processed_data_path = config.successful_trial_path
    files = os.listdir(path)
    files = [path + '/' + file for file in files]
    
    # load a dict of files keyed with the file name
    log_data = {file.split('/')[-1].split('.')[0]: pd.read_csv(file) for file in files}
    # get array of file names 
    file_names = list(log_data.keys())
    file_names.sort()

    mouse_name_list = ['M' + str(i) for i in range(1,9)]
    mouse_break_times = dict(zip(mouse_name_list, np.empty((8, 2))))
    successful_trials_by_day = dict()

    M1_M2 = []
    M3_M4 = []
    M5_M6 = []
    M7_M8 = []
    for file in file_names:
        df = log_data[file]
        matches = [ re.match('(M)[1-8]_[L | R]_beambreak_time in s', name) for name in df.columns.to_list()]
        for match in matches:
            if match is not None:
                if match.string.split('_')[1] == 'R':
                    R = match.string
                else:
                    L = match.string
        # get the time at the start of the trial
        start_time = hf.process_time_string(file.split('_')[0])
        # get the datetime of each cue
        cue_times = df['cue_time in s'].apply(lambda x : hf.process_time_string(x))
        # subtract the start time from each cue time
        time_since_start = cue_times.apply(lambda x : x - start_time)
        # Get the names of each mouse
        mouse_name_L = L.split('_')[0]
        mouse_name_R = R.split('_')[0]
        # subtract the 2nd collumn from the 4th abd 7th collumns
        mouse_L_break_time = df.apply(lambda x : hf.subtract_time_series(x, 'cue_time in s', L), axis=1)
        mouse_R_break_time = df.apply(lambda x : hf.subtract_time_series(x, 'cue_time in s', R), axis=1)

        if min(mouse_L_break_time) < 0 or min(mouse_R_break_time) < 0:
            print(file)

        # run through both to see when both broke the beam, then record as a success for the day
        sucessful_trials = []
        # absolute difference between mouse L and mouse R break times
        time_between_breaks = []
        for i in range(len(mouse_L_break_time)):
            if not math.isnan(mouse_L_break_time[i]) and not math.isnan(mouse_R_break_time[i]) :
                sucessful_trials.append(True)
                time_between_breaks.append(abs(mouse_L_break_time[i] - mouse_R_break_time[i]))
            else:
                sucessful_trials.append(False)
                time_between_breaks.append(None)
        successful_trials_by_day[file] = {
            'names' : mouse_name_L + mouse_name_R,
            'L': mouse_L_break_time,
            'R': mouse_R_break_time,
            'successful_trials': sucessful_trials,
            'time_between_breaks': time_between_breaks,
            'time_since_start': time_since_start,
            'cue_times': cue_times}
        if mouse_name_L == 'M1' or mouse_name_L == 'M2':
            M1_M2.append(file)
        elif mouse_name_L == 'M3' or mouse_name_L == 'M4':
            M3_M4.append(file)
        elif mouse_name_L == 'M5' or mouse_name_L == 'M6':
            M5_M6.append(file)
        elif mouse_name_L == 'M7' or mouse_name_L == 'M8':
            M7_M8.append(file)
        # convert successful trials by day to a dataframe
        successful_trials_df = pd.DataFrame(successful_trials_by_day[file])
        # save the dataframe to a csv file
        successful_trials_df.to_csv(processed_data_path + file + '.csv')

## For any session that hasnt been processsed go through and extract all clips of successful trials
def segment_successfull_videos():
    video_path = config.remote_video_path
    processed_data_path = config.successful_trial_path
    reference_path = [config.marker_path + dir for dir in os.listdir(config.marker_path)]
    # get list of video names from the folder of processed CSVs
    csv_times = [ hf.process_time_string(name.split('_')[0]) for name in os.listdir(processed_data_path)]
    csv_paths = [name for name in os.listdir(processed_data_path)]
    # iterate through each csv and find the closest video name
    video_times = [hf.process_time_string(name.split('_')[0]) for name in os.listdir(video_path)]
    video_paths = [name for name in os.listdir(video_path)]
    csv_vid_match = dict()
    for csv_time in csv_times:
        closest_time = video_times[0]
        closest_path = video_paths[0]
        for i, video_time in enumerate(video_times):
            if abs((csv_time - video_time).total_seconds()) <  abs((csv_time - closest_time).total_seconds()):
                closest_time = video_time
                closest_path = video_paths[i]
        total_delta = (csv_time - closest_time).total_seconds()
        if total_delta < 10:
            csv_vid_match[csv_paths[csv_times.index(csv_time)]] = {
                'video_name': closest_path,
                'delta_seconds': total_delta
            }

    significant_trials = []
    for csv_name in csv_vid_match.keys():
        df = pd.read_csv(processed_data_path + '/' + csv_name)
        if len(df['successful_trials']) >= 10:
            significant_trials.append(csv_name)
    significant_trials.sort()
    significant_trials.reverse()
    for csv_name in significant_trials:
        # check if folder for trial exists, if so skip
        if os.path.exists(config.video_output_path + csv_name.split('_')[0] + '/'):
            continue
        video = cv2.VideoCapture(video_path + csv_vid_match[csv_name]['video_name'])
        df = pd.read_csv(processed_data_path + '/' + csv_name)
        if len(df['successful_trials']) < 10:
            continue
        write_successful_trials(video, df, csv_vid_match[csv_name]['delta_seconds'], config.video_output_path + csv_name.split('_')[0] + '/', reference_path)


## Main function
def main():
    generate_csv_files()
    segment_successfull_videos()


if __name__ == 'main':
    main()