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
import pickle
import utilities.helper_functions as hf
import config
import cv2
import tdt

def write_successful_trials(video, df, offset, path):
    reference_path = [config.marker_path + dir for dir in os.listdir(config.marker_path)]
    # iterate through df and for each successful trial get the start time in seconds
    reference_path.sort(reverse=True)
    date_format = "%H:%M:%S.%f"
    reference_image = cv2.imread(reference_path[0])
    first_frame = video.read()[1]
    frame_size = first_frame.shape
    transformation, num_match, ref_path = hf.find_matching_transformation(first_frame, reference_path)
    # create video writer
    fps = video.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    for index, row in df.iterrows():
        if row['successful_trials']:
            transformation_new, num_match_new, ref_path = hf.find_matching_transformation(video.read()[1], reference_path)
            if num_match_new > num_match and num_match_new > 15:
                transformation = transformation_new
                num_match = num_match_new
                reference_image = cv2.imread(ref_path)
            start_time = (datetime.datetime.strptime(row['time_since_start'][7:], date_format) - datetime.datetime.strptime('00:00:00.000000', date_format)).total_seconds() - offset
            end_time = start_time + 30
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
            out_full_vid = cv2.VideoWriter(path + '_trial_' +str(index) + '_full_vid.mp4', fourcc, fps, (frame_size[1], frame_size[0]))

            while True:
                success, image = video.read()
                if success == False or float(video.get(cv2.CAP_PROP_POS_FRAMES))/fps > end_time:
                    break
                # add a countdown to the video 
                try:
                    transformed = cv2.warpPerspective(image, transformation, (reference_image.shape[1], reference_image.shape[0]))
                except:
                    print(transformation)
                    exit()
                out.write(transformed)
                out_full_vid.write(image)
            out.release()
            out_full_vid.release()


## Runs through all the raw data files and generates a csv file for each trial with information on successes
def generate_csv_files():
    path = config.remote_trial_path
    processed_data_path = config.successful_trial_path
    files = os.listdir(path)
    files = [path + file for file in files]

    # load a dict of files keyed with the file name
    log_data = {file.split('/')[-1].split('.')[0]: pd.read_csv(file) for file in files}
    # get array of file names 
    file_names = list(log_data.keys())
    file_names.sort()
    successful_trials_by_day = dict()
    for file in file_names:
        df = log_data[file]
        matches = [ re.match('((M[1-8])|(Toy))_?[L | R]?_beambreak_time in s', name) for name in df.columns.to_list()]
        L, R = None, None
        for match in matches:
            if match is not None:
                if L is None:
                    L = match.string
                else:
                    R = match.string
        
        # get the time at the start of the trial
        try:
            start_time = hf.process_time_string(file.split('_')[0])
        except:
            print('Error processing time string for ' + file)
            continue
        # get the datetime of each cue
        cue_times = df['cue_time in s'].apply(lambda x : hf.process_time_string(x))
        # subtract the start time from each cue time
        time_since_start = cue_times.apply(lambda x : x - start_time)

           
        # Get the names of each mouse
        try:
            mouse_name_L = L.split('_')[0]
            mouse_name_R = R.split('_')[0]
        except:
            print('Error: Mouse names not found for ' + file)
            continue
        # subtract the 2nd collumn from the 4th abd 7th collumns
        mouse_L_break_time = df.apply(lambda x : hf.subtract_time_series(x, 'cue_time in s', L), axis=1)
        mouse_R_break_time = df.apply(lambda x : hf.subtract_time_series(x, 'cue_time in s', R), axis=1)

        # get the time the reward was given for each mouse
        try:
            mouse_L_reward_time = df.apply(lambda x : hf.subtract_time_series(x, 'cue_time in s', mouse_name_L + '_L_reward time in s'), axis=1)
            mouse_R_reward_time = df.apply(lambda x : hf.subtract_time_series(x, 'cue_time in s', mouse_name_R + '_R_reward time in s'), axis=1)
        except:
            mouse_L_reward_time = df.apply(lambda x : hf.subtract_time_series(x, 'cue_time in s', mouse_name_L + '_reward time in s'), axis=1)
            mouse_R_reward_time = df.apply(lambda x : hf.subtract_time_series(x, 'cue_time in s', mouse_name_R + '_reward time in s'), axis=1)

        # replace all negative values with None 
        mouse_L_reward_time = mouse_L_reward_time.apply(lambda x : None if x is None or x < 0 else x)
        mouse_R_reward_time = mouse_R_reward_time.apply(lambda x : None if x is None or x < 0  else x)

        # run through both to see when both broke the beam, then record as a success for the day
        sucessful_trials = []
        # absolute difference between mouse L and mouse R break times
        time_between_breaks = []
        try:
            for i in range(len(mouse_L_break_time)):
                if not math.isnan(mouse_L_break_time[i]) and not math.isnan(mouse_R_break_time[i]) :
                    sucessful_trials.append(True)
                    time_between_breaks.append(abs(mouse_L_break_time[i] - mouse_R_break_time[i]))
                else:
                    sucessful_trials.append(False)
                    time_between_breaks.append(None)
        except:
            continue
        successful_trials_by_day[file] = {
            'names' : mouse_name_L + mouse_name_R,
            'L': mouse_L_break_time,
            'R': mouse_R_break_time,
            'successful_trials': sucessful_trials,
            'time_between_breaks': time_between_breaks,
            'time_since_start': time_since_start,
            'cue_times': cue_times,
            'mouse_L_reward_time': mouse_L_reward_time,
            'mouse_R_reward_time': mouse_R_reward_time}
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
        write_successful_trials(video, df, csv_vid_match[csv_name]['delta_seconds'], config.video_output_path + csv_name.split('_')[0] + '/')

def process_photometry_data():
    date_format = "%Y-%m-%d %H:%M:%S.%f"
    # create dict to hold correspondence between photometry data, trial data, and video data keyed by day
    experiment_data = dict()
    for day in os.listdir(config.remote_photometry_path):
        trial_paths = [config.remote_photometry_path + day + '/' + x for x in os.listdir(config.remote_photometry_path + day) if re.match('M[1-8]_F1_M[1-8]_F2', x)]
        date = day.split('-')[-1]
        # Search through all trials recorded and find ones matching the day of photometry recording
        matching_trials = [x for x in os.listdir(config.successful_trial_path) if x[2:8] == date]
        valid_trials = []
        mice_ids = []
        # remove trials with less than 10 trials
        for trial in matching_trials:
            if pd.read_csv(config.successful_trial_path + trial).shape[0] > 10:
                trial_csv = pd.read_csv(config.successful_trial_path + trial)
                mice_ids.append(trial_csv['names'][0])
                valid_trials.append(trial)
            
        matching_trials = valid_trials
        for trial in trial_paths:
            data = None
            # find the two mouse ids by regex matching to M1-8
            mouse_id = re.findall('M[1-8]', trial)
            mouse_id = mouse_id[0]+mouse_id[1]
            # get all indices of matching mouse ids
            matching_indices = [i for i, x in enumerate(mice_ids) if x == mouse_id]
            data = tdt.read_block(trial, t2=1)
            if data is None:
                print('No data found for ' + trial)
                continue
            print('Processing ' + trial)
            start_time = data.info.start_date
            # iterate through all matching trials and find the one with the closest start time
            closest_trial = None
            closest_time_delta = 10000000
            for i in matching_indices:
                match_trial = pd.read_csv(config.successful_trial_path + matching_trials[i])
                trial_start_time = datetime.datetime.strptime(match_trial['cue_times'][0], date_format)
                if abs((start_time - trial_start_time).total_seconds()) < closest_time_delta:
                    closest_time_delta = abs(start_time - trial_start_time).total_seconds()
                    closest_trial = matching_trials[i]
                    closest_trial_start_time = trial_start_time

            # find the video that matches closest trial
            video_times = [hf.process_time_string(name.split('_')[0]) for name in os.listdir(config.remote_video_path)]
            video_paths = [name for name in os.listdir(config.remote_video_path)]
            # iterate through all videos and find the one with the closest start time
            closest_video = None
            closest_time_delta = 10000000
            for i, video_time in enumerate(video_times):
                if abs((closest_trial_start_time - video_time).total_seconds()) <  abs(closest_time_delta):
                    closest_time_delta = abs((closest_trial_start_time - video_time).total_seconds())
                    closest_video = video_paths[i]
            # append trial data, video data, and photometry data to the experiment data dict
            # check if the day is already in the dict

            if str(start_time.date()) in experiment_data:
                experiment_data[str(start_time.date()) + '_2'] = {
                    'trial': config.successful_trial_path + closest_trial,
                    'video': config.remote_video_path + closest_video,
                    'photometry': trial
                }
            else:
                experiment_data[str(start_time.date())] = {
                    'trial': config.successful_trial_path + closest_trial,
                    'video': config.remote_video_path + closest_video,
                    'photometry': trial
                }
    # save experiment data to a pickle file
    with open(config.PROJECT_ROOT + '/data/experiment_data.pickle', 'wb') as handle:
        pickle.dump(experiment_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

## Imports array of videos and names of videos from a folder
# @param path: path to folder containing videos
# Returns a list of videos and a list of names
def load_videos(path, regex = None):
    videos = []
    names = []
    for file in os.listdir(path):
        if file.endswith('.mp4'):
            if regex is not None:
                if not re.search(regex, file):
                    continue
            names.append(path.split('/')[-1] + file.split('.')[0])
            video_capture = cv2.VideoCapture(path + '/' + file)
            video = []
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break
                # convert to grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                video.append(frame)
            videos.append(np.array(video))
    return videos, names

## Take each a video and then subtract each frame from the next frame to get the next movement
def calculate_movement(video):
    movement = []
    for i in range(len(video) - 1):
        movement.append(video[i + 1].astype('int16') - video[i].astype('int16'))
    return np.abs((np.array(movement) + 256) // 2).astype('uint8')

def extract_movement_frames(video):
    movement = []
    for i in range(len(video) - 1):
        movement.append(np.abs(np.sum(video[i + 1].astype('int16') - video[i].astype('int16'))))
    # take a moving average of the movement
    try:
        movement = np.convolve(movement, np.ones(100) / 100, mode='same')
    except (ValueError):
        movement = np.zeros(len(movement))
    return movement

def extract_interactions():
    video_paths = [config.video_output_path[:-1] + '/' + trial for trial in os.listdir(config.video_output_path[:-1] + '/') ]
    # iterate through all the videos, extract interactions, and then save the video of the interactions to a folder
    for path in video_paths:
        videos, names = load_videos(path, regex = '^((?!_full_vid).)*$')
        for video, name in zip(videos, names):
            movement = extract_movement_frames(video)
            if len(movement) < 2:
                continue
            if np.max(movement) > 1000:
                # find all frames where the movement rises above 1000
                movement_start = rising_edge(movement)
                # find all frames where the movement falls below 1000
                movement_end = falling_edge(movement)
                # # find the first frame where the movement is greater than 1000
                # start = np.where(movement > 1000)[0][0]
                # # find the last frame where the movement is greater than 1000
                # end = np.where(movement > 1000)[0][-1]
                for start, end in zip(movement_start, movement_end):
                    movemnent_extracted_video = video[int(start):int(end)]
                    # save the video to the movement_extracted folder
                    hf.save_video(movemnent_extracted_video, config.movement_extracted_output_path[:-1] + '/' + name.split('_')[0] + '/' + name + '_start-' + str(start) + '_end-' + str(end) + '.mp4')
                    hf.save_video(video, config.movement_extracted_output_path[:-1] + '/' + name.split('_')[0] + '/' + name + '_full_length' + '.mp4')

def rising_edge(data, thresh=1000):
    sign = data >= thresh
    pos = np.where(np.convolve(sign, [1, -1]) == 1)
    return pos[0]

def falling_edge(data, thresh=1000):
    sign = data >= thresh
    pos = np.where(np.convolve(sign, [1, -1]) == -1)
    return pos[0]

## Main function
def main():
    generate_csv_files()
    segment_successfull_videos()
    process_photometry_data()
    extract_interactions()



main()