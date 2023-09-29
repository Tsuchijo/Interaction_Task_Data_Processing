## How to Use
After installing follow these steps to make sure the repo is set up correctly
1. Change Paths in config.py to match your systems setup, the local paths should be the same but teamshare paths will be different depending on how your system is setup
    - 'teamshare' is the root of the teamshare path, change it to how teamshare is mounted on your system
    - 'video_output_path' and 'movement_extracted_output_path' can be local or teamshare, also might need to be change based on system setup
2. setup required python environment TODO: make requirements.txt
3. Run Process_Trials.py to process the data the outputs will be:
    - 30 seconds clips of the trials where both broke the beam, both cropped and uncropped located in 'video_output_path'
    - trimmed clips of where there is movement in the frame located in 'movement_extracted_output_path', with the timing of the video in the title given in frames at 25 fps
    - a pickle file stored locally in data called 'experiment_data.pickle' which stores a dict relating the source path of matching days of photometry, trial, and video data to eachother

4. Check the output of the cropped videos, if the perspective transform is off you may need to run the notebook 'Crop_Save_Markers.ipynb' to take a screenshot centered on the interaction area, then delete the folder with the odd videos and then try again