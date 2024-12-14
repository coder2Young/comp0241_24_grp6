# COMP241 Final Project of Group6

## File Organization

```
final - - Dataset - grp6(Please download from OneDrive)    - task1
        |            |                                         |
        |            |                                       task2
        |            |                                         |
        |            |                                       task3
        |           images(From the original course dataset)
        |            |
        |           masks(From the original course dataset)
        |
        | task1.ipynb
        | task2.py
        | task2c.py
        | task3.py
```

## Run Codes

Before running, download our dataset folder "grp6" from Onedrive and put it under folder "Dataset" according to the file organization above.

Onedrive link:
https://liveuclac-my.sharepoint.com/:f:/g/personal/ucab221_ucl_ac_uk/EnRw7wuXqGdDuqF4BpeyePoBoxPFsOXoCWEOPxeutmoQ6A?e=xyBgoh

After downloading, follow the instruction bellow to run our code.

### Task1 

For task1, run the jupyter notebook cell by cell.

### Task2

For task2a & 2b, set these variables as bellow.

```
# Enable debug mode
debug = True
# Write the output to a video file
write = False
# Set video source: from camera or file
from_camera = False
```

And run the task2.py

* Press "q" to quit.
* Press "c" to capture current center of AO
* Press "s" to save current frame as picture

For task2c, run the task2c.py, the height of AO will be printed in the terminal.

### Task3

For task3, set the variable video_name according to different sub-task

```
video_name = "task3e_top"  # Change the video name here
```

Run the code task3.py, the processing time for each frame will be printed to terminal every 90 frames to demonstrate the real-time processing.

Wait a few minutes for the program to finish running. The SSD variation curve will be displayed on the screen, and the period will be shown both on the curve and in the terminal.
