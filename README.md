# A Long-term Object Tracker based on LCT2 and fDSST

## Introduction:

This tracker is based on LCT2 and fDSST. With the help of a Long term correlation filter and a well-trained svm, it can find the position of the object even if it was lost.

## Prerequisites:

Windows10

## Installation:

1. Install mingw(x86_64-8.1.0-release-posix-seh-rt_v6-rev0) from https://sourceforge.net/projects/mingw-w64/ , and add its /bin to the environment variable PATH.

2. Download the opencv 4.1.1 source code, build it with the mingw above. You can use cmake for convinience. Or you can download the built opencv from https://github.com/huihut/OpenCV-MinGW-Build . Add its /bin to the environment variable PATH.

## Test:

We use the dataset from OTB as the demo to test our program. To test it, you have to first download the OTB Datasets from http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html .  Then, you can run the program after building it with mingw by typing command like:

>LCT2.exe d:\data_seq\Car4\img\ 1 659 58 50 107 87 car4

Here, the first parameter indicates where the OTB images are saved. The second parameter and the third parameters are the beginning frame and the last frame to track. The following four parameters define the initial rectangle, which is given in this way:

>x y width height

You can use the first rectangle in **groundtruth_rect.txt** . Not that the x and y should minus **one**. The last paramter indicates the name for output text. After running it, you can have a text named as:

>name_ans.txt

as the tracking result. Note that the first rectangle in name_ans.txt is always the groundtruth.