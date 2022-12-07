# ECGprocessing

## Description

**ECG-data_npy**: 50 ECG preprocessed signals (time series with size of 1x20000);

**Filters.py**: adaptive filters;

**Application.py**: SNR and MSE calculation,non-stationary noise addition, R-R period and heartrate analysis; 

**GUI.py**: integrated user interface; 

**tt1**: GUI logo;  

## Overview

Run GUI.py

![image](https://user-images.githubusercontent.com/89956877/206072217-ab38c6e2-1fa3-45d3-9fc2-c64d0b34e533.png)

PS:

a. On the left is the user control panel, the right is the ECG signal display (from top to bottom is original, noised, filtered & analyzed);

b. You can select the ECG file to load;

c. You can modify non-stationary noise and filters;

d. You can locate the R-wave, calculate the R-wave period and the heart rate;

