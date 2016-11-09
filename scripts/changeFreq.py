"""
Change the sampling frequency [Fs] of all the
wav files included in a file to 16000.

The user should provide the folder name in which the
wavs to be converted should be present.
"""

import os
import wave
import sys

import pylab

if __name__ == '__main__':
	#directory = '/home/nikos/Desktop/NCSR Demokritos/Soundscape Quality Estimation using Deep Learning/Database/soundscapeanalysis/wav'
	directory = sys.argv[1]
	for filename in os.listdir(directory):
		if filename.endswith(".wav"):
			file_before = directory+'/'+filename
			file_after = directory+'/1600'+filename
			avconv -i file_before -ar 16000 -ac 1 file_after
