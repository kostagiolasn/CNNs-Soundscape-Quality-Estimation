"""Generate a Spectrogram image for a given WAV audio sample.

A spectrogram, or sonogram, is a visual representation of the spectrum
of frequencies in a sound.  Horizontal axis represents time, Vertical axis
represents frequency, and color represents amplitude.
"""


import os
import sys
import wave

import pylab


def graph_spectrogram(full_path, wavfile):
    sound_info, frame_rate = get_wav_info(full_path)
    pylab.figure(num=None, figsize=(2.27, 2.27))
    pylab.subplot(111)
    pylab.title('spectrogram of %r' % wavfile)
    pylab.specgram(sound_info, Fs=frame_rate)
    name = os.path.splitext(wavfile)[0]+'.png'
    print name
    pylab.savefig(name)


def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'Int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate


if __name__ == '__main__':
    inputs = sys.argv
    for wavfile in os.listdir(inputs[1]):
		full_path =  inputs[1] + '/' + wavfile
		if wavfile.endswith(".wav"):
			graph_spectrogram(full_path, wavfile)
