# CNNs - Soundscape Quality Estimation

##Synopsis

This project describes an approach on Soundscape Quality Estimation. To the best of our
knowledge, the proposed method provides a novel approach to this problem by introducing
multi-label classification, in order to assess the quality of a soundscape (i.e. an audio
landscape) based on the qualitative evaluation of its individual sound elements. To achieve
this task we employ a Deep Convolutional Neural Network (_CNN_) which operates on 
pseudocolored RGB frequency-images, which represent audio segments.

**The repository consists of the following modules:**
  * Audio segmentation using the [PyAudio](https://github.com/tyiannak/pyAudioAnalysis.git) 
analysis library
  * CNN training using the [Lasagne Deep-Learning Framework](https://github.com/Lasagne/Lasagne).
  * Audio classification using:
    * CNNs
    * CNNs using an ImageNet pre-trained model to initialize the neuron values
    * CNNs using data augmentation
  * An audio dataset consisting of 30 second multi-label annotated instances of soundscape
  auditory data. At this point the data are available in the form of spectrograms. (_to be added_)
  The instances are annotated as e.g. {vehicles, voice_(children), rain} or {sirens, shouting}.
  
  ##Installation
  - Dependenices
  1. [PyAudio](https://github.com/tyiannak/pyAudioAnalysis.git) 
  2. [Lasagne Deep-Learning Framework](https://github.com/Lasagne/Lasagne)
  
   _* Installation instructions offered in detail on the above links_
   
   #### **Data Preparation**
   1. Change the frequency of the audio files into 16000 Hz using _changeFreq.py_
   2. Convert your audio files into pseudocolored RGB or grayscale spectrogram images using _generateSpectrograms.py_
   Data should be pseudo-colored RGB spectrogram images of size 227x227 as shown in Fig1 :
    <img src="https://github.com/MikeMpapa/CNNs-Speech-Music-Discrimination/blob/master/sampleIMg.png" width="227" height="227">
    <figcaption>Fig1. - Sample RGB Spectrogram</figcaption>
   3. Distribute the generated spectrograms to their respective classes using _fixDataset.sh_
