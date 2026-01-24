## Lab_02

# Title: Basic Sound Processing and Manipulation in Python

## 1. Objective

The objective of this laboratory experiment is to understand the fundamentals of digital sound processing using Python. The experiment focuses on loading audio files, visualizing waveforms, performing basic manipulations, and saving the processed audio.

## 2. Introduction

Sound is a form of analog signal that represents variations in air pressure over time. In digital systems, sound is represented as discrete samples obtained through a process called sampling. Sound processing involves analyzing and modifying these digital signals to achieve desired effects such as noise reduction, amplification, trimming, or filtering.

Python provides powerful libraries such as **NumPy**, **SciPy**, **Librosa**, and **Matplotlib** that make audio processing simple and efficient. This lab demonstrates basic sound processing operations using these tools.

## 3. Tools and Libraries Used

* Python 3.x
* NumPy – numerical computations
* Librosa – audio loading and processing
* SoundFile – saving audio files
* Matplotlib – waveform visualization


## Lab Folder Structure 
 lab_02/
│── Lab_with_Sound (2).ipynb
│── generated_5sec_drum.wav
│── guitar_sound.wav
│── harmonious_sound.wav
│── readme.md


## 4. Theory

### 4.1 Digital Audio Concepts

* **Sampling Rate**: Number of samples per second (Hz).
* **Amplitude**: Loudness of the sound signal.
* **Waveform**: Graphical representation of sound amplitude over time.
* **Mono vs Stereo**: Mono has one channel; stereo has two channels.

### 4.2 Sound Processing Operations

* Loading audio files
* Visualizing waveforms
* Changing volume
* Trimming audio
* Saving modified audio  

## output:

<audio controls src="harmonium_sound.wav" title="Title"></audio> 
<audio controls src="guitar_sound.wav" title="Title"></audio>
<audio controls src="generated_5sec_drum.wav" title="Title"></audio>

## 6. Result

The audio file was successfully loaded and visualized. Volume amplification and trimming operations were performed correctly. The processed audio was saved as a new WAV file without distortion.

## 7. Discussion

This experiment demonstrates that Python is an effective tool for basic sound processing tasks. Libraries like Librosa simplify complex audio operations. Care must be taken while amplifying audio, as excessive gain may cause clipping and distortion.

## 8. Conclusion

Basic sound processing and manipulation can be efficiently performed using Python. Understanding these fundamentals is essential for advanced applications such as speech recognition, music analysis, and audio-based machine learning systems.












