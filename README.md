#Guitar Sound Classifier

This project contains a dataset and model for classifying guitar sounds as either acoustic or electric based on spectrograms and MFCC features.
Dataset
The dataset loads .wav files of acoustic and electric guitar notes from two folders:
acoustic_note_*.wav
note_*.wav (electric)
For each .wav file, the following features are extracted:
Spectrogram: The short-time Fourier transform is used to compute the spectrogram.
Mel spectrogram: A mel spectrogram is computed using a mel filterbank.
MFCC: Mel frequency cepstral coefficients (MFCCs) are extracted.
These 3 features are stacked to form a 3-channel "image" representing the guitar sound.
The label (acoustic or electric) is determined based on the file path.
Model
The model uses the following architecture:
3x64 convolutional layer
LeakyReLU activation
2x2 max pooling
128x128 convolutional layer
LeakyReLU activation
2x2 max pooling
Fully connected layer with 128 nodes

