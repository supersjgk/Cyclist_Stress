import os
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample
from sklearn.preprocessing import MinMaxScaler
import wave
import torch
import noisereduce as nr

class DataPreprocessor:
    def __init__(self, folderpath):
        """
            Data folder map:
            /Data
            ├── Participant 1
            |   ├── Session 1
            │   │   ├── bike_data.csv - IMU data at 400 Hz
            |   |   ├── bike_audio_timestamp.csv - Labels
            |   |   └── bike_audio_timestamp.pcm - Audio data at 44100 Hz
            |   ├── Session 2 .....
            |   └── Session 3 ..... so on   
            |   
            ├── Participant 2 ..... so on
            |
        """
        self.folderpath = folderpath # folder containing the data with above heirarchy
        self.audio_sample_rate = 44100
        self.imu_sample_rate = 400
        self.session_length = 600 # Recording time of one session in seconds
        self.label_window = 5 # label is collected every 5 seconds
        self.label_count = self.session_length // self.label_window
        self.downsample_size = 20 # number of time-steps corresponding to one label
    
    def _pcm_to_wav_denoise(self, pcm_path, wav_path):
        """
            Helper function to convert PCM Audio to WAV file and denoise it.
            noisereduce package is used for non-stationary noise reduction. 
            https://pypi.org/project/noisereduce/
            https://github.com/timsainb/noisereduce/blob/master/noisereduce/spectralgate/nonstationary.py
        """
        pcm = np.fromfile(pcm_path,dtype=np.int16)
        with wave.open(wav_path, 'w') as wav:
            wav.setnchannels(1) # mono channel
            wav.setsampwidth(2) # 16-bit audio
            wav.setframerate(self.audio_sample_rate)
            wav.writeframes(pcm.tobytes())
            
        samplerate, data = wavfile.read(wav_path
        reduced_noise = nr.reduce_noise(y=data, sr=samplerate, stationary=False, use_torch=True, n_jobs=1)
        wavfile.write(wav_path, samplerate, reduced_noise)

    def pcm_to_wave_denoise(self):
        """
           Call the helper function to denoise Audio, replace pcm with wav 
        """
        for root, dirs, files in os.walk(self.folderpath):
            for file in files:
                if file.endswith('.pcm'):
                    pcm_file_path = os.path.join(root, file)
                    wav_file_path = os.path.splitext(pcm_file_path)[0] + '.wav'
                    try:
                        self._pcm_to_wav_denoise(pcm_file_path, wav_file_path)
                        os.remove(pcm_file_path)
                    except:
                        print(file)
                        break

    def get_sessions(self):
        """
            Group all files of a session of a participant for easier access
        """
        audio_files = []
        imu_files = []
        labels = []
        for root, dirs, files in os.walk(self.folderpath):
            for file in files:
                if file.endswith(".wav"):
                    audio_files.append(os.path.join(root, file))
                elif file.endswith("data.csv"):
                    imu_files.append(os.path.join(root, file))
                else:
                    labels.append(os.path.join(root, file))
                
        return list(zip(audio_files, imu_files, labels))
    
    def _preprocess_and_downsample(self, audio_file, imu_file, label_file):
        """
            This Helper function performs the following tasks: remove null/extra values, split data into samples, and Downsample
        """
        # Audio
        samplerate, audiodata = wavfile.read(audio_file)
        audiodata = np.trim_zeros(audiodata, 'f') # trim leading zeros
        audiodata = audiodata[:self.label_window * self.audio_sample_rate * self.label_count] # remove trailing data if any
        audiodata = audiodata.reshape((self.label_count, self.label_window * self.audio_sample_rate)) # split data into samples
        """
            16 bit audio has amplitudes between -2^16 and +2^16. To normalize, uncomment the following code.
            # scaler = MinMaxScaler(feature_range=(-1, 1))
            # temp = np.zeros_like(audiodata,dtype=np.float32)
            # for i in range(self.label_count):
            #     temp[i,:] = scaler.fit_transform(data[i,:].reshape(-1, 1)).flatten()
            audiodata = temp
        """
        # Downsampling Audio
        temp = np.zeros(shape=(self.label_count, self.downsample_size), dtype=np.float32)
        for i in range(self.label_count):
            temp[i] = resample(audiodata[i], self.downsample_size)
        audiodata = temp.reshape(self.label_count, self.downsample_size, 1) # shape = (num_samples x time-steps x amplitude)

        #-----------------------------------------------------------------
        # IMU
        imu_df = pd.read_csv(imu_file, usecols=['ACC X', 'ACC Y', 'ACC Z', 'GYRO X', 'GYRO Y', 'GYRO Z']) # 6 sensor readings/columns
        imu_df = imu_df.dropna()
        """
            Optional normalizing:
            scaler = MinMaxScaler(feature_range=(-1,1))
            imu_df = pd.DataFrame(scaler.fit_transform(imu_df), columns=imu_df.columns)
        """
        imu_df = imu_df[:self.label_count * self.imu_sample_rate * self.label_window]
        imu_df = imu_df.to_numpy()
        imu_df = imu_df.reshape(self.label_count, -1, 6) # split data into samples
        
        # Downsampling IMU
        temp = np.zeros(shape=(self.label_count, self.downsample_size, 6),dtype=np.float32)
        for i in range(temp.shape[0]):
            for j in range(temp.shape[2]):
                temp[i,:,j] = resample(imu_df[i,:,j], self.downsample_size)
        imudata = temp # shape = (num_samples x time-steps x 6 imu readings)

        #-----------------------------------------------------------------
        # Labels
        labels = pd.read_csv(label_file,usecols=['labels'])
        labels = labels.values.reshape(-1)

        return audiodata, imudata, labels

    def preprocess_and_downsample(self):
        """
            Calling the helper function on the data
        """
        sessions = self.get_sessions()
        data, labels = [], []
        for session in sessions:
            try:
                audiodata, imudata, labels = self._preprocess_and_downsample(session[0], session[1], session[2])
                ai = np.concatenate((audiodata, imudata), axis=2)
                data.append(ai)
                labels.append(l)
            except:
                print(files) # get the file with error
                break
        data, labels = np.concatenate(data, axis=0), np.concatenate(labels, axis=0)
        return data, labels

    def preprocessing_pipeline(self):
        self.pcm_to_wave_denoise()
        data, labels = self.preprocess_and_downsample()
        return data, labels
        