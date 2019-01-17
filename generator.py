'''
TODO:
The problem of the dataset structure is that speaker ID is binded with utterance ID, 
which needs to be fixed.
'''
import os
import sys
import glob
import random
import argparse
import numpy as np
from skimage import io, transform
from skimage.color import rgb2gray
from scipy.io import wavfile
from scipy import signal

class DataGenerator():
    def __init__(self, fft_size, fr, n_frames, source_path, ambient_path, car_path):
        self.fft_size = fft_size
        self.frame_rate = fr
        self.n_frames = n_frames
        self.source = []
        self.ambient = [] # Path of training ambient noise
        self.car = [] # Path of training car noise
        self.noise_gain_range = 5
        self.verbose = False
        
        try:
            with open(source_path, 'r') as s, open(ambient_path, 'r') as a, open(car_path, 'r') as c:
                self.source = s.read().splitlines()
                self.ambient = a.read().splitlines()
                self.car = c.read().splitlines()

        except IOError as e:
            print('Failed as {}'.format(e.strerror))

    '''
    # Old version
    def mix_wav(self, source, ambient, car):
        snr_ambient = random.randint(-2, 10)
        snr_car = random.randint(-2, 10)

        source_amp = sum(abs(source)) / len(source)

        ambient_amp = source_amp / (10 ** (snr_ambient / 20))
        car_amp = source_amp / (10 ** (snr_car / 20))
        ambient_norm = (ambient - ambient.mean()) / ambient.std()
        car_norm = (car - car.mean()) / car.std()

        ambient_scaled = ambient_amp * ambient_norm + ambient.mean()
        car_scaled = car_amp * car_norm + car.mean()

        mixed = source + ambient_scaled + car_scaled
        
        return mixed.astype(np.int16)
    '''
    
    def mix_wav(self, source, ambient, car):
        snr_ambient = random.randint(-5, 10)
        snr_car = random.randint(-5, 10)
        
        source_len, ambient_len, car_len = \
        source.shape[0], ambient.shape[0], car.shape[0]
        
        # Pad wave sequences
        if ambient_len > source_len:
            start_idx = random.randint(0, ambient_len - source_len)
            ambient = ambient[start_idx:start_idx + source_len]
        
        elif ambient_len < source_len:
            front = random.randint(0, source_len - ambient_len - 1)
            rear = source_len - ambient_len - front
            ambient = np.pad(ambient, (front, rear), 'wrap')
            
        if car_len > source_len:
            start_idx = random.randint(0, car_len - source_len)
            car = car[start_idx:start_idx + source_len]
        
        elif car_len < source_len:
            front = random.randint(0, source_len - ambient_len - 1)
            rear = source_len - car_len - front
            car = np.pad(car, (front, rear), 'wrap')
        
        assert source.shape[0] == ambient.shape[0] == car.shape[0]

        source_amp = sum(abs(source)) / len(source)

        ambient_amp = source_amp / (10 ** (snr_ambient / 20))
        car_amp = source_amp / (10 ** (snr_car / 20))
        ambient_norm = (ambient - ambient.mean()) / ambient.std()
        car_norm = (car - car.mean()) / car.std()

        ambient_scaled = ambient_amp * ambient_norm + ambient.mean()
        car_scaled = car_amp * car_norm + car.mean()

        mixed = source + ambient_scaled + car_scaled

        return mixed.astype(np.int16)
        
    def wav2spec(self, wav, sr):
        f, t, Zxx = signal.stft(
                wav, # in librosa, amplitude is devided by 32768
                fs=sr,
                nfft=self.fft_size,
                nperseg=sr // self.frame_rate,
                noverlap=0)
        if self.verbose:
            print('\
                Sample Rate: {}\n\
                n_fft: {}\n\
                nperseg: {}\n\
                noverlap: {}\n'.format(sr, self.fft_size, sr // self.frame_rate, 0))

        spec = np.log10(np.abs(Zxx))
        # print('Spectrogram Shape {}\n'.format(spec.shape))

        return spec
    
    def next_batch(self, batch_size, v_norm=True, a_norm=True, img_size=128):
        N, C, V = [], [], []

        while True:

            for i in range(batch_size):
                # choose an audio
                source_path = random.choice(self.source)
                ambient_path = random.choice(self.ambient)
                car_path = random.choice(self.car)

                # choose five frames
                uttr = source_path.split('/')[-1].split('.')[0]
                all_frame = sorted(glob.glob('dataset/{}/lip/*'.format(uttr)))


                # read waves
                sr_ambient, wav_ambient = wavfile.read(ambient_path)
                sr_car, wav_car = wavfile.read(car_path)
                sr_source, wav_source = wavfile.read(source_path)

                assert sr_ambient == sr_car == sr_source

                sr = sr_source

                del sr_source, sr_ambient, sr_car

                # truncate
                if wav_source.shape[0] // (sr // self.frame_rate) < len(all_frame):
                    len_org = len(all_frame)
                    for _ in range(len_org - len(wav_source) // (sr // self.frame_rate)):
                        del all_frame[-1]

                assert len(all_frame) <= wav_source.shape[0]

                frame_idx = random.randint(0, len(all_frame) - self.n_frames) # video frame id
                
                # mix source, ambient and car noises
                wav_mixed = self.mix_wav(wav_source, wav_ambient, wav_car)
                
                # get spectrogram
                spec_source = self.wav2spec(wav_source, sr)
                spec_noise = self.wav2spec(wav_mixed, sr)

                toggle = 0
                while spec_source.shape[1] > len(all_frame):
                    if toggle == -1:
                        toggle = 0
                    elif toggle == 0:
                        toggle = -1
                    spec_source = np.delete(spec_source, toggle, 1)
                    spec_noise = np.delete(spec_noise, toggle, 1)
                
                assert spec_source.shape[1] == spec_noise.shape[1] == len(all_frame)
                # print('source:\t{}\nnoise:\t{}\nimage:\t{}\n'.format(spec_source.shape[1], spec_noise.shape[1], len(all_frame)))

                
                # truncate spectrogram
                spec_source = spec_source[:, frame_idx:frame_idx + self.n_frames]
                spec_noise = spec_noise[:, frame_idx:frame_idx + self.n_frames]
                
                # get normalized image
                img_seq = []
                for fname in all_frame[frame_idx:frame_idx + self.n_frames]:
                    img = transform.resize(
                            io.imread(fname), 
                            (img_size, img_size)) 
                            #anti_aliasing=True)

                    if v_norm:
                        img = (img - img.mean()) / img.std()
                        
                    img_seq.append(rgb2gray(img))
                
                
                N.append(spec_noise)
                C.append(spec_source)
                V.append(np.dstack(img_seq))

                # Test output audio
                # wavfile.write('out.wav', sr, mixed)

            yield N, C, V

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '-s', type=str)
    parser.add_argument('--ambient', '-an', type=str)
    parser.add_argument('--car', '-cn', type=str)
    args =  parser.parse_args()

    fft_size = 512
    frame_rate = 50
    n_frames = 5

    d_gen = DataGenerator(
            fft_size, 
            frame_rate, 
            n_frames,
            args.source, 
            args.ambient,
            args.car)
