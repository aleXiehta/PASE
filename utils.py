import numpy as np
import librosa
import random
from scipy import signal
from functools import partial
from multiprocessing import Pool

eps = np.finfo(float).eps

class DataLoader():
    def __init__(self, fft_size, hop_length, win_length, sr, tr=16, norm=True, n_worker=8):
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.win_length = win_length
        self.sr = sr
        self.norm = norm
        self.s_train, self.n_train, s_test, n_test = [], [], [], []
        self.n_worker = n_worker
        self.truncate_size = tr

        try:
            with open('data/data_lists/s_train.txt', 'r') as f_s_train,\
                 open('data/data_lists/n_train.txt', 'r') as f_n_train,\
                 open('data/data_lists/s_test.txt', 'r') as f_s_test,\
                 open('data/data_lists/n_test.txt', 'r') as f_n_test:
                self.s_train = f_s_train.read().splitlines()
                self.n_train = f_n_train.read().splitlines()
                self.s_test = f_s_test.read().splitlines()
                self.n_test = f_n_test.read().splitlines()

        except IOError as e:
            print('Faild on {}'.format(e.strerror))


    def mix_wav(self, source, noise):
        snr = 5 * random.randint(-1, 2)
        source_len, noise_len = source.shape[0], noise.shape[0]
        if noise_len > source_len:
            start_idx = random.randint(0, noise_len - source_len)
            noise = noise[start_idx:start_idx + source_len]

        elif noise_len < source_len:
            front = random.randint(0, source_len - noise_len - 1)
            rear = source_len - noise_len - front
            noise = np.pad(noise, (front, rear), 'wrap')

        assert source.shape[0] == noise.shape[0]

        # mix with certain SNR
        source_amplitude = sum(abs(source)) / source.shape[0]
        noise_amplitude = source_amplitude / (10 ** (snr / 20))
        noise_norm = (noise - noise.mean()) / (noise.std() + eps)
        noise_scaled = noise_amplitude * noise_norm + noise.mean()

        mixed = source + noise_scaled

        return mixed

    def normalize(self, spec):
        mean = np.mean(spec, axis=1).reshape(self.fft_size // 2 + 1, 1)
        std = np.std(spec, axis=1).reshape(self.fft_size // 2 + 1, 1)
        spec_norm = (spec - mean) / (std + eps)

        return spec_norm

    def wav2spec(self, wavfile):
        spec = librosa.stft(
                wavfile,
                n_fft=self.fft_size,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=signal.hann)

        # real, imag = np.log10((spec.real + eps) ** 2), np.log10((spec.imag + eps) ** 2)
        real, imag = spec.real, spec.imag

        if self.norm:
            real = self.normalize(real)
            imag = self.normalize(imag)

        return real, imag
    
    def spec2wav(self, real, imag, length):
        # real = 10 ** (real / 2) - eps
        # imag = 10 ** (imag / 2) - eps
        wavfile = librosa.istft(
                real + 1j * imag,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=signal.hann)
        wavfile = librosa.util.fix_length(wavfile, length, mode='edge')
        wavfile = wavfile / np.max(np.abs(wavfile))

        return wavfile, self.sr

    def load_truncate(self, clean, noisy):
        assert len(clean) == len(noisy)

        noisy_real, noisy_imag = self.wav2spec(noisy)
        clean_real, clean_imag = self.wav2spec(clean)

        truncate_size = self.truncate_size
        start_idx = random.randint(0, clean_real.shape[1] - truncate_size)

        noisy_real, noisy_imag = \
                noisy_real[:, start_idx:start_idx + truncate_size], \
                noisy_imag[:, start_idx:start_idx + truncate_size]

        clean_real, clean_imag = \
                clean_real[:, start_idx:start_idx + truncate_size], \
                clean_imag[:, start_idx:start_idx + truncate_size]

        return np.dstack([clean_real, clean_imag]), np.dstack([noisy_real, noisy_imag])

    def load_wav(self, path):
        ret, _ = librosa.load(path, sr=self.sr)
        
        return ret

    def next_batch(self, batch_size):
        def _sample(data_list, k):
            return [random.choice(data_list) for _ in range(k)]

        sample_batch = partial(_sample, k=batch_size)

        while True:
            s_train_path, n_train_path, s_test_path, n_test_path = map(
                    sample_batch, 
                    [self.s_train, self.n_train, self.s_test, self.n_test])

            # load wav files
            '''
            with Pool(self.n_worker) as p:
                s_train = p.map(self.load_wav, s_train_path)
            with Pool(self.n_worker) as p:
                n_train = p.map(self.load_wav, n_train_path)
            with Pool(self.n_worker) as p:
                s_test = p.map(self.load_wav, s_test_path)
            with Pool(self.n_worker) as p:
                n_test = p.map(self.load_wav, n_test_path)
            '''
            s_train = map(self.load_wav, s_train_path)
            n_train = map(self.load_wav, n_train_path)
            s_test = map(self.load_wav, s_test_path)
            n_test = map(self.load_wav, n_test_path)

            # mix
            mixed_train, mixed_test = [], []
            c_train, c_test = [], []
            for s_tr, n_tr, s_te, n_te in zip(s_train, n_train, s_test, n_test):
                mixed_train.append(self.mix_wav(s_tr, n_tr))
                mixed_test.append(self.mix_wav(s_te, n_te))
                c_train.append(s_tr)
                c_test.append(s_te)

            # wav to spectrogram
            '''
            with Pool(self.n_worker) as p:
                spec_train = p.map(self.load_truncate, mixed_train)
            with Pool(self.n_worker) as p:
                spec_test = p.map(self.load_truncate, mixed_test)
            '''
            clean_train, noisy_train, clean_test, noisy_test = [], [], [], []

            for ctr, ntr, cte, nte in zip(c_train, mixed_train, c_test, mixed_test):
                ctr, ntr = self.load_truncate(ctr, ntr)
                cte, nte = self.load_truncate(cte, nte)
                clean_train.append(ctr)
                noisy_train.append(ntr)
                clean_test.append(cte)
                noisy_test.append(nte)

            yield np.stack(clean_train), np.stack(noisy_train), np.stack(clean_test), np.stack(noisy_test)
