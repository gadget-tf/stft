import numpy as np
import librosa
import argparse
import time
import soundfile as sf

def main():
    # 処理開始時間
    process_start = time.time()

    # 音声ファイル読込
    data, sr = librosa.load('org.wav')
    print('[Done] 音声ファイル読込  Total Time: ', len(data) / sr, ' [sec]')

    print('[Do] ノイズ軽減 計算')
    # short-time Fourier transfor (STFT)
    #  (n_fft = 2048, hop_length = win_length(=n_fft) / 4, window = 'hann')
    # D: np.ndarray [shape=(1+n_fft / 2, t) T = t * hop_length])
    S = np.abs(librosa.stft(data))

    # Convert a power spectrogram to decibel(dB)
    D = librosa.power_to_db(S**2)

    # Calc Noise FrameRate
    _n_fft = 2048
    _hop_length = _n_fft / 4
    #noise_start = 0#int(_hop_length * float(args.start))
    #noise_finish = 5#int(_hop_length * float(args.finish))

    # Noise Copy and calc Average powers
    noise_D = D[:]#[:, noise_start : noise_finish]
    noise_Ave = np.average(noise_D, axis = 1)

    # Calc Spectral Subtraction
    D = D.transpose()
    SS = D - noise_Ave
    SS = SS.transpose()

    # Convert decibel to power spectrogram
    SSP = librosa.db_to_power(SS)
    
    # Inverse short-time Fourier transfor(ISTFT)
    OutputS = librosa.istft(SSP)

    # 正規化(normalize)
    OutputS = librosa.util.normalize(OutputS)

    print('[Done] ノイズ軽減 計算')

    # Output File (WAV)
    sf.write('output.wav', OutputS, sr, 'PCM_24')

    # 処理時間計算
    process_finish = time.time()
    process_time = process_finish - process_start

    print('[Done] ノイズ軽減ファイル出力  処理時間 : ', process_time, ' [sec]')

if __name__ == '__main__':
    main()
