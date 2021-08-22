import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage import maximum_filter1d
import soundfile as sf

def envelope(y, rate, threshold):
    """
    Args:
        - y: 信号データ
        - rate: サンプリング周波数
        - threshold: 雑音判断するしきい値
    Returns:
        - mask: 振幅がしきい値以上か否か
        - y_mean: Sound Envelop
    """
    y_mean = maximum_filter1d(np.abs(y), mode="constant", size=rate//20)
    mask = [mean > threshold for mean in y_mean]
    return mask, y_mean

y, sr = librosa.load('org.wav')

mask, y_mean = envelope(y, sr, -0.5)

fig = plt.figure()
fig.add_subplot(2, 1, 1)
librosa.display.waveplot(y, sr=sr)
fig.add_subplot(2, 1, 2)
librosa.display.waveplot(y_mean, sr=sr)
plt.tight_layout()
plt.show()

n_fft = 2048
hop_length = 512
win_length = 2048
n_std_thresh=1.5

def _stft(y, n_fft, hop_length, win_length):
    return librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

def _amp_to_db(x):
    return librosa.amplitude_to_db(x, ref=1.0, amin=1e-20, top_db=80.0)

noise_clip, sr = librosa.load('0d57e1251c35ff84f087abb48d60864f.wav')
#noise_clip, sr = librosa.load('noised_futta-garden.wav')

noise_stft = _stft(noise_clip, n_fft, hop_length, win_length)
noise_stft_db = _amp_to_db(np.abs(noise_stft))

mean_freq_noise = np.mean(noise_stft_db, axis=1)
std_freq_noise = np.std(noise_stft_db, axis=1)
noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh

librosa.display.waveplot(mean_freq_noise, sr=sr)

audio_clip, sr = librosa.load('org.wav')
#audio_clip, sr = librosa.load('futta-garden.wav')

n_grad_freq = 2  # マスクで平滑化する周波数チャンネルの数
n_grad_time = 4  # マスクを使って滑らかにする時間チャンネル数
prop_decrease = 1.0  # ノイズをどの程度減らすか

# 音源もSTFTで特徴量抽出する
sig_stft = _stft(audio_clip, n_fft, hop_length, win_length)
sig_stft_db = _amp_to_db(np.abs(sig_stft))

# 時間と頻度でマスクの平滑化フィルターを作成
smoothing_filter = np.outer(
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_freq + 1, endpoint=False),
                np.linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1],
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_time + 1, endpoint=False),
                np.linspace(1, 0, n_grad_time + 2),
            ]
        )[1:-1],
    )
smoothing_filter = smoothing_filter / np.sum(smoothing_filter)

# 時間と周波数のしきい値の計算
db_thresh = np.repeat(
        np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
        np.shape(sig_stft_db)[1],
        axis=0,
    ).T
sig_mask = sig_stft_db < db_thresh
sig_mask = scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
sig_mask = sig_mask * prop_decrease

mask_gain_dB = np.min(_amp_to_db(np.abs(sig_stft)))

def _db_to_amp(x,):
    return librosa.core.db_to_amplitude(x, ref=1.0)

sig_stft_db_masked = (
        sig_stft_db * (1 - sig_mask)
        + np.ones(np.shape(mask_gain_dB)) * mask_gain_dB * sig_mask)

def _istft(y, hop_length, win_length):
    return librosa.istft(y, hop_length, win_length)

sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
sig_stft_amp = (_db_to_amp(sig_stft_db_masked) * np.sign(sig_stft)) + (1j * sig_imag_masked)

recovered_signal = _istft(sig_stft_amp, hop_length, win_length)

sf.write('result.wav', recovered_signal, sr, 'PCM_24')