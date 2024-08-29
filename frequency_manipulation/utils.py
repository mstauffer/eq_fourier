import numpy as np
from scipy.signal import butter, lfilter

def create_bandpass_filter(center_freq, bandwidth, fs, M):
    nyquist = 0.5 * fs
    f_low = (center_freq - bandwidth / 2) / nyquist
    f_high = (center_freq + bandwidth / 2) / nyquist

    n = np.arange(-M // 2, M // 2 + 1)
    h = np.sinc(2 * f_high * n) - np.sinc(2 * f_low * n)
    window = np.hamming(M + 1)
    h *= window
    h /= np.sum(h)
    
    return h

def calculate_band_magnitudes(audio_chunk, sampling_rate, center_frequencies):
    band_magnitudes = []
    nyquist = 0.5 * sampling_rate
    for center_freq in center_frequencies:
        bandwidth = center_freq * 0.5
        low = max(0, (center_freq - bandwidth) / nyquist)
        high = min(1, (center_freq + bandwidth) / nyquist)

        if high > low and low > 0 and high < 1:
            b, a = butter(2, [low, high], btype='band')
            filtered_chunk = lfilter(b, a, audio_chunk)
            magnitude = np.sqrt(np.mean(filtered_chunk**2))
            band_magnitudes.append(magnitude)
        else:
            band_magnitudes.append(0)

    return band_magnitudes