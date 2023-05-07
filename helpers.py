import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def apply_stft(tensor, frame_length=256, frame_step=64):
    return tf.signal.stft(tensor, frame_length=frame_length, frame_step=frame_step)

def apply_istft(tensor, frame_length=256, frame_step=64):
    return tf.signal.inverse_stft(tensor, frame_length=frame_length, frame_step=frame_step) / 1.5

def complex_from_polar(magnitude, phase):
    return tf.complex(magnitude, 0.0) * tf.exp(tf.complex(0.0, phase))

def get_num_samples_for_num_frames(num_frames, frame_length=256, hop_length=64):
    return (num_frames - 1) * hop_length + frame_length

def plot_stft(stft, frame_step=64, sr=16000):

    # By convention time is x-axis.
    stft = tf.transpose(stft)

    num_bins, num_frames = stft.shape

    time_tick_spacing = num_frames // 5
    frequency_tick_spacing = num_bins // 4

    duration_seconds = num_frames * frame_step / sr
    time = np.linspace(0, duration_seconds, num_frames, endpoint=False)
    frequency = np.linspace(0, sr / 2, num_bins)

    time = time[::time_tick_spacing]
    frequency = frequency[::frequency_tick_spacing]

    time_ticks = np.arange(len(time)) * time_tick_spacing
    frequency_ticks = np.arange(len(frequency)) * frequency_tick_spacing

    plt.imshow(stft, origin='lower', aspect='auto')
    plt.xticks(time_ticks, labels=time)
    plt.yticks(frequency_ticks, labels=frequency)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")

def get_rms(x):
    return np.mean(np.sqrt(x ** 2))

def get_segments_above_threshold(dataset, num_samples, threshold_db):
    segments = []
    for audio in dataset:

        audio = audio[0]
        
        # Chop audio into slices.
        num_segments = len(audio) // num_samples
        audio = audio[:num_segments * num_samples]
        as_slices = audio.reshape(num_segments, num_samples)

        for i in range(num_segments):
            candidate = as_slices[i, :]
            rms = get_rms(candidate)
            level_db = 20 * np.log10(rms)

            if level_db >= threshold_db:
                segments.append(candidate)

    return segments

def reconstruction_snr(y, y_hat):

    signal = tf.reduce_sum(y ** 2, axis=1)
    noise = tf.reduce_sum((y - y_hat) ** 2, axis=1)
    
    return 10 * np.log10(signal / noise)

def plot_signal(x, sr):
    time = np.arange(len(x)) / sr
    plt.plot(time, x, linewidth=0.25)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")