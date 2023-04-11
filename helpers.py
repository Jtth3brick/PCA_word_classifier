import numpy as np
import pyaudio
import matplotlib.pyplot as plt
from scipy.signal import correlate
import scipy.signal
import scipy
import librosa
from fastdtw import fastdtw


# Set up PyAudio and recording settings
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100

def record_audio_normalized(duration=2):
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    print("Recording...")

    # Start recording
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []

    # Record for the specified duration
    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.float32))

    # Stop recording
    stream.stop_stream()
    stream.close()
    p.terminate()

    print("Done recording")

    # Combine frames
    audio_data = np.concatenate(frames)

    # Normalize the audio data
    normalized_audio = audio_data / np.abs(audio_data).max()

    return normalized_audio

def align_clips(clips, ref_clip):
    aligned_clips = []

    for clip in clips:
        # print(f"ref clip shape: {ref_clip.shape}")
        # print(f"data clip shape: {clip.shape}")
        correlation = scipy.signal.correlate(ref_clip, clip, mode='full')
        delay = np.argmax(correlation) - len(ref_clip) + 1
        aligned_clip = np.roll(clip, delay)
        aligned_clips.append(aligned_clip)

    return aligned_clips


def noise_reduction_filter(audio_clips, n_fft=2048, hop_length=512):
    denoised_clips = []

    for clip in audio_clips:
        clip_stft = librosa.stft(clip, n_fft=n_fft, hop_length=hop_length)
        clip_mag, clip_phase = librosa.magphase(clip_stft)
        median_filter = scipy.signal.medfilt(clip_mag, kernel_size=(1, 3))
        denoised_stft = median_filter * clip_phase
        denoised_clip = librosa.istft(denoised_stft, hop_length=hop_length)
        denoised_clips.append(denoised_clip)

    return denoised_clips

def plot_audio_data_lists(audio_data_list_word1, audio_data_list_word2):
    num_plots = 2
    num_colors = len(audio_data_list_word1)

    # Generate distinct colors
    colors = plt.cm.viridis(np.linspace(0, 1, num_colors))

    # Plot audio data list word1
    plt.figure(figsize=(10, 12))

    plt.subplot(num_plots, 1, 1)
    for i, audio_data in enumerate(audio_data_list_word1, 1):
        time_vector = np.arange(audio_data.size) / RATE
        plt.plot(time_vector, audio_data, color=colors[i-1], label=f"Index {i}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Centered Audio Data for Word 1")
    plt.legend()

    # Plot audio data list word2
    plt.subplot(num_plots, 1, 2)
    for i, audio_data in enumerate(audio_data_list_word2, 1):
        time_vector = np.arange(audio_data.size) / RATE
        plt.plot(time_vector, audio_data, color=colors[i-1], label=f"Index {i}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Centered Audio Data for Word 2")
    plt.legend()

    plt.tight_layout()
    plt.show()