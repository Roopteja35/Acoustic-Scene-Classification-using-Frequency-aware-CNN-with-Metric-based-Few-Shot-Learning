import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

def calculate_euclidean_distance(audio_path1, audio_path2):
    # Load audio files
    waveform1, sr1 = librosa.load(audio_path1, sr=None)
    waveform2, sr2 = librosa.load(audio_path2, sr=None)

    # Check if sampling rates are the same
    if sr1 != sr2:
        print("Sampling rates are different. Comparison cannot proceed.")
        return None

    # Calculate Euclidean distance
    euclidean_distance = distance.euclidean(waveform1, waveform2)
    return euclidean_distance, waveform1, waveform2, sr1

# Example usage:
audio_path1 = "C:/Users/Dell/PycharmProjects/BL_2/dataset/airport/airport-barcelona-0-0-0-a.wav"
audio_path2 = "C:/Users/Dell/PycharmProjects/BL_2/dataset/airport/airport-barcelona-0-0-0-a.wav"

distance_threshold = 0.1  # Adjust the threshold as needed

euclidean_distance, waveform1, waveform2, sr1 = calculate_euclidean_distance(audio_path1, audio_path2)

if euclidean_distance is not None and euclidean_distance < distance_threshold:
    print("Audio files are similar.")
    # Update the new dataset with these files
else:
    print("Audio files are not similar.")

# Plotting the audio waveforms
plt.figure(figsize=(12, 4))

plt.subplot(2, 1, 1)
plt.plot(np.arange(len(waveform1)) / sr1, waveform1)
plt.title('Waveform - Audio File 1')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(np.arange(len(waveform2)) / sr1, waveform2)
plt.title('Waveform - Audio File 2')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

# Displaying sampling rates and Euclidean distance
print(f"Sampling Rate - Audio File 1: {sr1} Hz")
print(f"Sampling Rate - Audio File 2: {sr1} Hz")
print(f"Euclidean Distance: {euclidean_distance}")
