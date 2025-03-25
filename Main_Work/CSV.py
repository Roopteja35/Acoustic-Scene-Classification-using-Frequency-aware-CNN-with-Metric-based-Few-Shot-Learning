import os
import librosa
from scipy.spatial.distance import euclidean
import csv


def load_audio(file_path):
    audio_data, sr = librosa.load(file_path, sr=None)
    return audio_data, sr


def calculate_distance(audio_data1, audio_data2):
    return euclidean(audio_data1, audio_data2)


def main():
    directory = "C:/Users/Dell/PycharmProjects/BL_2/dataset_1/metro"
    audio_files = os.listdir(directory)

    selected_audios = set()  # Use a set to avoid duplicates
    min_distance = float('inf')
    max_distance = float('-inf')

    for i in range(len(audio_files)):
        for j in range(i + 1, len(audio_files)):
            file1 = os.path.join(directory, audio_files[i])
            file2 = os.path.join(directory, audio_files[j])

            audio_data1, sr1 = load_audio(file1)
            audio_data2, sr2 = load_audio(file2)

            if sr1 != sr2:
                print(f"Skipping comparison between {file1} and {file2} due to different sampling rates.")
                continue

            distance = calculate_distance(audio_data1, audio_data2)
            if distance < min_distance:
                min_distance = distance
            if distance > max_distance:
                max_distance = distance

            print(f"Distance between {file1} and {file2}: {distance}")

            if distance < 10:
                selected_audios.add(audio_files[i])  # Add to set to ensure uniqueness
                selected_audios.add(audio_files[j])

    new_dataset_path = "C:/Users/Dell/PycharmProjects/BL_2/dataset_1/metro/metro_10.csv"
    with open(new_dataset_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Audio File Name'])  # Write the header
        for audio_file in selected_audios:
            writer.writerow([audio_file])  # Write each audio file name

    print(f"Min distance: {min_distance}")
    print(f"Max distance: {max_distance}")


if __name__ == "__main__":
    main()
