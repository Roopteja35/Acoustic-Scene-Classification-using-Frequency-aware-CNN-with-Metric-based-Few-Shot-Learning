import os
import csv

# Set the path to the folder containing your audio files
folder_path = "C:/Users/Dell/PycharmProjects/MTASC/dataset/audio"

# List all files in the folder
audio_files = os.listdir(folder_path)

# Filter out non-audio files if needed
audio_files = [file for file in audio_files if file.endswith('.wav')]

# Create a CSV file to record the names and additional information
csv_file_path = "C:/Users/Dell/PycharmProjects/BL_2/dataset/audio_files.csv"
#csv_file_path = "C:/Users/Dell/PycharmProjects/MTASC/dataset/audio_files.csv"

# Writing the header to the CSV file
with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['File_Name', 'Scene'])

# Writing the audio file names and prefix to the CSV file
with open(csv_file_path, 'a', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    for audio_file in audio_files:
        prefix, _ = audio_file.split('-', 1)
        full_file_name = audio_file.replace('-', '_')  # Replace hyphens with underscores for CSV compatibility
        csv_writer.writerow([full_file_name, prefix])

print(f"CSV file with audio file names and prefix created at: {csv_file_path}")


# Create a CSV file to record the names
#csv_file_path = "C:/Users/Dell/PycharmProjects/BL_2/dataset/audio_files.csv"
#csv_file_path = "C:/Users/Dell/PycharmProjects/ASC/audio_files.csv"
