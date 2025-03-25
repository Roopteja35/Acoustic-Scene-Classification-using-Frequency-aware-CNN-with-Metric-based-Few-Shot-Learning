import os
import librosa
import numpy as np
from pandas.core.dtypes.common import classes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import librosa.display

# Function to extract log-mel spectrogram features from audio files
def extract_features(file_path, n_mels=44, hop_length=512, n_fft=2048):
    y, sr = librosa.load(file_path)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    return log_mel_spectrogram

# Function to load data and labels
def load_data_and_labels(data_dir):
    data = []
    labels = []
    label_encoder = LabelEncoder()

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if file_path.endswith(".wav"):
                    features = extract_features(file_path)
                    data.append(features)
                    labels.append(folder)

    # Encode labels
    encoded_labels = label_encoder.fit_transform(labels)

    return np.array(data), np.array(encoded_labels), label_encoder.classes_

# Load data and labels
data_dir = "C:/Users/Dell/PycharmProjects/BL_2/dataset"  # replace with the path to your dataset
data, labels, classes = load_data_and_labels(data_dir)

# Visualize log-mel spectrogram for one audio in each scene on a single plot
plt.figure(figsize=(15, 10))

for i, scene_class in enumerate(classes):
    scene_files = [os.path.join(data_dir, scene_class, f) for f in os.listdir(os.path.join(data_dir, scene_class)) if f.endswith(".wav")]
    example_file = scene_files[0]  # Take the first file for each scene as an example

    # Extract log-mel spectrogram
    y, sr = librosa.load(example_file)  # Get the sample rate
    example_features = extract_features(example_file)

    # Plot the log-mel spectrogram
    plt.subplot(2, 3, i + 1)  # Adjust the subplot layout based on the number of scenes
    librosa.display.specshow(example_features, x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Log-Mel Spectrogram - {scene_class}')

plt.tight_layout()
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Reshape data for SVM input (flatten the spectrogram)
X_train_flat = X_train.reshape((X_train.shape[0], -1))
X_test_flat = X_test.reshape((X_test.shape[0], -1))

# Build SVM model
svm_model = SVC(kernel='linear', C=1)  # You can experiment with different kernels and C values

# Train the SVM model
svm_model.fit(X_train_flat, y_train)

# Predict on the test set
svm_predictions = svm_model.predict(X_test_flat)

# Evaluate the SVM model
svm_accuracy = accuracy_score(y_test, svm_predictions)
print(f'SVM Test Accuracy: {svm_accuracy}')

# Visualize the output
plt.figure(figsize=(12, 4))

# Pick a sample index to visualize
sample_index = 0

# Plot the input log-mel spectrogram
plt.subplot(1, 2, 1)
librosa.display.specshow(X_test[sample_index], x_axis='time', y_axis='mel')  # Adjust the indexing here
plt.title('Input Log-Mel Spectrogram')

# Plot the SVM predicted class and ground truth
plt.subplot(1, 2, 2)
plt.bar([0, 1], [svm_predictions[sample_index], y_test[sample_index]], tick_label=['Predicted', 'Ground Truth'])
plt.title(f'SVM Output\nPredicted Class: {svm_predictions[sample_index]}, Ground Truth: {y_test[sample_index]}')

plt.tight_layout()
plt.show()

# Optionally, you can perform cross-validation to get a more robust estimate of performance
svm_cv_scores = cross_val_score(svm_model, X_train_flat, y_train, cv=3)
print(f'SVM Cross-Validation Mean Accuracy: {np.mean(svm_cv_scores)}')
