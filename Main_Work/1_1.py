import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

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
data_dir = "C:/Users/Dell/PycharmProjects/BL_2/dataset_2"  # replace with the path to your dataset
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

# Print the number of training and test samples
#print(f'Number of training samples: {len(X_train)}')
#print(f'Number of test samples: {len(X_test)}')

# Build Frequency-aware CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)))
model.add(layers.BatchNormalization())

# Pooling along the time axis only
model.add(layers.MaxPooling2D((1, 2)))  # Pooling window of (1, 2)

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((1, 2)))  # Pooling window of (1, 2)

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((1, 2)))  # Pooling window of (1, 2)

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(len(np.unique(labels)), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Reshape data for CNN input
X_train = X_train.reshape((*X_train.shape, 1))
X_test = X_test.reshape((*X_test.shape, 1))

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[callbacks.EarlyStopping(patience=3)])

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc}')

# Save the model
# model.save('acoustic_scene_classification_model.h5')

# Assuming you have a trained model, and your input data is a 4D array (batch_size, height, width, channels)
# In your case, it could be something like (num_samples, num_mel_bins, num_time_frames, 1)

# Assuming 'model' is your trained CNN model
model_output = model.predict(X_test)

# Let's pick one sample from the test set
sample_index = 0
sample_input = X_test[sample_index].reshape((1,) + X_test[sample_index].shape)  # Add batch dimension

# Get the model's prediction for the selected sample
prediction = model.predict(sample_input)

# Assuming your model has a softmax activation in the last layer
predicted_class = np.argmax(prediction)

# Visualize the output
plt.figure(figsize=(12, 4))

# Plot the input log-mel spectrogram
plt.subplot(1, 2, 1)
librosa.display.specshow(X_test[sample_index, :, :, 0], x_axis='time', y_axis='mel')
plt.title('Input Log-Mel Spectrogram')

# Plot the CNN output probabilities
plt.subplot(1, 2, 2)
plt.bar(range(len(classes)), prediction.ravel(), tick_label=classes)
plt.title(f'Model Output\nPredicted Class: {classes[predicted_class]}')

plt.tight_layout()
plt.show()

# Get predictions on the test set
# y_pred = model.predict_classes(X_test)

# Predict probabilities for the test set
y_pred_probabilities = model.predict(X_test)

# Extract predicted classes based on the highest probability
y_pred = np.argmax(y_pred_probabilities, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
