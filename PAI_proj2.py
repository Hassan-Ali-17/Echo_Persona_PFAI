import os
import csv
import string
import numpy as np
import speech_recognition as sr
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

dataset_path = "dataset"

recognizer = sr.Recognizer()

audio_extensions = [".wav", ".mp3", ".flac"]

stop_words = set(stopwords.words('english'))

texts = []
labels = []

with open("dataset.csv", "w", newline="", encoding="utf-8") as file:

    writer = csv.writer(file)

    writer.writerow(["text", "emotion"])

    for root, dirs, files in os.walk(dataset_path):

        for audio_file in files:

            extension = os.path.splitext(audio_file)[1].lower()

            if extension in audio_extensions:

                file_path = os.path.join(root, audio_file)

                emotion = os.path.basename(root)

                try:

                    with sr.AudioFile(file_path) as source:

                        audio = recognizer.record(source)

                        text = recognizer.recognize_google(audio)

                        writer.writerow([text, emotion])

                        texts.append(text)

                        labels.append(emotion)

                        print("Processed:", audio_file)

                except Exception as e:

                    print("Error:", audio_file)
                    print(e)

print("Dataset CSV Created")

def clean_text(text):

    text = text.lower()

    tokens = word_tokenize(text)

    cleaned_words = []

    for word in tokens:

        if word not in string.punctuation:

            if word not in stop_words:

                cleaned_words.append(word)

    return cleaned_words

processed_texts = []

for text in texts:

    processed_texts.append(clean_text(text))

print()
print("Cleaned Text Example:")
print(processed_texts[0])

vocabulary = []

for sentence in processed_texts:

    for word in sentence:

        if word not in vocabulary:

            vocabulary.append(word)

print()
print("Vocabulary Size:", len(vocabulary))

def bag_of_words(sentence, vocabulary):

    vector = np.zeros(len(vocabulary))

    for word in sentence:

        if word in vocabulary:

            index = vocabulary.index(word)

            vector[index] += 1

    return vector

X = []

for sentence in processed_texts:

    vector = bag_of_words(sentence, vocabulary)

    X.append(vector)

X = np.array(X)

print()
print("Shape Of Input Matrix:")
print(X.shape)

emotion_classes = []

for label in labels:

    if label not in emotion_classes:

        emotion_classes.append(label)

emotion_to_index = {}

for i in range(len(emotion_classes)):

    emotion_to_index[emotion_classes[i]] = i

y = []

for label in labels:

    encoded = [0] * len(emotion_classes)

    encoded[emotion_to_index[label]] = 1

    y.append(encoded)

y = np.array(y)

print()
print("Shape Of Output Matrix:")
print(y.shape)

print()
print("Emotion Mapping:")
print(emotion_to_index)