import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load Data
test = pd.read_csv('https://raw.githubusercontent.com/Depression-Detection/resources/main/Emotions%20NLP%20Data/test.txt',sep=";",header=None)
train = pd.read_csv('https://raw.githubusercontent.com/Depression-Detection/resources/main/Emotions%20NLP%20Data/train.txt',sep=";",header=None)
val = pd.read_csv('https://raw.githubusercontent.com/Depression-Detection/resources/main/Emotions%20NLP%20Data/val.txt',sep=";",header=None)


#Seperated train dataframe into Sad and notSad to assign is_sad? values and then appended them and shuffled
train.columns = ['text', 'emotion']
SadTrain = train.loc[(train.emotion == "sadness")]
SadTrain['is_sad?'] = 1
SadTrainingText= SadTrain['text']
#SadTrainingText.head()
#SadTrain.head()
#SadTrain.info()
notSadTrain = train.loc[(train.emotion!='sadness')]
notSadTrain['is_sad?'] = 0
#notSadTrain.head()
train = SadTrain.append(notSadTrain)
train=train.sample(frac=1)
trainText=train['text']
trainText.head()

#same for test data
test.columns = ['text', 'emotion']
SadTest = test.loc[(test.emotion == "sadness")]
SadTest['is_sad?'] = 1
SadTestingText= SadTest['text']
notSadTest = test.loc[(test.emotion!='sadness')]
notSadTest['is_sad?'] = 0
#notSadTrain.head()
test = SadTest.append(notSadTest)
test=test.sample(frac=1)
testText=test['text']
testText.head()

trainingLabels=train['is_sad?']
testingLabels=test['is_sad?']
#testingLabels.head()

tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(trainText)
word_index = tokenizer.word_index

TrainingSequences = tokenizer.texts_to_sequences(trainText)
TrainingPadded = pad_sequences(TrainingSequences, padding='post', truncating='post', maxlen=100)

TestingSequences = tokenizer.texts_to_sequences(testText)
TestingPadded = pad_sequences(TestingSequences, padding='post', truncating='post', maxlen=100)

# Need this block to get it to work with TensorFlow 2.x
import numpy as np
TrainingPadded = np.array(TrainingPadded)
trainingLabels = np.array(trainingLabels)
TestingPadded = np.array(TestingPadded)
testingLabels = np.array(testingLabels)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(5000, 16, input_length=100),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

num_epochs = 10
history = model.fit(TrainingPadded, trainingLabels, epochs=num_epochs, validation_data=(TestingPadded, testingLabels), verbose=2)