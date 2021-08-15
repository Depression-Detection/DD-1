import tensorflow as tf
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
from tensorflow.keras import layers

# Loading Data From Github
df = pd.read_csv("https://raw.githubusercontent.com/Depression-Detection/reddit-webscraping/main/Data/comment_data.csv", converters={'Comments': eval}, header=0)

# Dividing strings in data to separate rows & labeling them
depressed_status = []
comment_list = []
for x in df['Comments'][:15]:
  for comment in x:
    comment_list.append(comment)
    depressed_status.append(1)
for x in df['Comments'][15:]:
  for comment in x:
    comment_list.append(comment)
    depressed_status.append(0)

d = {'Depressed?': depressed_status, 'Comment': comment_list}
df = pd.DataFrame(data=d)

# Preprocessing

import nltk
import string
import re
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('stopwords')

def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

def tokenization(text):
    tokens = re.split(' ',text)
    return tokens

stopwords = nltk.corpus.stopwords.words('english')
def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output

def deep_clean(text):
  while '' in text:
    text.remove('')
  while ' ' in text:
    text.remove(' ')
  return text

wordnet_lemmatizer = WordNetLemmatizer()
def lemmatizer(text):
  lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
  return lemm_text

def join(text):
  " ".join(text)

# Removing Punctuation
df['Clean_Text']= df['Comment'].apply(lambda x:remove_punctuation(x)).apply(lambda x: x.lower()).apply(lambda x: tokenization(x)).apply(lambda x:remove_stopwords(x)).apply(lambda x:deep_clean(x)).apply(lambda x: lemmatizer(x)).apply(lambda x: " ".join(x))
df.head

# Shuffling and Tokenizing

df = df.sample(frac=1).reset_index(drop=True)
tokenizer = Tokenizer(num_words=10000, oov_token='<UNK>')
tokenizer.fit_on_texts(df.Comment)
train_sequences = tokenizer.texts_to_sequences(df.Comment)
train_padded = pad_sequences(train_sequences, padding='post', truncating='post', maxlen=100)

# Splitting Train and Test Data

df_train = train_padded[:900]
df_test = train_padded[900:]

# Train Data
x_train = df_train
trres = df['Depressed?'][:900]

# Test Data
x_val = df_test
teres = df['Depressed?'][900:]
maxlen = max([len(x) for x in x_train])

count0 = 0
count1 = 0
for x in teres:
  if x == 0:
    count0+=1
  if x==1:
    count1+=1
print(count1/(count0+count1))

# ML Model

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SpatialDropout1D, GlobalAveragePooling1D, Embedding

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 16, input_length=maxlen),
    tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

history = model.fit(
    x_train, trres, batch_size=16, epochs=25, validation_data=(x_val, teres)
)