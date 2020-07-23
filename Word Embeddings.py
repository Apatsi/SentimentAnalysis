import numpy as np
import pandas as pd
from sklearn import model_selection
import re
import seaborn as sns
import matplotlib.pyplot as plt
from keras_preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Flatten, GRU, Dense, Flatten, Embedding
from tensorflow.keras.models import Sequential

with open('dataset/train.ft.txt', 'r', encoding="utf8") as file:
    lines = file.readlines()
file.close()


# create a dataframe
DF_text_data = pd.DataFrame()

texts=[]
labels=[]
for line in lines:
    line=line.split()
    labels.append(1) if line[0] =="__label__2" else labels.append(0)
    texts.append(" ".join(line[1:]))

DF_text_data['reviews'] = texts
DF_text_data['labels'] = labels

_, X_data,_, y_data = \
    model_selection.train_test_split(DF_text_data['reviews'],
                                     DF_text_data['labels'], test_size=0.02)

def preprocess(in_text):
    # If we have html tags, remove them by this way:
    #out_text = remove_tags(in_text)

    # Remove punctuations and numbers
    out_text = re.sub('[^a-zA-Z]', ' ', in_text)
    # Convert upper case to lower case
    out_text="".join(list(map(lambda x:x.lower(),out_text)))
    # Remove single character
    out_text = re.sub(r"\s+[a-zA-Z]\s+", ' ', out_text)
    return out_text


text_data=[]
for review in list(X_data):
    text_data.append(preprocess(review))


DF_text= pd.DataFrame()
DF_text['reviews'] = text_data
DF_text['labels'] = list(y_data)

sns.countplot(x='labels', data=DF_text)
plt.show()


X_train, X_test, y_train, y_test = \
    model_selection.train_test_split(DF_text['reviews'],
                                     DF_text['labels'], test_size=0.30)


X_train=np.array(X_train.values.tolist())
X_test=np.array(X_test.values.tolist())
y_train=np.array(y_train.values.tolist())
y_test=np.array(y_test.values.tolist())

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
word_index=tokenizer.word_index
vocab_size = len(word_index)+1
print("vocab size:", vocab_size)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)


maxlen = 100
X_train_pad = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test_pad = pad_sequences(X_test, padding='post', maxlen=maxlen)

# Create the Model
model = Sequential()
model.add(Embedding(vocab_size, 20, input_length=maxlen))
model.add(GRU(units=32,dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())

model.fit(X_train_pad, y_train, batch_size=128, epochs=10, verbose=1, validation_data=(X_test_pad, y_test))
