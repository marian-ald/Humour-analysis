from bert_embedding import BertEmbedding
import os
import sys
from helpers import data_process
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

#############
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.models import load_model
from sklearn.model_selection import train_test_split

import csv

os.environ["CUDA_VISIBLE_DEVICES"]="0"

max_sent_len = 45
input_size = 768 * max_sent_len
nb_epochs = 5



d = data_process()
train_orig_hline, train_edit_hline, train_labels = d.load_edited_prep_csv("../data/task-1/edited_train.csv")
test_orig_hline, test_edit_hline = d.load_edited_test_csv("../data/task-1/edited_dev.csv")

#train_text, train_labels = d.build_text("../data/task-1/edited_train.csv", 'train')
#test_text, test_labels = d.build_text("../data/task-1/edited_dev.csv", 'dev')

#sys.exit()
test_ids = d.load_dev_ids("../data/task-1/dev.csv")





"""
train_text = train_text[:10]
train_labels = train_labels[:10]
test_text = test_text[:10]
test_labels = test_labels[:10]

print("train text")
#[print(l) for l in train_text]
print(len(train_text))
print("train labels")
#[print(l) for l in train_labels]
"""

"""
sent = ""
max_len = 0
for i in range(len(train_text)):
    spl = train_text[i].split()
    if len(spl) > max_len:
        max_len = len(spl)
        sent = train_text[i]

print(max_len)i
print(sent)

sys.exit()
"""


"""
        sentences:- list of strings
"""
def extract_bert(sentences):
    bert_embedding = BertEmbedding()
    processed = []
    step = 0
    for s in sentences:
        processed.append(bert_embedding([s]))
        if step%100 == 0:
            print("prep step nb " + str(step))
        step = step + 1

    vectors = [p[0][1] for p in processed]

    print("Transformed to BERT  "+ str(len(vectors)) + str("strings"))

    return vectors



def transform_text(text):
    vectors = extract_bert(text)


    # Fill with zeros until the array has the maximum sentence size(45)
    for i in range(len(vectors)):
        vectors[i] = vectors[i] + [([0.0001] * 768)]*(max_sent_len-len(vectors[i]))

    vectors = np.array(vectors)
    vectors = np.array([line.flatten() for line in vectors])

    return vectors

def create_mlp(dim, regress=False):
    # define our MLP network
    model = Sequential()
    model.add(Dense(100, input_dim=dim, activation="relu"))
    model.add(Dense(50, activation="relu"))

    # check to see if the regression node should be added
    if regress:
        model.add(Dense(1, activation="linear"))

    return model

#############
#X_train, X_test, Y_train, Y_test = train_test_split(train_edit_hline, train_labels, test_size=0.20, random_state=10)


#train_x = transform_text(train_edit_hline)
test_x = transform_text(test_edit_hline)


train_x = d.open_pickle("train_x.pickle")
X_train, X_test, Y_train, Y_test = train_test_split(train_x, train_labels, test_size=0.20, random_state=10)

#d.save_pickle("train_x.pickle", train_x)
d.save_pickle("test_x.pickle", test_x)



#train_x = d.open_pickle("train_x.pickle")
#test_x = d.open_pickle("test_x.picle")
#print(len(test_x[0]))

#sys.exit()

def train_model(train_x, train_labels, test_x, test_labels):
    model = create_mlp(train_x.shape[1], regress=True)
    print(model.summary())

    #opt = Adam(lr=1e-i3, decay=1e-3 / 200)
    model.compile(loss='mean_squared_error', optimizer='sgd')

    # train the model
    print("[INFO] training model...")
#    model.fit(train_x, train_labels, validation_data=(test_x, test_labels), epochs=10, batch_size=8)
    model.fit(train_x, train_labels, epochs=10, batch_size=8)

    print("save model")
    model.save("saved_models/mlp_l100_l50_sgd_e100.h5")



#for i in range(20i0):
#    print(str(train_labels[i]) + "  " + str(train_text[i]))

#print(type(test_x))
#sys.exit()

def write_csv(ids, preds, fname):
    with open(fname, 'w') as csvFile:
        writer = csv.writer(csvFile)
        head = ['id', 'pred']
        writer.writerow(head)

        for i in range(len(ids)):
            row = [ids[i], preds[i]]
            writer.writerow(row)
    csvFile.close()



def make_all_tests(saved_model, test_text):
    model = load_model(saved_model)
    print(model.summary())

    preds = []
    #bert_text_x = transform_text(test_text)

    bert_text_x = d.open_pickle("test_x.pickle")

    print(len(bert_text_x))
    print(len(bert_text_x[0]))

    #d.save_pickle("test_text_bert_unproc.pickle", bert_text_x)

    for line in bert_text_x:
        a = np.array(line)
        
        line = a.reshape((1, 34560))

        pred = model.predict(line, verbose=2)
        preds.append(pred[0][0])
    write_csv(test_ids, preds, "dev_preds2.csv")
    #d.build_dev_results(test_ids, preds, "dev_preds.csv")



def test_model(saved_model):
    model = load_model(saved_model)
    print(model.summary())
    for i in range(200):
        print(str(train_labels[i]) + "  " + str(train_text[i]))

    while True:
        text = input("Enter your text> ")
        if text == "exit":
            sys.exit()
        print("text is: " + text)
        text_x = transform_text([text])
        a = np.array(text_x)
        print("shape")
        print(a.shape)
        sys.exit()
        pred = model.predict([text_x], verbose=2)
        print("pred: " + str(pred))



#train_model(X_train, Y_train, X_test, Y_test)
#train_model(train_x, train_labels, X_test, Y_test)


make_all_tests("saved_models/mlp_l100_l50_sgd_e100.h5", None)
#test_model("saved_models/mlp_l100_l50_sgd_e100.h5")




