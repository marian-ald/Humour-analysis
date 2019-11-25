from bert_embedding import BertEmbedding
import os
import sys
from helpers import data_process
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

#############
from keras.layers.core import Dense
from keras.models import Model
from keras.models import load_model
import csv

os.environ["CUDA_VISIBLE_DEVICES"]="0"

max_sent_len = 45
input_size = 768 * max_sent_len
nb_epochs = 5
dim_layer1 = 100
dim_layer2 = 50

d = data_process()

train_text, train_labels = d.build_text("../data/task-1/edited_train.csv", 'train')
test_text, test_labels = d.build_text("../data/task-1/edited_dev.csv", 'dev')

test_ids = d.load_dev_ids("../data/task-1/dev.csv")


"""
    sentences:- list of strings
"""
def extract_bert(sentences):
    bert_embedding = BertEmbedding()
    processed = []
    step = 0
    for s in sentences:
        processed.append(bert_embedding([s]))
        if step%1000 == 0:
            print("Step nb " + str(step))
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
    model.add(Dense(dim_layer1, input_dim=dim, activation="relu"))
    model.add(Dense(dim_layer2, activation="relu"))

    if regress:
    model.add(Dense(1, activation="linear"))

    return model


#sys.exit()

def train_model():
    model = create_mlp(train_x.shape[1], regress=True)
    print(model.summary())

    #opt = Adam(lr=1e-i3, decay=1e-3 / 200)
    model.compile(loss='mean_squared_error', optimizer='sgd')

    # train the model
    print("[INFO] training model...")
    model.fit(train_x, train_labels, validation_data=(test_x, test_labels), epochs=10, batch_size=8)

    print("save model")
    model.save("saved_models/mlp_l100_l50_sgd_e100.h5")


def write_csv(ids, preds, fname):
    with open(fname, 'w') as csvFile:
        writer = csv.writer(csvFile)
        head = ['id', 'pred']
        writer.writerow(head)

        for i in range(len(ids)):
            row = [ids[i], preds[i]]
            writer.writerow(row)
    csvFile.close()



def test_model(saved_model, test_text):
    model = load_model(saved_model)
    print(model.summary())

    preds = []
    i = 0
    print(test_text[:2])
    #bert_text_x = transform_text(test_text)
    bert_text_x = d.open_pickle("test_text_bert_unproc.pickle")
    print(len(bert_text_x))
    print(len(bert_text_x[0]))

    for line in bert_text_x:
        a = np.array(line)
        
        line = a.reshape((1, 34560))

        pred = model.predict(line, verbose=2)
        preds.append(pred[0][0])
    write_csv(test_ids, preds, "dev_preds2.csv")



def enter_your_text(saved_model):
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
        print("shape" + str(a.shape))
        pred = model.predict([text_x], verbose=2)
        print("pred: " + str(pred))



test_model("saved_models/mlp_l100_l50_sgd_e100.h5", test_text)
#enter_your_text("saved_models/mlp_l100_l50_sgd_e100.h5")




