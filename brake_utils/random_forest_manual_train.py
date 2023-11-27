from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import pickle

class RandomForest():
    def __init__(self, load_path = None):

        self.RF = RandomForestClassifier(n_estimators=100)

        if load_path != None:
            self.RF.load()

    def train(self):
        print("Random Forest Train Init")
        # acquire training data from folders
        #!!!!!!!!!!!!!!!!!!!!!!!! Random fores train folders !!!!!!!!!!!!!!!!!!!!!!!!
        input_folders_test = ["D:/InspiritAI/Code/brake-light-project/TEST/brake_TEST_OFF", \
        "D:/InspiritAI/Code/brake-light-project/TEST/brake_ON_TEST"]
        input_folders_train = ["D:/InspiritAI/Code/brake-light-project/TRAIN/brake_TRAIN_OFF", \
        "D:/InspiritAI/Code/brake-light-project/TRAIN/brake_ON_TRAIN"]

        features_test = []
        labels_test = []
        for folder in input_folders_test:
            # print(folder)
            # for subdir, dirs, files in os.walk(folder):
            for (subdir, dirs, files) in os.walk(folder):
                # print(files)
                for image in files:
                    # print(image)
                    img = cv2.imread(folder + '/' + image)
                    # print(folder + '/' + image)
                    img = cv2.resize(img, (30,30))
                    #print(img.shape)
                    features_test.append(img.flatten())

                    if folder[-3:] == "OFF":
                        labels_test.append(0)
                    else:
                        labels_test.append(1)

        # all data acquired
        labels_test = np.array(labels_test)
        print("labels_test = ", labels_test)
        # print("features_test = ", features_test)
        features_test = np.stack(features_test, axis = 0)

        features_train = []
        labels_train = []
        for folder in input_folders_train:
            for subdir, dirs, files in os.walk(folder):
                for image in files:
                    img = cv2.imread(folder + '/' + image)
                    img = cv2.resize(img, (30,30))
                    features_train.append(img.flatten())

                    if folder[-3:] == "OFF":
                        labels_train.append(0)
                    else:
                        labels_train.append(1)

        # all data acquired
        labels_train = np.array(labels_train)
        print("labels_train = ", labels_train)

        features_train = np.stack(features_train, axis = 0)
        history = self.RF.fit(features_train, labels_train)
        prediction_train = self.RF.predict(features_train)
        prediction_test = self.RF.predict(features_test)
        # print(f"{prediction_train=}")
        # print(f"{prediction_test=}")


        # print("train accuracy:")
        # print(self.RF.score(features_train, labels_train))
        train_accuracy = self.RF.score(features_train, labels_train)
        print(f"{train_accuracy=}")

        # print("test accuracy:")
        # print(self.RF.score(features_test, labels_test))
        test_accuracy = self.RF.score(features_test, labels_test)
        print(f"{test_accuracy=}")

        print("Train with manual test/train split succesfully completed")

        # return self.RF.score(features_test, labels_test)
        return test_accuracy


    def predict(self, img):
        img = cv2.resize(img, (30,30))
        ################### Prediction prob  ###############
        on_predict = self.RF.predict_proba(np.expand_dims(img.flatten(), axis=0))
        # print("on_predict = ", on_predict)
        on_predict_return = on_predict[0][0]
        return on_predict_return

    def save(self):
        print("Saved the model!")
        filename = 'finalized_model.sav'
        pickle.dump(self.RF, open(filename, 'rb'))
