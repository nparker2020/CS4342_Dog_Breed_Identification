# noinspection PyUnresolvedReferences
from PIL import Image
import os
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import pickle
from datetime import datetime
import numpy as np
import time

pathString = os.path.join("C:", os.sep, "Users", "Noah Parker", "ML_Final_Project_Repo", "dog-breed-identification")
resized_directory = pathString+"\\resized"
resizeWidth = 200
resizeHeight = 200
limit = None;


class ML_Final_Project():

    def __init__(self):
        #self.convertImages();
        self.trainNClasses(4)

    def trainNClasses(self, n):
        print("training", n, " classes...")
        X, y = self.loadLabelsForNClasses(n);
        halfOfXIndex = int(X.shape[0] / 2);
        print("half of X index: ", halfOfXIndex);
        print("size of X: ", X.shape[0], " images.")
        indexRanges = np.arange(0, X.shape[0]);
        np.random.shuffle(indexRanges);

        testingX = X[  indexRanges[halfOfXIndex : X.shape[0]] ] #second half of training set
        testingY = y[ indexRanges[halfOfXIndex : X.shape[0]] ]
        trainingX = X[indexRanges[0 : halfOfXIndex]];
        trainingY = y[indexRanges[0 : halfOfXIndex]];

        descent_classifier = SGDClassifier(loss='log', tol=1e-5, max_iter=5000, n_jobs=4);
        ITERATIONS = 5000;
        print("Training....")
        for i in range(ITERATIONS):
            timeBefore = time.time()
            if i == 0:
                descent_classifier.partial_fit(trainingX, trainingY, classes=np.unique(y));
            else:
                descent_classifier.partial_fit(trainingX, trainingY);
            timeAfter = time.time();
            elapsed = timeAfter - timeBefore;
            estimatedTime = (elapsed / 60) * (ITERATIONS - i);
            if i % 1000 == 0:
                print("", i, " iterations complete. Took ", elapsed, " seconds.");
                print("estimated time to complete: ", estimatedTime, " minutes.")

        print("Done!")
        print("Testing....")
        meanError = descent_classifier.score(testingX, testingY);
        print("mean testing error: ", meanError)
        # print("Confidences: ");
        predictions = descent_classifier.predict(testingX);

        yHat = predictions
        print("yHat: ", predictions);


        print("y: ", testingY)
        fPCvalue = np.sum(testingY == yHat) / testingY.shape[0];
        print("testing FPC: ", fPCvalue)
        # yHat = np.argmax(confidences, axis=1);
        # print("yHat: ", yHat)
        # print("testingY: ", testingY)
        #
        #




    def trainAllClasses(self):
        X, y = self.loadLabels();
        testingX = self.reshapeTrainingImages(self.training_ids[9000:10000]);
        testingY = self.training_labels[9000:10000];
        print("Training....")
        print("testing y.shape: ", testingY.shape)
        # clf = LogisticRegression(multi_class='multinomial', max_iter=5000).fit(X, y);
        clfList = [];
        ENSEMBLE_SIZE = 1;
        BATCH_SIZE = 4500;
        confidenceSum = np.zeros((testingY.shape[0], np.unique(y[0:ENSEMBLE_SIZE * BATCH_SIZE]).shape[0]));
        print("confidenceSum shape: ", confidenceSum.shape)

        descent_classifier = SGDClassifier(loss='squared_loss', tol=1e-3, max_iter=5000);
        ITERATIONS = 50;
        for i in range(ENSEMBLE_SIZE):
            startIndex = i * BATCH_SIZE;
            endIndex = startIndex + BATCH_SIZE;
            X = self.reshapeTrainingImages(self.training_ids[startIndex:endIndex])
            y = self.training_labels[startIndex:endIndex]
            # clf = LogisticRegression(multi_class='multinomial', solver='sag', tol=0.01, max_iter=2000, n_jobs=4).fit(X,y);
            for b in range(ITERATIONS):
                timeBefore = time.time()
                descent_classifier.partial_fit(X, y, classes=np.unique(testingY));
                timeAfter = time.time();
                elapsed = timeAfter - timeBefore;
                print("", b, " iterations complete. Took ", elapsed, " seconds.");
                estimatedTime = (elapsed / 60) * (ITERATIONS - b);
                print("estimated time to complete: ", estimatedTime, " minutes.")

            print("", i, " done...")

        # for i in range(ENSEMBLE_SIZE):
        #     clf = clfList[i];
        #     scores = clf.decision_function(testingX);
        #     if scores.shape != confidenceSum.shape:
        #         #resizes with appended zeros
        #         #(this happens in the case where a machine doesn't see one or more classes when training
        #         scores = np.resize(scores, confidenceSum.shape)
        #     confidenceSum = confidenceSum + scores
        #
        # averagedConfidences = confidenceSum / testingY.shape[0];
        averagedConfidences = descent_classifier.decision_function(testingX);
        print("Pickling confidences (averaged)...")
        fileName = "scores_" + str(datetime.now()) + ".p";
        fileName = fileName.replace(":", "_")
        saved = pickle.dump(averagedConfidences, open(fileName, "wb"));
        #
        # #averagedConfidences = pickle.load(open("scores_2020-04-27 12_47_57.206737.p", "rb"));
        #
        print("averagedConfidences shape: ", averagedConfidences.shape)
        averageYhat = np.argmax(averagedConfidences, axis=1);
        print("averageYhat.shape: ", averageYhat.shape)
        print("testingY shape:", testingY.shape)

        y = np.argmax(testingY);

        print("averageYhat: ", averageYhat)
        print("testingY:", testingY)
        print("breed[testingY[1]]: ", self.training_classes[testingY[1]])
        fPC = np.sum(testingY == averageYhat) / testingY.shape[0]
        print("testing FPC: ", fPC)

        print("Attempting to save model.....")
        fileName = "model_" + str(datetime.now()) + ".p";
        fileName = fileName.replace(":", "_")
        saved_model = pickle.dump(descent_classifier, open(fileName, "wb"));

    def convertImages(self):

        #This should point to the base directory of all the data:
        print("converting images....")
        #C:\Users\Noah Parker\ML_Final_Project_Repo\dog-breed-identification\test
        count = 0;
        for image_file in os.listdir(pathString + "\\train"):
            try:
                with Image.open(pathString+"\\train\\"+image_file) as im:
                    resized = im.resize((resizeWidth, resizeHeight));
                    bw = resized.convert(mode="L");
                    bw.save(os.path.join(resized_directory, image_file), "JPEG")
                    count = count + 1;
                    im.close();
                    resized.close();
                    bw.close();
            except IOError as e:
                print("Error resizing image: ", e)

            if limit is not None and (count >= limit):
                break

        print("Done! ", count," images resized to ", resizeWidth, "x", resizeHeight)
        print("Saved to: ", resized_directory)

    def loadLabelsForNClasses(self, n):
        d = pandas.read_csv(pathString + '\\labels.csv')
        breeds = d.breed;
        unique = breeds.unique();  # All of the classes (120)
        print("Loading labels for ", n, " classes.")

        imageRows = None;
        yRows = None;
        for i in range(n):
            print(" ", unique[i], "selected.");
            rows = d[d.breed == unique[i]].copy();
            images = rows;
            rows.breed[rows.index] = i;
            print("", rows.id.size, " images for ", unique[i])
            if imageRows is None:
                imageRows = rows.id
                yRows = np.array(rows.breed, dtype=np.int32);
            else:
                print("appending images! size before: ", imageRows.size, ", size of rows: ", rows.id.size)
                imageRows = pandas.concat((imageRows, rows.id));
                print("size after: ", imageRows.size)
                yRows = np.concatenate((yRows, np.array(rows.breed, dtype=np.int32)));
            print("size of imageRows: ", imageRows.size)

        self.training_labels = yRows;
        return self.reshapeTrainingImages(imageRows), yRows;


    def loadLabels(self):
        d = pandas.read_csv(pathString+'\\labels.csv')
        breeds = d.breed;
        unique = breeds.unique(); #All of the classes (120)
        print(unique)

        #replace string breed labels with class # according to its index in unique
        for i in range(len(unique)):
            indices = d.breed[d.breed == unique[i]].index;
            d.breed[indices] = np.argwhere(unique == unique[i])[0][0]

        #d.breed is now labels for scikit learn
        #print("d.breed: ", d.breed)

        self.training_labels = np.array(d.breed, dtype=np.int32);
        self.training_classes = unique;
        self.training_ids = d.id;

        return self.reshapeTrainingImages(d.id[0:2000]), np.array(d.breed[0:2000], dtype=np.int32)

    def reshapeTrainingImages(self, fileNames):
        #X should be n_samples x n_features according to scikit learn.
        X = np.empty((fileNames.shape[0], (resizeWidth*resizeHeight)))
        testPath = os.path.join(resized_directory, "000bec180eb18c7604dcecc8fe0dba07.jpg")
        opened = Image.open(testPath);

        count = 0;
        for imageName in fileNames:
            try:
                with Image.open(os.path.join(resized_directory, imageName+".jpg")) as im:
                    array = np.array(im);
                    #if count == 0 or count == 90:
                        #im.show()
                    reshaped = np.reshape(array, (1, (resizeWidth*resizeHeight)))
                    X[count] = reshaped;
                    count = count + 1;
            except IOError as e:
                print("Error loading resized image!.... ", e);

        return X

if __name__ ==  "__main__":
    ML_Final_Project();

    # X,y = loadLabels()
    # y = np.array(y, dtype=np.int32)
    # print("y type: ", type(y))
    # print("Training...")
    # clf = LogisticRegression(multi_class='multinomial', max_iter=5000).fit(X,y);
    # fileName = "model_"+str(datetime.now())+".p";
    #
    # saved = pickle.dump(clf, open(fileName, "wb"));
    #
    # print("Training Accuracy (mean accuracy): ")
    # scores = clf.decision_function(X);
    # mean_acc = clf.score(X, y);
    # print(mean_acc)



#***********************************************************
#100,000 Iterations, dingo v boston terrier
# yHat:  [1 1 1 1 0 0 0 1 0 1 1 0 1 0 0 0 1 1 0 1 0 1 1 1 0 0 1 1 0 1 1 1 0 0 1 0 0
#  0 0 0 1 0 1 0 1 1 1 1 1 1 1 1 1 0 0 1 1 0 0 1 0 0 1 1 1 0 1 1 1 1 1 0 1 1
#  0 1 1 1 1 0 1 1 0 0]
# y:  [1 1 1 0 0 0 0 1 0 0 1 1 0 0 0 0 0 1 0 0 1 0 1 1 1 0 1 1 1 0 0 1 0 0 1 0 0
#  1 0 1 0 0 1 0 1 1 0 1 0 0 1 1 1 0 0 1 0 1 1 0 0 1 0 1 1 0 0 0 1 1 0 1 1 1
#  0 0 0 0 1 0 1 1 0 0]
# testing FPC:  0.6309523809523809
