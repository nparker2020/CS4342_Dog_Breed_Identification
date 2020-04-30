from sklearn.linear_model import LogisticRegression
from PIL import Image, ImageOps
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import csv

# pathString = os.path.join("C:", os.sep, "Users", "Nicholas", "Documents", "CS4342", "FinalProject", "CS4342_Dog_Breed_Identification","test")
pathString = os.path.join("C:", os.sep, "Users", "Nicholas", "Documents", "CS4342", "FinalProject", "CS4342_Dog_Breed_Identification", "test_resized")


# resizeWidth = 200
# resizeHeight = 200
# limit = None;
#
# count = 0;
# for image_file in os.listdir(pathString):
#     try:
#         with Image.open(os.path.join(pathString, image_file)) as im:
#             resized = im.resize((resizeWidth, resizeHeight));
#             resized.save(os.path.join(pathString2, image_file), "JPEG")
#             count = count + 1;
#     except IOError as e:
#         print("Error resizing image: ", e)
#
#     if limit is not None and (count >= limit):
#         break

# count = 0
# X = 0
# for image_file in os.listdir(pathString):
#     try:
#         temp_img = Image.open(os.path.join(pathString, image_file))
#         temp_img = ImageOps.grayscale(temp_img)
#         temp_img = np.asarray(temp_img)
#         # plt.imshow(temp_img)
#         # plt.show()
#         temp_img = temp_img.reshape((-1, 1))
#         if count == 0:
#             X = temp_img
#         else:
#             X = np.append(X, temp_img, axis=1)
#     except:
#         print("error opening image: " + str(image_file))
#
#     # if count > 1:
#     #     break
#     if count % 1000 == 0:
#         print(count)
#     count += 1
# print("X shape")
# print(X.T.shape)
# np.save("dog_breed_test_data.npy", X.T)

# mydict = {}
# with open("labels.csv", mode='r') as infile:
#     reader = csv.reader(infile)
#     mydict = {rows[0]: rows[1] for rows in reader}
#
# print(mydict)
# count = 0
# y = 0
# for image_file in os.listdir(pathString):
#     if count == 0:
#         y = np.array([mydict[image_file.split('.')[0]]])
#         y = np.reshape(y, (-1, 1))
#     else:
#         temp = np.array([mydict[image_file.split('.')[0]]])
#         temp = np.reshape(temp, (-1, 1))
#         print(temp.shape)
#         print(temp)
#         y = np.append(y, temp, axis=0)
#
#     count += 1
#
# print(y.shape)
# print(y)
# np.save("dog_breed_labels.npy", y)

# mydict = {}
# with open("labels.csv", mode='r') as infile:
#     reader = csv.reader(infile)
#     mydict = {rows[1]: 1 for rows in reader}
#
# print(len(mydict))
# print(mydict)
# count = 0
# for i in mydict.keys():
#     if count == 0:
#         pass
#     elif count == 1:
#         categories = np.array([str(i)])
#         categories = np.reshape(categories, (-1,1))
#     else:
#         temp = np.array([str(i)])
#         temp = np.reshape(temp, (-1,1))
#         categories = np.append(categories, temp, axis=0)
#     count += 1
#
# print(categories)
# print(categories.shape)
# np.save("labels.npy", categories)
# exit()


X_in = np.load("dog_breed_data.npy")
print("data")
print(X_in.shape)

y_in = np.load("dog_breed_labels.npy")
print("correct labels")
print(y_in.shape)

labels = np.load("labels.npy")
print("labels")
print(labels.shape)
labels = np.sort(labels, axis=0)


X_train, X_test, y_train, y_test = train_test_split(X_in, y_in, test_size=0.2, random_state=42)
print("done splitting")
LR = LogisticRegression(random_state=0, verbose=2, n_jobs=4, multi_class='multinomial', max_iter=2000)
LR.classes_ = labels
LR.fit(X_train, y_train.flatten())
print("done fitting")
print(LR.predict(X_test))
temp = LR.predict_proba(X_test)
print(temp)
print(list(temp[0]))
print(y_test)
score = LR.score(X_test, y_test)
print(score)
# yhat = LR.predict_proba(X_test)
# print("predicted")
# print(yhat)
# print("actual")
# print(y_test)
# print("done predicting")
#
# print("scoring")
# print(auc)

# out =
#
# np.savetxt("output.csv", out, delimiter=",")