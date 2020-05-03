import csv
import os
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform as tf


pathString = '/Users/parkersimpson/PycharmProjects/CS4342/DogBreedProject/dog-breed-identification/resized'
count = 0
X = 0
flipped = True
rotate = False
translate = True
for image_file in os.listdir(pathString):
    try:
        temp_img = Image.open(os.path.join(pathString, image_file))
        temp_img = ImageOps.grayscale(temp_img)
        temp_img = np.asarray(temp_img)
        if flipped:
            flip_img = np.fliplr(temp_img)
        if rotate:
            rot_img = tf.rotate(temp_img, 7)
        if translate:
            x,y = 10,10
            trans_locs = [(x,y), (-x,y), (-x,-y), (x,-y), (x,0), (-x, 0), (0,y), (0,-y)]
            trans = np.random.randint(0, len(trans_locs), 1)[0]
            trans_mx = tf.EuclideanTransform(translation=trans_locs[trans])
            trans_img = tf.warp(temp_img, trans_mx)

        # fig, axes = plt.subplots(nrows=1, ncols=2)
        # axes[0].imshow(temp_img)
        # axes[1].imshow(flip_img)
        # axes[2].imshow(rot_img)
        # axes[1].imshow(trans_img)
        # plt.show()
        temp_img = temp_img.reshape((-1, 1))
        if flipped:
            flip_img = flip_img.reshape((-1, 1))
        if translate:
            trans_img = trans_img.reshape((-1, 1))
        if rotate:
            rot_img = rot_img.reshape((-1,1))

        if count == 0:
            X = temp_img
            if flipped:
                Y = flip_img
            if translate:
                Z = trans_img
            if rotate:
                W = rot_img
        else:
            X = np.append(X, temp_img, axis=1)
            if flipped:
                Y = np.append(Y, flip_img, axis=1)
            if translate:
                Z = np.append(Z, trans_img, axis=1)
            if rotate:
                W = np.append(W, rot_img, axis=1)

    except:
        print("error opening image: " + str(image_file))

    # if count > 1:
    #     break
    if count % 1000 == 0:
        print(count)
    count += 1

print("X.T shape:", X.T.shape)
if flipped:
    print("Y.T shape:", Y.T.shape)
if translate:
    print("Z.T shape:", Z.T.shape)
if rotate:
    print("W.T shape:", W.T.shape)

comb = np.hstack((X,Y,Z))
print('comb.T shape:', comb.T.shape)

# np.save("dog_breed_data.npy", X.T)
np.save('dog_breed_og_flp_trans_data.npy', comb.T)

mydict = {}
with open("/Users/parkersimpson/PycharmProjects/CS4342/DogBreedProject/dog-breed-identification/labels.csv", mode='r') as infile:
    reader = csv.reader(infile)
    mydict = {rows[0]: rows[1] for rows in reader}

# print(mydict)
count = 0
y = 0
for image_file in os.listdir(pathString):
    if count == 0:
        y = np.array([mydict[image_file.split('.')[0]]])
        y = np.reshape(y, (-1, 1))
    else:
        temp = np.array([mydict[image_file.split('.')[0]]])
        temp = np.reshape(temp, (-1, 1))
        # print(temp.shape)
        # print(temp)
        y = np.append(y, temp, axis=0)

    # if count > 1:
    #     break

    count += 1

y = np.vstack((y,y,y))

print(y.shape)
print(y[:5])
np.save("dog_breed_og_flp_trans_labels.npy", y)