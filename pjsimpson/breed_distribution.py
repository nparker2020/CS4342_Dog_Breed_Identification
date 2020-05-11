import numpy as np
import pandas as pd
from skimage import io
from sklearn import preprocessing, linear_model
import warnings
import skimage.transform as tf
import os
import matplotlib.pyplot as plt
import time

def convert2DBins(images_df, binSize, flip=False, translate=False):
    image_dir = '/Users/parkersimpson/PycharmProjects/CS4342/DogBreedProject/dog-breed-identification/resized/'
    for i in range(images_df.shape[0]):
        # if i == 2:
        #     break

        label = images_df.breed.iloc[i]
        image_name = images_df.id.iloc[i]
        image = io.imread(image_dir+image_name+'.jpg') # (200,200,3)
        if flip:
            image = np.fliplr(image)
        if translate:
            x, y = 10, 10
            trans_locs = [(x, y), (-x, y), (-x, -y), (x, -y), (x, 0), (-x, 0), (0, y), (0, -y)]
            trans = np.random.randint(0, len(trans_locs), 1)[0]
            trans_mx = tf.EuclideanTransform(translation=trans_locs[trans])
            image = tf.warp(image, trans_mx)

        binNum = int(256/binSize)
        image_bin = np.zeros((binNum, 3))
        for j in range(binNum):
            count1 = np.sum((image[:,:,0] >= (j*binSize)) * (image[:,:,0] < binSize+(j*binSize)))
            count2 = np.sum((image[:,:,1] >= (j*binSize)) * (image[:,:,1] < binSize+(j*binSize)))
            count3 = np.sum((image[:,:,2] >= (j*binSize)) * (image[:,:,2] < binSize+(j*binSize)))

            image_bin[j,:] = [count1, count2, count3]

        image_bin = image_bin.ravel()

        if i == 0:
            X = image_bin[None,:]
            Y = np.array([label])
        else:
            X = np.concatenate((X, image_bin[None, :]), axis=0)
            Y = np.concatenate((Y, np.array([label])), axis=0)

    # print(X.shape, Y.shape, f'Class 1:', len(Y[Y == 'scottish_deerhound']))
    return X, Y


def convert3DBins(images_df, binSize, flip=False, translate=False):
    image_dir = '/Users/parkersimpson/PycharmProjects/CS4342/DogBreedProject/dog-breed-identification/resized/'
    for i in range(images_df.shape[0]):
        # if i == 2:
        #     break

        label = images_df.breed.iloc[i]
        image_name = images_df.id.iloc[i]
        image = io.imread(image_dir+image_name+'.jpg') # (200,200,3)
        if flip:
            image = np.fliplr(image)
        if translate:
            x, y = 10, 10
            trans_locs = [(x, y), (-x, y), (-x, -y), (x, -y), (x, 0), (-x, 0), (0, y), (0, -y)]
            trans = np.random.randint(0, len(trans_locs), 1)[0]
            trans_mx = tf.EuclideanTransform(translation=trans_locs[trans])
            image = tf.warp(image, trans_mx)

        binNum = int(256/binSize)
        image_bin = np.zeros((binNum, binNum, binNum))
        for j in range(binNum):
            for k in range(binNum):
                for l in range(binNum):
                    r_bin = (image[:, :, 0] >= (j * binSize)) * (image[:, :, 0] < binSize + (j * binSize))
                    g_bin = (image[:, :, 1] >= (k * binSize)) * (image[:, :, 1] < binSize + (k * binSize))
                    b_bin = (image[:, :, 2] >= (l * binSize)) * (image[:, :, 2] < binSize + (l * binSize))

                    rgb_bin = r_bin * g_bin * b_bin
                    image_bin[j,k,l] = np.sum(rgb_bin)

        image_bin = image_bin.ravel()

        if i == 0:
            X = image_bin[None, :]
            Y = np.array([label])
        else:
            X = np.concatenate((X, image_bin[None, :]), axis=0)
            Y = np.concatenate((Y, np.array([label])), axis=0)

    # print(X.shape, Y.shape, 'Class 1:', len(Y[Y == 'scottish_deerhound']))
    return X, Y

# Load Labels
labels = pd.read_csv('/Users/parkersimpson/PycharmProjects/CS4342/DogBreedProject/dog-breed-identification/labels.csv')

unique_breeds = labels.breed.unique() # 120

# Group labels by breed in ascending order
gr_labels = labels.groupby('breed').count()
gr_labels = gr_labels.rename(columns = {"id" : "count"})
gr_labels = gr_labels.sort_values("count", ascending=False)

binSizes = [8, 16, 32]

saved = True
if not saved:
    n = 2
    for i in range(n):
        breed_i = labels.loc[labels.breed == gr_labels.index[i]]

        if i == 0:
            breeds = breed_i
        else:
            breeds = pd.concat([breeds, breed_i])

    for bsize in binSizes:
        binNum = int(256/bsize)

        binned2_og_data, binned2_og_labels = convert2DBins(breeds, bsize)
        binned2_flip_data, binned2_flip_labels = convert2DBins(breeds, bsize, flip=True)
        binned2_translate_data, binned2_translate_labels = convert2DBins(breeds, bsize, translate=True)

        binned2_data = np.vstack((binned2_og_data, binned2_flip_data, binned2_translate_data))
        binned2_labels = np.hstack((binned2_og_labels, binned2_flip_labels, binned2_translate_labels))
        # print(binned2_data.shape, binned2_labels.shape)

        binned3_og_data, binned3_og_labels = convert3DBins(breeds, bsize)
        binned3_flip_data, binned3_flip_labels = convert3DBins(breeds, bsize, flip=True)
        binned3_translate_data, binned3_translate_labels = convert3DBins(breeds, bsize, translate=True)

        binned3_data = np.vstack((binned3_og_data, binned3_flip_data, binned3_translate_data))
        binned3_labels = np.hstack((binned3_og_labels, binned3_flip_labels, binned3_translate_labels))
        # print(binned3_data.shape, binned3_labels.shape)

        np.save(f'/Users/parkersimpson/PycharmProjects/CS4342/DogBreedProject/CS4342_Dog_Breed_Identification/'
                f'pjsimpson/binned-data/{binNum}_binned2D_data.npy', binned2_data)
        np.save(f'/Users/parkersimpson/PycharmProjects/CS4342/DogBreedProject/CS4342_Dog_Breed_Identification/'
                f'pjsimpson/binned-data/{binNum}_binned2D_labels.npy', binned2_labels)
        np.save(f'/Users/parkersimpson/PycharmProjects/CS4342/DogBreedProject/CS4342_Dog_Breed_Identification/'
                f'pjsimpson/binned-data/{binNum}_binned3D_data.npy', binned3_data)
        np.save(f'/Users/parkersimpson/PycharmProjects/CS4342/DogBreedProject/CS4342_Dog_Breed_Identification/'
                f'pjsimpson/binned-data/{binNum}_binned3D_labels.npy', binned3_labels)

        print(f'Finished binNum = {binNum}')

tr_frac = [(3,5), (13,20), (7,10), (3,4), (4,5), (17,20)] # [60/40, 65/45, 70/30, 75/25, 80/20, 85/15]
avg_scores = np.zeros((len(binSizes)*len(tr_frac), 2))
count = 0

start = time.time()
for bs in binSizes:
    data_2D = np.load(f'binned-data/{bs}_binned2D_data.npy')
    labels_2D = np.load(f'binned-data/{bs}_binned2D_labels.npy')
    data_3D = np.load(f'binned-data/{bs}_binned3D_data.npy')
    labels_3D = np.load(f'binned-data/{bs}_binned3D_labels.npy')
    for frac in tr_frac:
        num, denom = frac
        iters = 200
        score_vector = np.zeros((iters, 2))
        for t in range(iters):
            indices = np.arange(labels_2D.shape[0])
            np.random.shuffle(indices)
            cutoff = num*labels_2D.shape[0] // denom

            trainX_2D = data_2D[indices[:cutoff],:]
            trainY_2D = labels_2D[indices[:cutoff]]
            testX_2D = data_2D[indices[cutoff:],:]
            testY_2D = labels_2D[indices[cutoff:]]

            trainX_3D = data_3D[indices[:cutoff],:]
            trainY_3D = labels_3D[indices[:cutoff]]
            testX_3D = data_3D[indices[cutoff:],:]
            testY_3D = labels_3D[indices[cutoff:]]

            # Standardize training data
            trainX_2D, testX_2D = map(lambda x: preprocessing.scale(x, axis=0), [trainX_2D, testX_2D])
            trainX_3D, testX_3D = map(lambda x: preprocessing.scale(x, axis=0), [trainX_3D, testX_3D])

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                # Sklearn logistic regressiom
                clf_2D = linear_model.SGDClassifier(loss='log') #, random_state=25)
                clf_2D.fit(trainX_2D, trainY_2D)
                scre2 = clf_2D.score(testX_2D, testY_2D)
                score_vector[t,0] = scre2

                clf_3D = linear_model.SGDClassifier(loss='log')  # , random_state=25)
                clf_3D.fit(trainX_3D, trainY_3D)
                scre3 = clf_3D.score(testX_3D, testY_3D)
                score_vector[t, 1] = scre3

        avg = np.average(score_vector,axis=0)
        print(avg)
        avg_scores[count,:] = avg
        count += 1

end = time.time()
print((end-start)/60)
print(avg_scores)
