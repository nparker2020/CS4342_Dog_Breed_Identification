import numpy as np
import pandas as pd
from skimage import io
from PIL import Image


def convert2DBins(images_df):
    image_dir = '/Users/parkersimpson/PycharmProjects/CS4342/DogBreedProject/dog-breed-identification/resized/'
    for i in range(images_df.shape[0]):
        # if i == 2:
        #     break

        label = images_df.breed.iloc[i]
        image_name = images_df.id.iloc[i]
        image = io.imread(image_dir+image_name+'.jpg') # (200,200,3)

        binSize = 32
        binNum = int(256/binSize)
        image_bin = np.zeros((binNum, 3))
        for j in range(binNum):
            count1 = np.sum((image[:,:,0] >= (j*binSize)) * (image[:,:,0] < binSize+(j*binSize)))
            count2 = np.sum((image[:,:,1] >= (j*binSize)) * (image[:,:,1] < binSize+(j*binSize)))
            count3 = np.sum((image[:,:,2] >= (j*binSize)) * (image[:,:,2] < binSize+(j*binSize)))

            image_bin[j,:] = [count1, count2, count3]

        image_bin = image_bin.ravel()
        # print(image_bin.shape)
        if i == 0:
            X = image_bin[:,None]
            Y = np.array([label])
        else:
            X = np.concatenate((X, image_bin[:,None]), axis=1)
            Y = np.concatenate((Y, np.array([label])), axis=0)

    print(X.shape, Y.shape)
    return X, Y

# Load Labels
labels = pd.read_csv('/Users/parkersimpson/PycharmProjects/CS4342/DogBreedProject/dog-breed-identification/labels.csv')

unique_breeds = labels.breed.unique() # 120

# Group labels by breed in ascending order
gr_labels = labels.groupby('breed').count()
gr_labels = gr_labels.rename(columns = {"id" : "count"})
gr_labels = gr_labels.sort_values("count", ascending=False)

# Create training and testing dataframes of top n breeds
n = 2
for i in range(n):
    breed_i = labels.loc[labels.breed == gr_labels.index[i]]
    tr_breed_i = breed_i.sample(n=4*breed_i.shape[0]//5, random_state=1) # random 80% of breed
    te_breed_i = breed_i[~breed_i.isin(tr_breed_i)].dropna() # random 20% of breed
    # print(tr_breed_i.shape, te_breed_i.shape)

    if i == 0:
        tr_breeds = tr_breed_i
        te_breeds = te_breed_i
    else:
        tr_breeds = pd.concat([tr_breeds, tr_breed_i])
        te_breeds = pd.concat([te_breeds, te_breed_i])


trainX, trainY = convert2DBins(tr_breeds)
testX, testY = convert2DBins(te_breeds)
