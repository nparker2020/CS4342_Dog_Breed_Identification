import numpy as np
import pandas as pd
from skimage import io
from sklearn import preprocessing, linear_model
from PIL import Image

def convert2DBins(images_df):
    image_dir = '/Users/parkersimpson/PycharmProjects/CS4342/DogBreedProject/dog-breed-identification/resized/'
    for i in range(images_df.shape[0]):
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

        if i == 0:
            X = image_bin[None,:]
            Y = np.array([label])
        else:
            X = np.concatenate((X, image_bin[None, :]), axis=0)
            Y = np.concatenate((Y, np.array([label])), axis=0)

    # print(X.shape, Y.shape, f'Class 1:', len(Y[Y == 'scottish_deerhound']))
    return X, Y

# Load Labels
labels = pd.read_csv('/Users/parkersimpson/PycharmProjects/CS4342/DogBreedProject/dog-breed-identification/labels.csv')

unique_breeds = labels.breed.unique() # 120

# Group labels by breed in ascending order
gr_labels = labels.groupby('breed').count()
gr_labels = gr_labels.rename(columns = {"id" : "count"})
gr_labels = gr_labels.sort_values("count", ascending=False)

score_vector = np.zeros((100, 1))
for t in range(100):
    # Create training and testing dataframes of top n breeds
    n = 2
    for i in range(n):
        breed_i = labels.loc[labels.breed == gr_labels.index[i]]
        tr_breed_i = breed_i.sample(n=17*breed_i.shape[0]//20) #, random_state=1) # random 80% of breed
        te_breed_i = breed_i[~breed_i.isin(tr_breed_i)].dropna() # random 20% of breed

        if i == 0:
            tr_breeds = tr_breed_i
            te_breeds = te_breed_i
        else:
            tr_breeds = pd.concat([tr_breeds, tr_breed_i])
            te_breeds = pd.concat([te_breeds, te_breed_i])

    trainX, trainY = convert2DBins(tr_breeds) # (24, 193) (193,)
    testX, testY = convert2DBins(te_breeds) # (24, 50) (50,)

    # Standardize training data
    trainX, testX = map(lambda x: preprocessing.scale(x, axis=0), [trainX, testX])

    # Sklearn logistic regressiom
    clf = linear_model.SGDClassifier(loss='log') #, random_state=25)
    clf.fit(trainX, trainY)
    scre = clf.score(testX, testY)
    score_vector[t,0] = scre

print(np.average(score_vector))

