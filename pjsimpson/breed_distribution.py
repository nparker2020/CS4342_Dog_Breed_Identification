import numpy as np
import pandas as pd
from skimage import io
from sklearn import preprocessing, linear_model
import warnings
from PIL import Image

def convert2DBins(images_df, binSize):
    image_dir = '/Users/parkersimpson/PycharmProjects/CS4342/DogBreedProject/dog-breed-identification/resized/'
    for i in range(images_df.shape[0]):
        label = images_df.breed.iloc[i]
        image_name = images_df.id.iloc[i]
        image = io.imread(image_dir+image_name+'.jpg') # (200,200,3)

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


binSizes = [32, 16, 8]
tr_frac = [(3,5), (13,20), (7,10), (3,4), (4,5), (17,20)] # [60/40, 65/45, 70/30, 75/25, 80/20, 85/15]
avg_scores = np.zeros((12, 1))
count = 0

for bs in binSizes:
    for frac in tr_frac:
        num, denom = frac
        iters = 200
        score_vector = np.zeros((iters, 1))
        for t in range(iters):
            # Create training and testing dataframes of top n breeds
            n = 2
            for i in range(n):
                breed_i = labels.loc[labels.breed == gr_labels.index[i]]
                tr_breed_i = breed_i.sample(n=num*breed_i.shape[0]//denom) #, random_state=1) # random 80% of breed
                te_breed_i = breed_i[~breed_i.isin(tr_breed_i)].dropna() # random 20% of breed

                if i == 0:
                    tr_breeds = tr_breed_i
                    te_breeds = te_breed_i
                else:
                    tr_breeds = pd.concat([tr_breeds, tr_breed_i])
                    te_breeds = pd.concat([te_breeds, te_breed_i])

            trainX, trainY = convert2DBins(tr_breeds, bs)
            testX, testY = convert2DBins(te_breeds, bs)

            # Standardize training data
            trainX, testX = map(lambda x: preprocessing.scale(x, axis=0), [trainX, testX])

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                # Sklearn logistic regressiom
                clf = linear_model.SGDClassifier(loss='log') #, random_state=25)
                clf.fit(trainX, trainY)
                scre = clf.score(testX, testY)
                score_vector[t,0] = scre

        avg = np.average(score_vector)
        avg_scores[count,0] = avg
        count += 1

print(avg_scores)
