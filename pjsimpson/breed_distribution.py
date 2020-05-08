import numpy as np
import pandas as pd
from skimage import io
from PIL import Image


def convert2DBins(images_df):
    image_dir = '/Users/parkersimpson/PycharmProjects/CS4342/DogBreedProject/dog-breed-identification/resized/'
    for i in range(images_df.shape[0]):
        if i == 1:
            break

        label = images_df.breed.iloc[i]
        image_name = images_df.id.iloc[i]
        image = io.imread(image_dir+image_name+'.jpg') # (200,200,3)

        binSize = int(256/32).
        image_bin = np.zeros((binSize, 1))
        for j in range(binSize):
            bin1 = (image[:,:,0] >= (j*binSize)) * (image[:,:,0] < binSize+(j*binSize))
            count1 = np.sum(bin1)
            print(count1)
            # count2 = np.sum(np.nonzero(image[:,:,1] >= (j*binSize) * image[:,:,1] < binSize+(j*binSize)))
            # count3 = np.sum(np.nonzero(image[:,:,2] >= (j*binSize) * image[:,:,2] < binSize+(j*binSize)))

            image_bin[j,:] = [count1] #, count2, count3]
        print(np.sum(image_bin))


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
    tr_breed_i = breed_i.sample(n=4*breed_i.shape[0]//5) # random 80% of breed
    te_breed_i = breed_i[~breed_i.isin(tr_breed_i)].dropna() # random 20% of breed
    # print(tr_breed_i.shape, te_breed_i.shape)

    if i == 0:
        tr_breeds = tr_breed_i
        te_breeds = te_breed_i
    else:
        tr_breeds = pd.concat([tr_breeds, tr_breed_i])
        te_breeds = pd.concat([te_breeds, te_breed_i])

# print(tr_breeds.shape, te_breeds.shape)
# print(tr_breeds.breed.iloc[0])
convert2DBins(tr_breeds)
