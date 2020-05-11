from skimage import io
import numpy as np
import os



# Image bin transformation
count = 0
images_dir = '/Users/parkersimpson/PycharmProjects/CS4342/DogBreedProject/dog-breed-identification/resized'
for image_file in os.listdir(images_dir):
    if count > 0:
        break
    count += 1

    image = io.imread(os.path.join(images_dir, image_file)) # (200,200,3)
    image = np.reshape(image, (-1,3)) # (40000, 3)
    binSize = 200/20
    print(image.shape)

