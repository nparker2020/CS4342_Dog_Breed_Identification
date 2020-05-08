# Code from: https://towardsdatascience.com/histograms-in-image-processing-with-skimage-python-be5938962935

from skimage import io
import matplotlib.pyplot as plt

# Plot one image rgb histogram (use )
test_image = io.imread('/Users/parkersimpson/PycharmProjects/CS4342/DogBreedProject/dog-breed-identification/resized'
                       '/0a0c223352985ec154fd604d7ddceabd.jpg')

# Histogram for total pixel values
plt.hist(test_image.ravel(), bins=int(200/20), color='orange')

# Histogram for red collor channel
plt.hist(test_image[:,:,0].ravel(), bins=int(200/20), color='red', alpha=0.5)

# Histogram for green channel
plt.hist(test_image[:,:,1].ravel(), bins=int(200/20), color='Green', alpha=0.5)

# Histogram for blue channel
plt.hist(test_image[:,:,2].ravel(), bins=int(200/20), color='Blue', alpha=0.5)

plt.xlabel('Intensity Value')
plt.ylabel('Count')
plt.legend(['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])
plt.show()