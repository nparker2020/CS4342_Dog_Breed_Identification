from sklearn.linear_model import LogisticRegression
from PIL import Image, ImageOps
import os
import numpy as np
import matplotlib.pyplot as plt

# pathString = os.path.join("C:", os.sep, "Users", "Nicholas", "Documents", "CS4342", "CS4342_Dog_Breed_Identification", "resized")
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
# np.save("dog_breed_data.npy", X.T)

data = np.load("dog_breed_data.npy")
print("data")
print(data.shape)
# LR = LogisticRegression(random_state=0).fit(X, y)