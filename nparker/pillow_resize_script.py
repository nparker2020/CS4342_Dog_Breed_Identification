# noinspection PyUnresolvedReferences
from PIL import Image
import os

#This should point to the base directory of all the data:
pathString = os.path.join("c:", os.sep, "Users", "Noah Parker", "ML_Final_Project_Repo", "dog-breed-identification")
resized_directory = pathString + "\\resized\\";
resizeWidth = 200
resizeHeight = 200
limit = None;

#C:\Users\Noah Parker\ML_Final_Project_Repo\dog-breed-identification\test
count = 0;
for image_file in os.listdir(pathString + "\\test"):
    try:
        with Image.open(pathString+"\\test\\"+image_file) as im:
            resized = im.resize((resizeWidth, resizeHeight));
            resized.save(resized_directory+str(count)+".jpg")
            count = count + 1;
    except IOError as e:
        print("Error resizing image: ", e)

    if limit is not None and (count >= limit):
        break

print("Done! ", count," images resized to ", resizeWidth, "x", resizeHeight)
print("Saved to: ", resized_directory)




