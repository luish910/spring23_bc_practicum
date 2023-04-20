import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import rescale, resize
import os

#Function to read image from file or url to an array (as per library documentation)
def upload_image(id, data: None, path = None):
    #url
    if path == None:
        image_url = data[data["ID"]==str(id)]['Image URL'].item()
        image = io.imread(image_url)
    #image
    elif data == None:
        full_path = os.path.join(path, str(id) + ".jpg")
        image = plt.imread(full_path) 
    return image

#Function to plot an image 
def show_image(image):    
    plt.imshow(image)
    plt.axis('off')
    plt.show() 

#function to downsize an image. returns scaled rgb values
def resize_image(image, width, height): 
    resized_image = resize(image, (width, height), anti_aliasing=True)
    return resized_image 

def rescale_image(image, factor): 
    rescaled_image = rescale(image, factor, anti_aliasing=True, channel_axis=2)
    return rescaled_image 

#function to save an image to a path
def save_image(path,id,image): 
    full_path = os.path.join(path, str(id) + ".jpg")
    plt.imsave(full_path,image)
    
#function to convert matrix image dimensions to make 1 row per pipxel
def reshape_image(image):
    reshaped_image = image.reshape(-1,3)  
    return reshaped_image

df = pd.read_csv("WikiArt5Colors.csv")
directory = os.getcwd()
image_path = os.path.join(directory, "images")
resized_path = os.path.join(directory, "resized-images")
display_path = os.path.join(directory, "display-images")

#Download all dataset images from web
for image_id in df["ID"]:
    #print(image_id)
    uploaded_image = upload_image(image_id, data = df) 
    save_image(image_path, image_id, uploaded_image)
print("saved",len(df["ID"]) ," images to ",image_path)

#resize all images for clustering pixels
for image_id in df["ID"]:
    uploaded_image = upload_image(image_id,data=None,path=image_path) 
    resized_image = resize_image(uploaded_image,100,100)
    save_image(resized_path,image_id,resized_image)
print("saved",len(df["ID"]) ," images to ",resized_path)

#resize all images for for front-end display (dynamic factor)
for image_id in df["ID"]:
    uploaded_image = upload_image(image_id,data=None,path=image_path)
    factor = 1/(max(uploaded_image.shape[0:2])/200)
    rescaled_image = rescale_image(uploaded_image,factor)
    save_image(display_path,image_id,rescaled_image)
print("saved",len(df["ID"]) ," images to ",display_path)