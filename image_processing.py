import os
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
from keras.utils import img_to_array

# function for preprocessing images before feeding them into model
def preprocessing(image, target_size=(224, 224)):

    if isinstance(image, str):
        try: 
            # open the image file
            img = Image.open(image)

        except FileNotFoundError:
            raise FileNotFoundError(f"File not found.")
        except Exception as e:
            raise ValueError(f"Error occurred while opening the image")
        
    elif isinstance(image, Image.Image):
        # if image already a Pillow Image object, uses it directly
        img = image  

    else:
        raise ValueError("Unsupported Format")

    # resize the opened image 
    img = img.resize(target_size)

    # convert image to an array 
    img_array = img_to_array(img)  

    # normalize array of pixel values to [0, 1]
    img_array = img_array / 255.0  

    # expand image dimensions to add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array