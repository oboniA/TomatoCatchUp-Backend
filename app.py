from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import json
from dotenv import load_dotenv
from pymongo import MongoClient
import os
from PIL import Image
from image_processing import preprocessing 
from pymongo import MongoClient


# Flask activation
app = Flask(__name__)

# TODO: 
# pre-trained custom CNN model
# PLEASE DOWNLOAD "Experimental_trial_26_model.keras" From the provided Google Drive Link on README.md
# OR from here: 
custom_cnn_model = tf.keras.models.load_model("Experimental_trial_26_model.keras")

# load fallback data from local JSON file
with open("fallbacks.json", "r", encoding="utf-8") as f:
    fallback = json.load(f)

# connect to MongoDB Atlas
try: 
    load_dotenv()
    mongodb_uri = os.getenv("MongoDB_URI")
    client = MongoClient(mongodb_uri)
    db = client['tomato_plants']
    collection = db['tomato_plant_diseases']
    print("Connected to MongoDB")
except Exception as e:
    print(f"Error: {e}")

# class labels (folder names) in ascending order of label index (from CNN model training)
class_labels = {0: "Tomato_Bacterial_spot", 
                1: "Tomato_Early_blight", 
                2: "Tomato_Late_blight", 
                3: "Tomato_Leaf_Mold", 
                4: "Tomato_Septoria_leaf_spot",
                5: "Tomato_Spider_mites_Two_spotted_spider_mite",
                6: "Tomato__Target_Spot",
                7: "Tomato__Tomato_YellowLeaf__Curl_Virus",
                8: "Tomato__Tomato_mosaic_virus",
                9: "Tomato_healthy"
                }

# homepage route
@app.route('/')
def index():
    print("Homepage accessed")
    return render_template('index.html')

# HTTP method to send (upload) testing image to server: POST
@app.route('/classify', methods=['POST'])
def classify():
    print("Classification route accessed")

    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    # fetch uploaded image file
    file = request.files['image']  

    if file:
        print("File received")

        # open image file
        image = Image.open(file)

        # preprocess image file
        processed_image = preprocessing(image)  # preprocessing function of image_preprocessing.py
        print("Processed image shape:", processed_image.shape)  

        # classification using custom CNN model
        prediction = custom_cnn_model.predict(processed_image)

        # prediction on class label
        predicted_index = np.argmax(prediction, axis=1)[0]
        label = class_labels[predicted_index]
        print("Leaf Health:", label)

        # predicted disease details from MongoDB
        disease_details = collection.find_one({"name": label})
        print("Disease details from DB:", disease_details)
        
        # set-up output from MongoDB Atlas database; fallbacks from .json file
        if disease_details:
            description = disease_details.get("description", fallback["description"])
            symptoms = disease_details.get("symptoms", fallback["symptoms"])
            prevention = disease_details.get("prevention", fallback["prevention"])
            treatment = disease_details.get("treatment", fallback["treatment"])
            read_more = disease_details.get("read_more", None)
        else:
           description = fallback["description"]
           symptoms = fallback["symptoms"]
           prevention = fallback["prevention"]
           treatment = fallback["treatment"]
           read_more = None
        
        result = {
            'prediction': label,
            'description': description,
            'symptoms': symptoms,
            'prevention': prevention,
            'treatment': treatment,
            'read_more': read_more
        }
        return jsonify(result)
       
    return jsonify({'error': 'No file uploaded/classification failed'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)