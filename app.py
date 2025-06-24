from flask import Flask, request, render_template
from ultralytics import YOLO
import cv2
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
import pandas as pd
import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from PIL import Image, ImageFile
# Allow truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads/"
RESULT_FOLDER = "static/results/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

#start
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model("BONE NET V2.h5")
print(model.input_shape)
print(model.summary())

loaded_model = tf.keras.models.load_model('fracture_classification_model.h5')
# Define class labels (Ensure this matches the original class order)
class_names = [
    "Avulsion fracture", "Comminuted fracture", "Fracture Dislocation",
    "Greenstick fracture", "Hairline Fracture", "Impacted fracture",
    "Longitudinal fracture", "Oblique fracture", "Pathological fracture",
    "Spiral Fracture"
]


def load_preprocessed_image(image_path, img_height=180, img_width=180):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((img_height, img_width))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict a new image
def predict_image(image_path):
    img_size = (256, 256)  # Change to match model input size
    img = image.load_img(image_path, target_size=img_size)  # Resize correctly
    img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)
    if class_index==0:
        pres='''A small piece of bone is pulled away by a tendon or ligament due to sudden force. It commonly occurs in joints like the ankle, knee, or hip. This type of fracture is often seen in athletes.  

            First Aid
            - *Keep the injured area immobilized using a splint or bandage* 🩹  
            - *Apply ice packs wrapped in a cloth to reduce swelling* ❄️  
            - *Avoid moving the affected joint or limb* 🚫  
            - *Keep the injured limb elevated to reduce swelling* 🔼  
            - *Use painkillers if prescribed by a doctor* 💊  
            - *Seek immediate medical attention to prevent further damage* 🚑  

            ⚠️ *Warning:* Painkillers should only be taken if not allergic and under medical supervision. Always seek professional medical care'
            '''
    elif class_index==1:
        pres='''
            The bone breaks into three or more fragments due to high-impact trauma, such as a car accident or severe fall. It is a complex fracture that requires surgical intervention.  

            #### *First Aid*  
            - *Do not move the affected limb unnecessarily* 🚫  
            - *Use a firm support (splint or bandage) to stabilize the injury* 🩹  
            - *Apply cold packs to reduce pain and swelling* ❄️  
            - *Keep the injured limb elevated if possible* 🔼  
            - *Use painkillers if prescribed by a doctor* 💊  
            - *Call for emergency medical help immediately* 🚑  

            ⚠️ *Warning:* Painkillers should only be taken if not allergic and under medical supervision. Always seek professional medical care.  

            '''
    elif class_index==2:
        pres='''
            ### *3. Fracture Dislocation*  
            A combination of a broken bone and a joint dislocation, usually caused by extreme force. The fracture often occurs near joints, making treatment more complicated.  

            #### *First Aid*  
            - *Do not attempt to relocate the joint or push the bone back* 🚫  
            - *Immobilize the limb using a splint or sling* 🩹  
            - *Apply ice packs to minimize swelling and pain* ❄️  
            - *Keep the injured area elevated if possible* 🔼  
            - *Use painkillers if prescribed by a doctor* 💊  
            - *Seek immediate medical attention* 🚑  

            ⚠️ *Warning:* Painkillers should only be taken if not allergic and under medical supervision. Always seek professional medical care.  

            '''

    elif class_index==3:
        pres='''
            Greenstick Fracture*  
            A partial fracture where the bone bends and cracks without breaking completely. It is most common in children due to their softer, more flexible bones.  

            #### *First Aid*  
            - *Keep the injured limb still to prevent further injury* 🚫  
            - *Use soft padding to support the affected area* 🩹  
            - *Apply ice packs to reduce swelling* ❄️  
            - *Avoid putting weight on the injured limb* 🚫  
            - *Use painkillers if prescribed by a doctor* 💊  
            - *Consult a doctor for proper evaluation and treatment* 🚑  

            ⚠️ *Warning:* Painkillers should only be taken if not allergic and under medical supervision. Always seek professional medical care.  

            '''

    elif class_index==4:
        pres='''
            A small, thin crack in the bone that often develops due to repetitive stress or overuse. It is common in athletes and runners.  

            #### *First Aid*  
            - *Limit movement and avoid putting weight on the injured area* 🚫  
            - *Use crutches or a brace for support if needed* 🩹  
            - *Apply ice packs to reduce swelling and pain* ❄️  
            - *Elevate the limb to minimize swelling* 🔼  
            - *Use painkillers if prescribed by a doctor* 💊  
            - *Seek medical advice for proper diagnosis* 🚑  

            ⚠️ *Warning:* Painkillers should only be taken if not allergic and under medical supervision. Always seek professional medical care.  

            '''

    elif class_index==5:
        pres='''
            One broken piece of bone is forced into another due to strong pressure or force, such as during falls from heights.  

            #### *First Aid*  
            - *Immobilize the injured limb using a splint* 🩹  
            - *Avoid any unnecessary movement* 🚫  
            - *Apply ice packs carefully to reduce swelling* ❄️  
            - *Keep the limb elevated when possible* 🔼  
            - *Use painkillers if prescribed by a doctor* 💊  
            - *Get urgent medical care* 🚑  

            ⚠️ *Warning:* Painkillers should only be taken if not allergic and under medical supervision. Always seek professional medical care.  

            '''

    elif class_index==6:
        pres='''
            A fracture that runs along the length of the bone, often caused by direct impact or twisting forces.  

            #### *First Aid*  
            - *Support the affected limb with a splint or brace* 🩹  
            - *Do not try to realign the bone* 🚫  
            - *Apply ice to control swelling and pain* ❄️  
            - *Keep the limb elevated to prevent excessive swelling* 🔼  
            - *Use painkillers if prescribed by a doctor* 💊  
            - *Visit a hospital as soon as possible* 🚑  

            ⚠️ *Warning:* Painkillers should only be taken if not allergic and under medical supervision. Always seek professional medical care.  

            '''

    elif class_index==7:
        pres='''
            A diagonal break across the bone, usually caused by a sharp angled force, such as twisting or a direct blow.  

            #### *First Aid*  
            - *Keep the fractured limb stable using a splint* 🩹  
            - *Do not apply pressure to the injury* 🚫  
            - *Ice packs can help with swelling* ❄️  
            - *Elevate the limb to reduce discomfort* 🔼  
            - *Use painkillers if prescribed by a doctor* 💊  
            - *Immediate medical attention is necessary* 🚑  

            ⚠️ *Warning:* Painkillers should only be taken if not allergic and under medical supervision. Always seek professional medical care.  

            '''

    elif class_index==8:
        pres='''
            A fracture caused by weakened bones due to conditions like osteoporosis, cancer, or infections.  

            #### *First Aid*  
            - *Avoid putting pressure on the fractured bone* 🚫  
            - *Use a support like a sling or splint* 🩹  
            - *Apply ice to manage pain and swelling* ❄️  
            - *Keep the affected area elevated if possible* 🔼  
            - *Use painkillers if prescribed by a doctor* 💊  
            - *Seek medical attention for further diagnosis and treatment* 🚑  

            ⚠️ *Warning:* Painkillers should only be taken if not allergic and under medical supervision. Always seek professional medical care.  

            '''

    elif class_index==9:
        pres='''
            A fracture that occurs when the bone is twisted forcefully, creating a spiral-shaped break. It is common in sports injuries.  

            #### *First Aid*  
            - *Keep the injured limb stable and immobilized* 🩹  
            - *Do not attempt to move the broken bone* 🚫  
            - *Apply ice to manage swelling and pain* ❄️  
            - *Elevate the limb gently to reduce discomfort* 🔼  
            - *Use painkillers if prescribed by a doctor* 💊  
            - *Get immediate medical evaluation* 🚑  

            ⚠️ *Warning:* Painkillers should only be taken if not allergic and under medical supervision. Always seek professional medical care.

            '''
    else:
        pres="No Fracture"
    
    pres_lines = pres.strip().split('\n')
    print(pres_lines)
    return class_names[class_index],pres_lines, confidence


# Provide an image path (Change this to your test image path)

#end


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded!", 400
        
        file = request.files["file"]
        if file.filename == "":
            return "No selected file!", 400

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        test_image_path = "static/uploads/"+file.filename

        #image_path = '/content/10-rotated1-rotated1-rotated1-rotated1.jpg'
        image_path =test_image_path
        preprocessed_img = load_preprocessed_image(image_path)
        predictions = loaded_model.predict(preprocessed_img)
        predicted_class = np.argmax(predictions, axis=1)
        class_names = ['fractured', 'not fractured']
        predicted_class = (predictions > 0.5).astype("int32")
        print(f"Predicted class: {class_names[predicted_class[0][0]]}")

        if predicted_class[0][0]==0:
        
            predicted_label, res,confidence = predict_image(test_image_path)

            print(f"Predicted Class: {predicted_label}, Confidence: {confidence:.4f}")

            print(file.filename)
            #result_image = os.path.join(RESULT_FOLDER, "predict", file.filename)

            return render_template("predict.html", pres=res,image=predicted_label,fname=file.filename)
        else:
            return render_template("predict.html", pres="",image="No Fracture",fname=file.filename)
    return render_template("index.html")
if __name__ == "__main__":
    app.run(debug=True)
