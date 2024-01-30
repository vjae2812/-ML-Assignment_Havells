from flask import Flask, request, render_template,jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
import numpy as np
app = Flask(__name__)

# Load the trained model on zomato and others
model_predict = tf.keras.models.load_model('model/mobilenet_custom_model_delivery.h5')

# Load your pre-trained model for human or not
# model = tf.keras.applications.MobileNetV2(weights='imagenet')
model = tf.keras.models.load_model('model/mobilenet_custom_model_human_detection.h5')

def preprocess_image_human(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

# Function to preprocess the image for your model
def preprocess_image_class(img):
    #img = image.load_img(image, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'photo' in request.files:
        photo = request.files['photo']

        image = Image.open(photo.stream).convert('RGB')
        preprocessed_image = preprocess_image_human(image)
        #preprocessed_image = preprocess_image_human(image)

        # Make prediction
        preds = model.predict(preprocessed_image)
        # Interpret the prediction
        class_labels = ['human', 'others']  # Assuming 'others' is class 0 and 'zomato' is class 1
        predicted_class = np.argmax(preds)
        predicted_label = class_labels[preds]
        if predicted_label== 'human:
            # detect_image = preprocess_image_class(image)
            # Make prediction
            predictions = model_predict.predict(preprocessed_image)
    
            # Interpret the prediction
            class_labels = ['others', 'zomato']  # Assuming 'others' is class 0 and 'zomato' is class 1
            predicted_p = np.argmax(predictions)
            predicted_label_p = class_labels[predicted_p]
    
            # Print the prediction
            print(f'The predicted class is: {predicted_label_p}')
            if predicted_label_p == 'zomato'
                return jsonify({"message": "Delivery guy Detected"})
            else:
                return jsonify({"message": "No Delivery Detected"})
            
        else:
            return jsonify({"message": "No Human Detected"})
        
        # Save or process the photo here
        # For example, save to a directory (make sure it exists)
        photo.save('uploads/' + photo.filename)
        # return 'Photo uploaded successfully!'

if __name__ == '__main__':
    app.run()
