# import other utils
import sys

# load Keras
from keras.models import load_model
from keras.preprocessing import image
#import ImageDataGenerator,array_to_img,img_to_array,load_img

# load Flask 
from flask import Flask,request
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/mnist_model_keras_collab.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()
print('model_loaded')
        

# define a predict function as an endpoint 
@app.route('/', methods=['POST','GET'])
def predict():
    if request.method=='POST':
        return "GET request-response"
    if request.method=='GET':
        print(sys.version)
        print("Process started")
        img = image.load_img(path='input.png',color_mode="grayscale",target_size=(28,28,1))
        print('image_loaded')
        img = image.img_to_array(img)
        print('running_inference')
        output = model.predict(img.reshape(1,28,28,1)) # To do it for a new data point
        print('Got the result !')
        output.argmax()
        print(output.argmax())
        return str(output.argmax())
