from numpy import argmax
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# load and prepare the image
def load_image(filename):
    # load the image in grayscale
    img = load_img(filename, color_mode='grayscale', target_size=(28, 28))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    # normalize pixel values
    img = img.astype('float32') / 255.0
    return img

# load an image and predict the class
def run_example():
    img = load_image('sample_image.png')           # <== Your image here
    model = load_model('final_model.h5')           # <== Your model here
    predict_value = model.predict(img)
    digit = argmax(predict_value)
    print("Predicted Digit:", digit)

# entry point
run_example()
