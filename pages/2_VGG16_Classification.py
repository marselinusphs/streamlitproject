import streamlit as st
from PIL import Image
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import numpy as np


def predict(image):
    model = VGG16()

    image = Image.open(image)
    image = image.resize((224, 224))

    image = np.array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    # prepare the image for the VGG model
    image = preprocess_input(image)

    # predict the probability across all output classes
    yhat = model.predict(image)

    # convert the probabilities to class labels
    label = decode_predictions(yhat)

    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    return label


if __name__ == '__main__':
    st.title("Upload + Classification Example")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        label = predict(uploaded_file)
        st.write('%s (%.2f%%)' % (label[1], label[2]*100))
