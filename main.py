import numpy as np
import pickle
import streamlit as st


# creating a function for Prediction
def diabetes_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'


if __name__ == '__main__':
    # loading the saved model
    loaded_model = pickle.load(open('./trained_model.sav', 'rb'))

    # giving a title
    st.title('Diabetes Prediction Web App')

    # getting the input data from the user
    pregnancies = st.text_input('Number of Pregnancies')
    glucose = st.text_input('Glucose Level')
    bloodPressure = st.text_input('Blood Pressure value')
    skinThickness = st.text_input('Skin Thickness value')
    insulin = st.text_input('Insulin Level')
    bmi = st.text_input('BMI value')
    diabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    age = st.text_input('Age of the Person')

    # code for Prediction
    diagnosis = ''

    # creating a button for Prediction

    try:
        if st.button('Diabetes Test Result'):
            diagnosis = diabetes_prediction(
                [pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi, diabetesPedigreeFunction, age]
            )
    except:
        diagnosis = 'Isi lengkap bos'

    st.success(diagnosis)
    st.error(diagnosis)
