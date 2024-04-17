import streamlit as st
import torch
from PIL import Image
from torchvision.transforms import transforms
from network import CNN
from infer import predict_text
# Define a title for the Streamlit app
st.title('Image Text Recognition')

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["png"])

# Check if a file is uploaded
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Check if the user clicked the 'Predict' button
    if st.button('Predict'):
        # Load your trained PyTorch model
        model = CNN()
        model.load_state_dict(torch.load('/Users/smritikumari/Desktop/ML_Project/hand_dataset_model/final_project/model.pt'))

        # Perform text recognition prediction
        recognized_text = predict_text(image, model)

        # Display the predicted text
        st.write('Predicted Text:', recognized_text)
        
    # Input box for users to enter their correction
    correction = st.text_input("Enter any prediction correction:")

    # Submit button
    if st.button('Submit'):
        if correction:  # Check if the correction is not empty
            st.success("Your correction was submitted successfully!")
            st.write("You submitted: ", correction)
        else:
            st.error("Please enter a correction before submitting.")






