import streamlit as st
import torch
from PIL import Image
from torchvision.transforms import transforms
from network import CNN
from infer import predict_text
# Define a title for your Streamlit app
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







