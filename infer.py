import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from Dataset import CharDataset
from network import CNN
from PIL import Image
from utils import extract_characters, emnist_byclass_label_to_char

def predict_single_character(image, network):
    # Define transforms for the dataset
    transform = transforms.Compose([
        transforms.Resize((32, 32)),    # Resize images to (32, 32)
        transforms.Grayscale(),         # Convert images to grayscale
        transforms.ToTensor()           # Convert images to PyTorch tensors
    ])
    image = transform(image)
    image = torch.unsqueeze(image, 0)  # Add batch dimension
    with torch.no_grad():
        output = network(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

def predict_text(image, network):
    network.eval()
    predicted_chars = []
    characters = extract_characters(image)
    for character in characters:
        predicted_label = predict_single_character(character, network)
        predicted_char = emnist_byclass_label_to_char(predicted_label)
        predicted_chars.append(predicted_char)   
    return ''.join(predicted_chars)

if __name__=="__main__":
    
    # Provide the path to your input image
    input_image_path = 'test_imgs/Words/coin.png'
    image = Image.open(input_image_path)
    model_path = '/Users/smritikumari/Desktop/ML_Project/hand_dataset_model/final_project/model.pt'

    model = CNN()
    model.load_state_dict(torch.load(model_path))

    # Recognize the text in the image
    recognized_text = predict_text(image, model)

    print("Recognized Text:", recognized_text)
