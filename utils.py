import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from Dataset import CharDataset
from network import CNN
from PIL import Image
import cv2
import numpy as np


def test_accuracy(network, test_loader, device):
    # Evaluate on test set
    network.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            outputs = network(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

def emnist_byclass_label_to_char(label):
        # Decode label using the label mapping
    label_mapping = {label: index for index, label in enumerate("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")}
    return list(label_mapping.keys())[list(label_mapping.values()).index(label)]


def crop_pillow_image(image, bounding_boxes, padding=60):
    """
    Crop a Pillow image based on the given list of bounding box coordinates and pad with white pixels.

    Args:
    - image: Pillow image object
    - bounding_boxes: List of bounding box coordinates (each bounding box is a tuple (x, y, w, h))
    - padding: Amount of padding to add around the bounding box (default: 10 pixels)

    Returns:
    - cropped_images: List of cropped and padded Pillow image objects
    """
    cropped_images = []
    for bbox in bounding_boxes:
        x, y, w, h = bbox
        # Crop the image using the provided coordinates
        cropped_image = image.crop((x, y, x + w, y + h))
        # Create a larger canvas with white background
        padded_image = Image.new('L', (w + 2 * padding, h + 2 * padding), color=255)
        # Paste the cropped image onto the canvas with an offset
        padded_image.paste(cropped_image, (padding, padding))
        # padded_image.show()
        cropped_images.append(padded_image)
    return cropped_images

def pillow_to_opencv(pil_image):
    """
    Convert a Pillow image to a grayscale OpenCV image.

    Args:
    - pil_image: A Pillow image object.

    Returns:
    - opencv_image: A grayscale OpenCV image (NumPy array).
    """
    # Convert the Pillow image to a NumPy array
    np_array = np.array(pil_image)

    # Convert the image to grayscale if it's in RGB mode
    if np_array.ndim == 3 and np_array.shape[2] == 3:
        gray_np_array = cv2.cvtColor(np_array, cv2.COLOR_RGB2GRAY)
    elif np_array.ndim == 3 and np_array.shape[2] == 4:  # If the image has an alpha channel (RGBA)
        gray_np_array = cv2.cvtColor(np_array, cv2.COLOR_RGBA2GRAY)
    else:
        gray_np_array = np_array

    return gray_np_array

def extract_characters(image):
    
    orig_img = image
    
    img = pillow_to_opencv(image)
    # Apply thresholding to get binary image
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Invert the image
    img = cv2.bitwise_not(img)
    # Find contours
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Extract bounding boxes of contours
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    # Sort bounding boxes by x-coordinate to get characters from left to right
    bounding_boxes.sort(key=lambda x: x[0])

    return crop_pillow_image(orig_img, bounding_boxes)