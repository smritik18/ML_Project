import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from Dataset import CharDataset
from network import CNN
from utils import test_accuracy


def train(network, dataset, num_epochs, batch_size, save_path):
    # Define your loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=0.001)


    # Split dataset into train and test partitions
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create dataloaders for train and test partitions
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cpu")
    network.to(device)

    for epoch in range(num_epochs):
        network.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = network(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        
        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_dataset)
        print(f"Training Loss: {epoch_loss:.4f}")

        # Evaluate on test set
        accuracy = test_accuracy(network, test_loader, device)
        
        print(f"Accuracy on test set: {accuracy:.4f}")
    torch.save(network.to(torch.device('cpu')).state_dict(),save_path)


if __name__=="__main__":

    network = CNN()

    data_dir = 'data/'
    csv_file = 'data/english.csv'

    # Define transforms for the dataset
    data_transform = transforms.Compose([
        transforms.Resize((32, 32)),    # Resize images to (32, 32)
        transforms.Grayscale(), # Convert images to grayscale
        transforms.ToTensor()           # Convert images to PyTorch tensors
    ])


    # Create a custom dataset and dataloader
    dataset = CharDataset(data_dir=data_dir,
                            csv_file=csv_file,
                            transform=data_transform)
    
    num_epochs = 2

    batch_size = 32
    
    save_path = '/Users/smritikumari/Desktop/ML_Project/hand_dataset_model/final_project/model.pt'

    train(network, dataset, num_epochs, batch_size, save_path)

