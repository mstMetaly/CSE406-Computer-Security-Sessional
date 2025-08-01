import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit

# Configuration
DATASET_PATH = "dataset.json"
MODELS_DIR = "saved_models"
PLOTS_DIR = "training_plots"
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-4 
TRAIN_SPLIT = 0.8
INPUT_SIZE = 188
HIDDEN_SIZE = 128 

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


class FingerprintClassifier(nn.Module):
    """Basic neural network model for website fingerprinting classification."""
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(FingerprintClassifier, self).__init__()
        
        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions and pooling
        conv_output_size = input_size // 8  # After two 2x pooling operations
        self.fc_input_size = conv_output_size * 64
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Reshape for 1D convolution: (batch_size, 1, input_size)
        x = x.unsqueeze(1)
        
        # Convolutional layers
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        
        # Flatten for fully connected layers
        x = x.view(-1, self.fc_input_size)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
        
class ComplexFingerprintClassifier(nn.Module):
    """A more complex neural network model for website fingerprinting classification."""
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(ComplexFingerprintClassifier, self).__init__()
        
        # 1D Convolutional layers with batch normalization
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions and pooling
        conv_output_size = input_size // 8  # After three 2x pooling operations
        self.fc_input_size = conv_output_size * 128
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, hidden_size*2)
        self.bn4 = nn.BatchNorm1d(hidden_size*2)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(hidden_size*2, hidden_size)
        self.bn5 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Reshape for 1D convolution: (batch_size, 1, input_size)
        x = x.unsqueeze(1)
        
        # Convolutional layers
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Flatten for fully connected layers
        x = x.view(-1, self.fc_input_size)
        
        # Fully connected layers
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

def plot_training_curves(train_losses, train_accs, test_losses, test_accs, model_name):
    """Plot training and validation curves."""
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Validation Loss')
    plt.title(f'{model_name} - Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Validation Accuracy')
    plt.title(f'{model_name} - Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'{model_name}_training_curves.png'))
    plt.close()

def plot_confusion_matrix(true_labels, predictions, class_names, model_name):
    """Plot confusion matrix."""
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'{model_name}_confusion_matrix.png'))
    plt.close()

def train(model, train_loader, test_loader, criterion, optimizer, epochs, model_save_path):
    """Train a PyTorch model and evaluate on the test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for traces, labels in train_loader:
            traces, labels = traces.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(traces)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * traces.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        # Evaluation
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for traces, labels in test_loader:
                traces, labels = traces.to(device), labels.to(device)
                outputs = model(traces)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * traces.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(test_loader.dataset)
        epoch_accuracy = correct / total
        test_losses.append(epoch_loss)
        test_accuracies.append(epoch_accuracy)
        
        # Print status
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}, '
              f'Test Loss: {test_losses[-1]:.4f}, Test Acc: {test_accuracies[-1]:.4f}')
        
        # Save the best model
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved with accuracy: {best_accuracy:.4f}')
    
    return best_accuracy, train_losses, train_accuracies, test_losses, test_accuracies



def evaluate(model, test_loader, website_names):
    """Evaluate a PyTorch model on the test set and show classification report with website names.
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader for testing data
        website_names: List of website names for classification report
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for traces, labels in test_loader:
            traces, labels = traces.to(device), labels.to(device)
            outputs = model(traces)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Print classification report with website names instead of indices
    print("\nClassification Report:")
    print(classification_report(
        all_labels, 
        all_preds, 
        target_names=website_names,
        zero_division=1
    ))
    
    return all_preds, all_labels


class WebsiteDataset(Dataset):
    """Custom Dataset for loading website fingerprinting data."""
    def __init__(self, traces, labels):
        self.traces = torch.FloatTensor(traces)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.traces)
    
    def __getitem__(self, idx):
        return self.traces[idx], self.labels[idx]

def main():
    """ Implement the main function to train and evaluate the models.
    1. Load the dataset from the JSON file, probably using a custom Dataset class
    2. Split the dataset into training and testing sets
    3. Create data loader for training and testing
    4. Define the models to train
    5. Train and evaluate each model
    6. Print comparison of results
    """
    # Load and preprocess data
    with open(DATASET_PATH, 'r') as f:
        data = json.load(f)
    
    # Extract traces and labels
    traces = []
    labels = []
    website_names = set()  # Use set to get unique website names
    
    # Process list structure
    for entry in data:
        website = entry['website']
        trace_data = entry['trace_data']
        website_index = entry['website_index']
        website_names.add(website)
        
        # Ensure all traces are of length INPUT_SIZE by truncating or padding
        trace = np.array(trace_data[:INPUT_SIZE] + [0] * (INPUT_SIZE - len(trace_data[:INPUT_SIZE])))
        traces.append(trace)
        labels.append(website_index)
    
    # Convert website names to ordered list
    website_names = sorted(list(website_names))
    
    traces = np.array(traces)
    labels = np.array(labels)
    
    # Additional preprocessing: Standardization
    traces = (traces - np.mean(traces, axis=1, keepdims=True)) / (np.std(traces, axis=1, keepdims=True) + 1e-8)
    
    # Split dataset
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=1-TRAIN_SPLIT, random_state=42)
    train_idx, test_idx = next(splitter.split(traces, labels))
    
    # Create datasets
    train_dataset = WebsiteDataset(traces[train_idx], labels[train_idx])
    test_dataset = WebsiteDataset(traces[test_idx], labels[test_idx])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Training settings
    num_classes = len(website_names)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Train and evaluate simple model
    print("\nTraining Simple Model:")
    simple_model = FingerprintClassifier(INPUT_SIZE, HIDDEN_SIZE, num_classes)
    simple_optimizer = optim.Adam(simple_model.parameters(), lr=LEARNING_RATE)
    simple_model_path = os.path.join(MODELS_DIR, "simple_model.pth")
    
    simple_accuracy, simple_train_losses, simple_train_accs, simple_test_losses, simple_test_accs = train(
        simple_model,
        train_loader,
        test_loader,
        criterion,
        simple_optimizer,
        EPOCHS,
        simple_model_path
    )
    
    # Plot training curves for simple model
    plot_training_curves(
        simple_train_losses, simple_train_accs,
        simple_test_losses, simple_test_accs,
        "Simple_Model"
    )
    
    # Load best simple model and evaluate
    simple_model.load_state_dict(torch.load(simple_model_path))
    print("\nFinal Evaluation of Simple Model:")
    simple_preds, simple_labels = evaluate(simple_model, test_loader, website_names)
    
    # Plot confusion matrix for simple model
    plot_confusion_matrix(simple_labels, simple_preds, website_names, "Simple_Model")
    
    # Train and evaluate complex model
    print("\nTraining Complex Model:")
    complex_model = ComplexFingerprintClassifier(INPUT_SIZE, HIDDEN_SIZE, num_classes)
    complex_optimizer = optim.Adam(complex_model.parameters(), lr=LEARNING_RATE)
    complex_model_path = os.path.join(MODELS_DIR, "complex_model.pth")
    
    complex_accuracy, complex_train_losses, complex_train_accs, complex_test_losses, complex_test_accs = train(
        complex_model,
        train_loader,
        test_loader,
        criterion,
        complex_optimizer,
        EPOCHS,
        complex_model_path
    )
    
    # Plot training curves for complex model
    plot_training_curves(
        complex_train_losses, complex_train_accs,
        complex_test_losses, complex_test_accs,
        "Complex_Model"
    )
    
    # Load best complex model and evaluate
    complex_model.load_state_dict(torch.load(complex_model_path))
    print("\nFinal Evaluation of Complex Model:")
    complex_preds, complex_labels = evaluate(complex_model, test_loader, website_names)
    
    # Plot confusion matrix for complex model
    plot_confusion_matrix(complex_labels, complex_preds, website_names, "Complex_Model")
    
    # Print comparison
    print("\nModel Comparison:")
    print(f"Simple Model Best Accuracy: {simple_accuracy:.4f}")
    print(f"Complex Model Best Accuracy: {complex_accuracy:.4f}")

if __name__ == "__main__":
    main()
