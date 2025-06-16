import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit

# Configuration
DATASET_PATH = "dataset.json"
MODELS_DIR = "saved_models"
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
TRAIN_SPLIT = 0.8
INPUT_SIZE = 1000
HIDDEN_SIZE = 128

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)


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
        # After two 2x pooling operations, the sequence length is reduced by a factor of 4.
        # But the conv layers also change size, let's calculate it dynamically
        conv_output_size = self._get_conv_output_size(input_size)
        self.fc_input_size = conv_output_size * 64

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, num_classes)

        # Activation functions
        self.relu = nn.ReLU()

    def _get_conv_output_size(self, input_size):
        x = torch.randn(1, 1, input_size)
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        return x.shape[2]

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
        self.fc1 = nn.Linear(self.fc_input_size, hidden_size * 2)
        self.bn4 = nn.BatchNorm1d(hidden_size * 2)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
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


def train(model, train_loader, test_loader, criterion, optimizer, epochs, model_save_path):
    """Train a PyTorch model and evaluate on the test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_accuracy = 0.0
    patience = 10  # Number of epochs to wait for improvement
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        for traces, labels in train_loader:
            traces, labels = traces.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(traces)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for traces, labels in test_loader:
                traces, labels = traces.to(device), labels.to(device)
                outputs = model(traces)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_accuracy = correct / total
        print(f'Epoch {epoch+1}/{epochs}, Test Accuracy: {epoch_accuracy:.4f}')

        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved with accuracy: {best_accuracy:.4f}')
            patience_counter = 0  # Reset patience
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Stopping early. No improvement in {patience} epochs.")
            break


def evaluate(model, test_loader, website_names):
    """Evaluate a PyTorch model on the test set and show classification report."""
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

    print("\nClassification Report:")
    string_website_names = [str(name) for name in website_names]
    print(classification_report(all_labels, all_preds, target_names=string_website_names, zero_division=0))


class WebsiteDataset(Dataset):
    """Custom PyTorch Dataset for website fingerprinting traces."""

    def __init__(self, json_path, sequence_length):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.sequence_length = sequence_length
        
        # Get unique website names and create mappings
        website_names = sorted(list(set(item['website'] for item in self.data)))
        self.website_to_idx = {name: i for i, name in enumerate(website_names)}
        self.idx_to_website = {i: name for i, name in self.website_to_idx.items()}
        
        self.samples = []
        for item in self.data:
            label = self.website_to_idx[item['website']]
            trace = item['trace_data']
            self.samples.append((trace, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        trace, label = self.samples[idx]
        
        if len(trace) > self.sequence_length:
            trace = trace[:self.sequence_length]
        else:
            trace = np.pad(trace, (0, self.sequence_length - len(trace)), 'constant')
            
        if np.std(trace) > 1e-8:
            trace = (trace - np.mean(trace)) / np.std(trace)
        else:
            trace = trace - np.mean(trace)

        trace_tensor = torch.tensor(trace, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return trace_tensor, label_tensor


def main():
    """Main function to train and evaluate the model."""
    dataset = WebsiteDataset(DATASET_PATH, INPUT_SIZE)
    website_names = list(dataset.idx_to_website.values())
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - TRAIN_SPLIT, random_state=42)
    labels = [label for _, label in dataset.samples]
    train_idx, test_idx = next(sss.split(np.zeros(len(labels)), labels))
    
    train_subset = Subset(dataset, train_idx)
    test_subset = Subset(dataset, test_idx)
    
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = ComplexFingerprintClassifier(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_classes=len(website_names)
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    model_save_path = os.path.join(MODELS_DIR, 'best_model.pth')
    train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=EPOCHS,
        model_save_path=model_save_path
    )
    
    print("\n--- Final Evaluation on Test Set ---")
    best_model = ComplexFingerprintClassifier(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_classes=len(website_names)
    )
    best_model.load_state_dict(torch.load(model_save_path, weights_only=True))
    evaluate(best_model, test_loader, website_names)


if __name__ == '__main__':
    main()
