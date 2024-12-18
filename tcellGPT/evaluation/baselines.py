import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# Define the MLP Classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

        # initialize
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def train_mlp_classifier(features, responses, epochs=20, hidden_size=128, learning_rate=0.001, test_ratio=0.1):
    # One-hot encode responses
    encoder = OneHotEncoder()
    responses_encoded = encoder.fit_transform(responses.reshape(-1, 1))
    num_classes = responses_encoded.shape[1]
    print(responses_encoded.shape)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(features, responses_encoded, test_size=test_ratio, shuffle=True)

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    if hasattr(y_train, "toarray"):
        y_train = y_train.toarray()
    y_train = torch.tensor(y_train, dtype=torch.float32)
    if hasattr(y_test, "toarray"):
        y_test = y_test.toarray()
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # create dataloader
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize model, loss function, and optimizer
    input_size = X_train.shape[1]
    model = MLPClassifier(input_size, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        for inputs, labels in tqdm(train_loader):
          outputs = model(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        y_pred = torch.argmax(outputs, dim=1)

    # Convert predictions to numpy arrays
    y_pred = y_pred.numpy()
    y_test = torch.argmax(y_test, dim=1).numpy()

    # Calculate accuracy and classification report
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\nAccuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(report)
