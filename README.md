# Positional Encoding in PyTorch

## Overview
This repository contains a Jupyter Notebook implementing positional encoding in PyTorch. Positional encoding is crucial for sequence-based models such as Transformers, as it provides order information to the model.

## Features
- Implementation of a **custom sequence dataset** with positional encoding.
- Use of **PyTorch** for model creation and training.
- Visualization of positional encodings.
- Training and evaluation of a model on generated sequential data.

## Requirements
To run the notebook, install the necessary dependencies:
```bash
pip install torch numpy matplotlib
```

## Usage
1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Positional_Encoding.ipynb
   ```
2. Run all cells to:
   - Generate and visualize positional encodings.
   - Train a simple model using PyTorch.
   - Evaluate the model's performance.

## Detailed Code Explanation

### 1. Importing Libraries
The notebook begins by importing essential libraries:
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
```
- `torch` and `torch.nn`: For defining the neural network and its operations.
- `torch.optim`: Optimization library for training the model.
- `numpy`: For numerical computations.
- `matplotlib.pyplot`: For visualizing the positional encoding.
- `torch.utils.data`: Dataset utilities to create and load data efficiently.

### 2. Sequence Dataset Implementation
A custom dataset is created using PyTorch's `Dataset` class:
```python
class SequenceDataset(Dataset):
    def __init__(self, seq_length=20, num_samples=1000, feature_dim=10):
        self.seq_length = seq_length
        self.num_samples = num_samples
        self.feature_dim = feature_dim
        self.sequences = torch.randn(num_samples, seq_length, feature_dim) * 0.5
        weights = torch.linspace(0.1, 0.5, seq_length).unsqueeze(1)
        self.labels = torch.sum(self.sequences * weights, dim=(1, 2))
        self.labels = (self.labels - self.labels.mean()) / (self.labels.std() + 1e-8)
```
- Random sequences are generated as input.
- Labels are created based on weighted summation of input features.
- Labels are normalized to maintain a stable training process.

### 3. Positional Encoding Implementation
A function is created to implement positional encoding:
```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]
```
- The encoding is created using sine and cosine functions.
- The positional encoding is added to input sequences before passing to the model.

### 4. Model Definition
A simple model utilizing positional encoding is defined:
```python
class SimpleTransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleTransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x
```
- The model consists of an embedding layer, positional encoding, LSTM layers, and a final linear layer.
- The LSTM processes the encoded sequence before passing it to the fully connected layer for prediction.

### 5. Training the Model
A function is defined to train the model:
```python
def train_model(model, dataloader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")
```
- The function loops through epochs, computing loss and updating model weights.
- The loss function is optimized using backpropagation.

### 6. Running the Training Process
```python
dataset = SequenceDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = SimpleTransformerModel(input_dim=10, hidden_dim=32, output_dim=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, dataloader, criterion, optimizer, epochs=10)
```
- The dataset is loaded using `DataLoader`.
- The model is initialized with a specified input and hidden dimension.
- The training process is executed over 10 epochs.

## File Structure
- `Positional_Encoding.ipynb`: The main notebook containing code and explanations.

## Author
This project was developed using PyTorch for learning and experimentation with positional encoding in neural networks.

## License
This project is open-source and can be used freely for educational and research purposes.

