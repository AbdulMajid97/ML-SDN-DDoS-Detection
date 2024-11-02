# train/train_deep_model.py
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from models.dl_ddos_classifier import DDoSClassifier

def train_deep_model(X_train, y_train, input_dim, epochs=150, batch_size=32):
    # Prepare data
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train.to_numpy(), dtype=torch.float32).view(-1, 1))
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, criterion, optimizer, and TensorBoard writer
    model = DDoSClassifier(input_dim=input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    writer = SummaryWriter(log_dir='runs/ddos_experiment')

    # Training loop
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        writer.add_scalar('Training Loss', avg_loss, epoch)

    writer.close()
    return model
