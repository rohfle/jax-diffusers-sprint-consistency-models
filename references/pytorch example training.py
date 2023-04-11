import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

# Step 1: Initialize the model
model = nn.Linear(1, 1)

# Step 2: Prepare the data
data = [(1, 2), (2, 4), (3, 6)]
loader = DataLoader(data, batch_size=2, shuffle=True)

# Step 3: Define the loss function
loss_fn = nn.MSELoss()

# Step 4: Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Step 5: Training loop
for epoch in range(10):
    for batch in loader:
        x, y_true = batch

        # Compute the predictions
        y_pred = model(x)

        # Compute the loss
        loss = loss_fn(y_pred, y_true)

        # Compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # Update the parameters
        optimizer.step()

    # Print the loss after each epoch
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Step 6: Evaluation
model.eval()
val_data = [(4, 8), (5, 10), (6, 12)]
val_loader = torch.utils.data.DataLoader(val_data, batch_size=2, shuffle=True)

with torch.no_grad():
    for batch in val_loader:
        x, y_true = batch
        y_pred = model(x)
        loss = loss_fn(y_pred, y_true)
        print(f"Validation loss: {loss.item()}")

# Step 7: Save the model
torch.save(model.state_dict(), "model.pt")