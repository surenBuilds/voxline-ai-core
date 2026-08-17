import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# VOXLINE AI CORE v0.1
# First learning neural network
# ============================================================


class VoxlineAI(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.network(x)


X = torch.tensor([
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 4.0],
    [4.0, 5.0],
    [5.0, 6.0],
    [6.0, 7.0],
    [7.0, 8.0],
    [8.0, 9.0],
    [9.0, 10.0],
    [10.0, 11.0],
])

Y = torch.tensor([
    [3.0],
    [5.0],
    [7.0],
    [9.0],
    [11.0],
    [13.0],
    [15.0],
    [17.0],
    [19.0],
    [21.0],
])

model = VoxlineAI()
print("=" * 60)
print("VOXLINE AI CORE v0.1")
print("=" * 60)
print("\nModel:")
print(model)

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("\nStarting training...\n")

epochs = 2000
for epoch in range(epochs):
    predictions = model(X)
    loss = loss_function(predictions, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.8f}")

model_path = "voxline_ai_v0_1.pth"
torch.save(model.state_dict(), model_path)
print("\nTraining complete.")
print(f"Model saved to: {model_path}")

model.eval()
with torch.no_grad():
    result = model(torch.tensor([[20.0, 30.0]]))

print("\nTest:")
print("Input: 20 + 30")
print(f"AI prediction: {result.item():.4f}")

print("\n" + "=" * 60)
print("VOXLINE AI CORE v0.1 IS RUNNING")
print("=" * 60)
