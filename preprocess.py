import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.gates = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4
            * hidden_channels,  # for input, forget, cell, and output gates
            kernel_size=kernel_size,
            padding=self.padding,
        )

    def forward(self, input_tensor, hidden_state):
        h_cur, c_cur = hidden_state
        # print("[input_tensor, h_cur]", [input_tensor.size(), h_cur.size()])
        # concatenate along the channel dimension
        combined = torch.cat([input_tensor, h_cur], dim=1)
        gates = self.gates(combined)

        # Split the combined gate tensor into its components
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)

        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        output_gate = torch.sigmoid(output_gate)

        c_next = forget_gate * c_cur + input_gate * cell_gate
        h_next = output_gate * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (
            torch.zeros(
                batch_size,
                self.hidden_channels,
                height,
                width,
                device=self.gates.weight.device,
            ),
            torch.zeros(
                batch_size,
                self.hidden_channels,
                height,
                width,
                device=self.gates.weight.device,
            ),
        )


class ConvLSTM(nn.Module):
    def __init__(self):
        super(ConvLSTM, self).__init__()
        # Spatial Encoder
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=128,
            kernel_size=11,
            stride=4,
            padding=(11 - 1) // 2,
        )
        self.bn1 = nn.BatchNorm2d(128)
        self.dropout1 = nn.Dropout(0.5)

        self.conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=64,
            kernel_size=5,
            stride=2,
            padding=(5 - 1) // 2,
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(0.5)

        # Temporal Encoder (ConvLSTM)
        self.convlstm1 = ConvLSTMCell(
            input_channels=64, hidden_channels=64, kernel_size=3
        )
        self.convlstm2 = ConvLSTMCell(
            input_channels=64, hidden_channels=32, kernel_size=3
        )
        self.convlstm3 = ConvLSTMCell(
            input_channels=32, hidden_channels=64, kernel_size=3
        )
        # Spatial Decoder
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=0,
        )
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout3 = nn.Dropout(0.5)

        self.deconv2 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=128,
            kernel_size=12,
            stride=4,
            padding=4,
            output_padding=0,
        )
        self.bn4 = nn.BatchNorm2d(128)
        self.dropout4 = nn.Dropout(0.5)

        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=3, kernel_size=11, padding=(11 - 1) // 2
        )

    def forward(self, x):
        # Initialize hidden states and cell states
        b, seq_len, _, h, w = x.size()
        # print("x.size()", x.size())
        h1, c1 = self.convlstm1.init_hidden(b, (h // 8, w // 8))
        h2, c2 = self.convlstm2.init_hidden(b, (h // 8, w // 8))
        h3, c3 = self.convlstm3.init_hidden(b, (h // 8, w // 8))
        output_sequence = []

        for t in range(seq_len):
            # Spatial Encoder
            xt = self.conv1(x[:, t])
            xt = self.bn1(xt)
            xt = self.dropout1(xt)
            xt = F.relu(xt)

            xt = self.conv2(xt)
            xt = self.bn2(xt)
            xt = self.dropout2(xt)
            xt = F.relu(xt)

            # Temporal Encoder
            h1, c1 = self.convlstm1(xt, (h1, c1))
            h2, c2 = self.convlstm2(h1, (h2, c2))
            h3, c3 = self.convlstm3(h2, (h3, c3))

            # Spatial Decoder
            xt = self.deconv1(xt)
            xt = self.bn3(xt)
            xt = self.dropout3(xt)
            xt = F.relu(xt)

            xt = self.deconv2(xt)
            xt = self.bn4(xt)
            xt = self.dropout4(xt)
            xt = F.relu(xt)

            xt = torch.sigmoid(self.conv3(xt))

            output_sequence.append(xt.unsqueeze(1))

        # Concatenate along the sequence dimension

        output_sequence = torch.cat(output_sequence, dim=1)
        return output_sequence
# training 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Testing the dataset
batch_size = 5
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Instantiate the model
model = ConvLSTM()
model = nn.DataParallel(model)
model.to(device)

criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5, eps=1e-6)

# Early stopping parameters
patience = 5
best_val_loss = float('inf')
patience_counter = 0
best_model_path = 'best_model.pth'  # Path to save the best model

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for (images,) in train_loader:
        images = images.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    average_loss = total_loss / num_batches
    print(f'Epoch [{epoch + 1}/{num_epochs}] Average Training Loss: {average_loss:.4f}')
    
    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for (images,) in val_loader:
            images = images.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, images).item()

    val_loss /= len(val_loader)
    print(f'Validation Loss after Epoch {epoch+1}: {val_loss:.4f}')

    # Early stopping check and model saving
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save the best model
        torch.save(model.state_dict(), best_model_path)
        print(f'Model saved at epoch {epoch + 1}')
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f'Early stopping triggered after epoch {epoch + 1}')
        break

# Load the best model after training
best_model = ConvLSTM()
best_model = nn.DataParallel(best_model)
best_model.load_state_dict(torch.load(best_model_path))
best_model.to(device)
best_model.eval()
# Now best_model holds the best model state