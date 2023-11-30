import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights


class ConvLSTMCell(nn.Module):

    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.gates = nn.Conv2d(in_channels=input_channels + hidden_channels,
                               out_channels=4 * hidden_channels,  # For input, forget, cell, and output gates
                               kernel_size=kernel_size,
                               padding=self.padding)

    def forward(self, input_tensor, hidden_state):
        h_cur, c_cur = hidden_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        # print(f"combined shape: {[input_tensor.shape, h_cur.shape]}")
        # print(f"combined shape -->: {combined.shape}")
        gates = self.gates(combined)
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
        return (torch.zeros(batch_size, self.hidden_channels, height, width, device=self.gates.weight.device),
                torch.zeros(batch_size, self.hidden_channels, height, width, device=self.gates.weight.device))


class ConvLSTM_VGG19(nn.Module):
    """
    ConvLSTM model with VGG19-based architecture
    """

    def __init__(self):
        super(ConvLSTM_VGG19, self).__init__()

        # Load the pre-trained VGG19 model
        vgg19_model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        vgg19_features = vgg19_model.features
        
        # using the first 10 layers of VGG19
        self.encoder = nn.Sequential(*list(vgg19_features.children())[:12])
        

        # ConvLSTM layers
        self.convlstm1 = ConvLSTMCell(
            input_channels=256, hidden_channels=64, kernel_size=3)
        self.convlstm2 = ConvLSTMCell(
            input_channels=64, hidden_channels=32, kernel_size=3)
        self.convlstm3 = ConvLSTMCell(
            input_channels=32, hidden_channels=64, kernel_size=3)

        # Spatial Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64,
                               kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=128,
                               kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=3,
                               kernel_size=4, stride=2, padding=2)
        )

    def forward(self, x):
        b, seq_len, _, h, w = x.size()
        print("******:", x.shape)
        # input to convLSTM will be whatever coming from the encoder 
        # concatenate whatever the initialization of hidden LSTM will be
        
        _, _, vgg_height, vgg_width = self.encoder(x[:, 0]).shape

        h1, c1 = self.convlstm1.init_hidden(b, (vgg_height, vgg_width))
        h2, c2 = self.convlstm2.init_hidden(b, (vgg_height, vgg_width))
        h3, c3 = self.convlstm3.init_hidden(b, (vgg_height, vgg_width))

        output_sequence = []
        for t in range(seq_len):
            xt = self.encoder(x[:, t])
            h1, c1 = self.convlstm1(xt, (h1, c1))
            h2, c2 = self.convlstm2(h1, (h2, c2))
            h3, c3 = self.convlstm3(h2, (h3, c3))
            xt = self.decoder(h3)
            xt = torch.sigmoid(xt)
            output_sequence.append(xt.unsqueeze(1))

        output_sequence = torch.cat(output_sequence, dim=1)
        return output_sequence
