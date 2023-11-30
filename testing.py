from ConvLSTM_VGG19 import ConvLSTM_VGG19
import torch
# Example usage
model = ConvLSTM_VGG19()
image = torch.randn(1, 10, 3, 224, 224)
output = model(image)
print(output.size())
