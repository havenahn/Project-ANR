import torch
from nn_multistep_loss_1_bilinear.helpers.networkarch import NeuralNetwork

# Load your trained model
model = NeuralNetwork(N_x = 4, N_h = 32, N_e = 100)
checkpoint = torch.load('nn_multistep_loss_1_bilinear/model_100.pt', map_location='cpu')
model.load_state_dict(checkpoint['model_dict'])
model.eval()

# Dummy input for export (shape: [1, D_xi])
D_xi = model.D_x + model.D_e
dummy_xi = torch.randn(1, D_xi)

# Export only the B_bilinear part
torch.onnx.export(
    model.B_bilinear,
    dummy_xi,
    "B_bilinear.onnx",
    input_names=['xi'],
    output_names=['B_out'],
    dynamic_axes={'xi': {0: 'batch_size'}, 'B_out': {0: 'batch_size'}}
)