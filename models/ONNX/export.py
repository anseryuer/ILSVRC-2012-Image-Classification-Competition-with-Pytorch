import onnx
import torch

def export_onnx(model:torch.nn.Module, input_shape, filename = None):
    model.eval()
    model.to('cpu')
    if filename is None:
        filename = model.__class__.__name__ + '.onnx'
    x = torch.randn(input_shape, dtype=torch.float32, requires_grad=True)
    torch.onnx.export(model, x, filename, export_params=True, opset_version=10, do_constant_folding=True, input_names = ['input'], output_names = ['labels'], dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})
    print(f"Model exported to {filename}")