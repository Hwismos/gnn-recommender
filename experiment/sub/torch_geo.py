import torch

def format_pytorch_version(version):
  return version.split('+')[0]

TORCH_version = torch.__version__
TORCH = format_pytorch_version(TORCH_version)

def format_cuda_version(version):
  return 'cu' + version.replace('.', '')

CUDA_version = torch.version.cuda
CUDA = format_cuda_version(CUDA_version)

print(f'TORCH: {type(TORCH)}, CUDA: {CUDA}')


# TORCH: 1.10.2 
# CUDA: cu102
# pip install torch-scatter     -f https://pytorch-geometric.com/whl/torch-1.10.2+cu102.html
# pip install torch-sparse      -f https://pytorch-geometric.com/whl/torch-1.10.2+cu102.html
# pip install torch-cluster     -f https://pytorch-geometric.com/whl/torch-1.10.2+cu102.html
# pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.10.2+cu102.html
# pip install torch-geometric 