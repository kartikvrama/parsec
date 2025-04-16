TORCH=2.1.0
CUDA=cu118
python -m pip install torch-scatter --no-cache-dir --no-index --find-links https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
python -m pip install torch-sparse --no-cache-dir --no-index --find-links https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
python -m pip install torch-cluster --no-cache-dir --no-index --find-links https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
python -m pip install torch-spline-conv --no-cache-dir --no-index --find-links https://pytorch-geometric.com/whl/torch-${TORCH}+${CUDA}.html
python -m pip install torch-geometric --no-cache-dir