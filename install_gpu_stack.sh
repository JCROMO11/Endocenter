#!/bin/bash
# EndoCenter MLOps - GPU Stack Installation

echo "🚀 Instalando stack GPU optimizado para CUDA 12.9..."

# Activar virtual environment
source venv/bin/activate

# Install PyTorch with CUDA 12.9 support (nightly para RTX 5070)
echo "⚡ Instalando PyTorch con CUDA 12.9..."
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu129

# Install FAISS-GPU
echo "�� Instalando FAISS-GPU..."
pip install faiss-gpu --no-cache-dir

# Install rest of the stack from requirements
echo "📦 Instalando resto del stack..."
pip install -r requirements/base.txt

# Test installation
echo "🧪 Verificando instalación..."
python -c "
import torch, faiss, yaml, pydantic
print('✅ PyTorch:', torch.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✅ GPU:', torch.cuda.get_device_name(0))
print('✅ FAISS GPUs:', faiss.get_num_gpus())
print('✅ PyYAML:', yaml.__version__)
print('✅ Pydantic:', pydantic.__version__)
print('🎉 Stack GPU completo listo!')
"

echo "✅ Instalación completada!"
