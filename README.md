# 🚀 EndoCenter MLOps - GPU Accelerated

AI-powered endocrinology diagnosis system optimized for RTX 5070 + CUDA 12.9.

## ⚡ Performance Optimized

- **GPU**: RTX 5070 (12GB VRAM)
- **CUDA**: 12.9 + Toolkit
- **PyTorch**: 2.9.0+ with CUDA support
- **FAISS**: GPU-accelerated vector search
- **Batch Size**: 128 (optimized for RTX 5070)

## 🚀 Quick Start

```bash
# 1. Setup environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
make install-dev

# 3. Test GPU setup
make gpu-test

# 4. Create configuration
make env

# 5. Run API
make run
```

## 📊 GPU Performance

With RTX 5070 optimization:
- **Embedding generation**: <200ms for 100 texts
- **Vector search**: <5ms for 10K vectors
- **End-to-end RAG**: <500ms response time

## 🏗️ Architecture

```
endocenter-mlops/
├── src/endocenter/          # Main package
│   ├── rag/                 # GPU-optimized RAG
│   ├── api/                 # FastAPI application
│   ├── db/                  # Database models
│   └── utils/               # Utilities
├── data/                    # Data storage
├── config/                  # GPU-optimized configs
├── tests/                   # Test suite
└── infrastructure/          # Deployment
```

## 🔧 Development

```bash
# Format code
make format

# Run tests
make test

# Monitor GPU
make gpu-monitor

# Clean cache
make clean
```

## 🐳 Docker Support

GPU-optimized containers with CUDA runtime support.

## 📈 Monitoring

Real-time GPU monitoring and performance metrics.
