"""
Configuración centralizada para EndoCenter MLOps
Optimizada para RTX 5070 + CUDA 12.9 + PyTorch nightly
"""

import os
import yaml
import torch
from pathlib import Path
from typing import Dict, Optional, List
from pydantic import validator, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuración principal GPU-optimizada para RTX 5070"""
    
    # =============================================================================
    # APPLICATION CONFIG
    # =============================================================================
    app_name: str = "EndoCenter MLOps"
    app_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = True
    
    # =============================================================================
    # API CONFIG
    # =============================================================================
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_prefix: str = "/api/v1"
    
    # =============================================================================
    # GPU CONFIG - RTX 5070 Optimized
    # =============================================================================
    gpu_enabled: bool = True
    cuda_device: str = "cuda:0"
    gpu_memory_limit: int = 11000  # 11GB of 12GB VRAM
    gpu_optimization_level: str = "high"
    
    # =============================================================================
    # MODEL CONFIG - GPU Optimized
    # =============================================================================
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_device: str = "cuda"
    embedding_batch_size: int = 128  # Large batch for RTX 5070
    embedding_precision: str = "fp16"  # Mixed precision
    
    # RAG Config
    top_k_results: int = 10
    max_chunk_size: int = 500
    chunk_overlap: int = 100
    similarity_threshold: float = 0.3
    
    # =============================================================================
    # FAISS CONFIG - GPU Optimized
    # =============================================================================
    faiss_gpu_enabled: bool = True
    faiss_gpu_memory_limit: int = 8000  # MB for FAISS indices
    faiss_index_type: str = "IndexFlatL2"
    faiss_training_batch_size: int = 10000
    
    # =============================================================================
    # LLM CONFIG
    # =============================================================================
    llm_api_url: str = "http://localhost:11434/api/generate"
    llm_model: str = "llama3"
    llm_timeout: int = 30
    llm_max_retries: int = 3
    
    # =============================================================================
    # DATA PATHS
    # =============================================================================
    data_dir: Path = Path("./data")
    embeddings_dir: Path = Path("./data/embeddings")
    chunks_dir: Path = Path("./data/chunks")
    preprocessed_dir: Path = Path("./data/preprocessed")
    raw_dir: Path = Path("./data/raw")
    
    # =============================================================================
    # DATABASE CONFIG
    # =============================================================================
    db_host: str = 'localhost'
    db_port: int = 5433
    db_name: str = 'endocenter'
    db_user: str = 'endocenter_user'
    db_password: str = 'endocenter_pass'
    database_url: str = Field(
        default="postgresql+psycopg2://endocenter_user:endocenter_pass@localhost:5433/endocenter",
        description="Database connection URL",
    )
    
    # =============================================================================
    # MONITORING CONFIG
    # =============================================================================
    log_level: str = "INFO"
    log_format: str = "detailed"
    metrics_enabled: bool = True
    gpu_monitoring: bool = True
    performance_logging: bool = True
    
    # =============================================================================
    # DISEASES CONFIG
    # =============================================================================
    diseases: Dict[str, str] = {
        "cushing": "Síndrome de Cushing",
        "diabetes_tipo_2": "Diabetes tipo 2",
        "hipotiroidismo": "Hipotiroidismo",
        "insuficiencia_suprarrenal": "Insuficiencia suprarrenal",
        "osteoporosis": "Osteoporosis",
        "sop_sindrome_ovario_poliquistico": "Síndrome de ovario poliquístico (SOP)",
        "greenspan": "Libro de Greenspan"
    }
    
    @validator('data_dir', 'embeddings_dir', 'chunks_dir', 'preprocessed_dir', 'raw_dir')
    def ensure_path_exists(cls, v):
        """Crear directorios si no existen"""
        if isinstance(v, str):
            v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @validator('gpu_enabled', always=True)
    def check_gpu_availability(cls, v):
        """Verificar disponibilidad de GPU"""
        if v and not torch.cuda.is_available():
            print("⚠️ GPU solicitada pero CUDA no disponible, usando CPU")
            return False
        return v
    
    @validator('embedding_device', always=True)
    def validate_device(cls, v, values):
        """Validar device según disponibilidad"""
        if values.get('gpu_enabled', False) and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    @validator('embedding_batch_size', always=True)
    def adjust_batch_size(cls, v, values):
        """Ajustar batch size según device"""
        if values.get('embedding_device') == "cpu":
            return min(v, 32)  # Smaller batch for CPU
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class GPUManager:
    """Manager para operaciones GPU optimizadas"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self._device_info = None
        self._gpu_resources = None
        
    def get_device_info(self) -> Dict:
        """Obtener información detallada del GPU"""
        if self._device_info is None:
            self._device_info = self._collect_device_info()
        return self._device_info
    
    def _collect_device_info(self) -> Dict:
        """Recopilar información del dispositivo"""
        info = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": None,
            "gpu_name": None,
            "compute_capability": None,
            "total_memory": 0,
            "available_memory": 0,
            "pytorch_version": torch.__version__,
        }
        
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            info.update({
                "current_device": current_device,
                "gpu_name": torch.cuda.get_device_name(current_device),
                "compute_capability": torch.cuda.get_device_capability(current_device),
                "total_memory": torch.cuda.get_device_properties(current_device).total_memory,
                "available_memory": torch.cuda.get_device_properties(current_device).total_memory - torch.cuda.memory_allocated(current_device),
            })
        
        return info
    
    def optimize_for_inference(self):
        """Optimizar GPU para inferencia"""
        if torch.cuda.is_available():
            # Optimizaciones específicas para RTX 5070
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
            torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
            
            # Set memory management
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                memory_fraction = self.settings.gpu_memory_limit / 12000  # 12GB total VRAM
                torch.cuda.set_per_process_memory_fraction(memory_fraction)
            
            print(f"✅ GPU optimizado para inferencia: {self.get_device_info()['gpu_name']}")
        
    def clear_cache(self):
        """Limpiar cache GPU"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 Cache GPU limpiado")
    
    def get_memory_stats(self) -> Dict:
        """Obtener estadísticas de memoria GPU"""
        if not torch.cuda.is_available():
            return {"error": "CUDA no disponible"}
        
        return {
            "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
            "reserved": torch.cuda.memory_reserved() / 1024**3,   # GB
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,  # GB
        }


class ConfigManager:
    """Manager para cargar configuraciones desde YAML y combinar con Settings"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.config_dir = self.project_root / "config"
        self._settings = None
        self._gpu_manager = None
    
    def load_yaml_config(self, environment: str = None) -> Dict:
        """Cargar configuración desde archivo YAML"""
        if environment is None:
            environment = os.getenv("ENVIRONMENT", "development")
        
        config_file = self.config_dir / f"{environment}.yaml"
        
        if not config_file.exists():
            print(f"⚠️ Config file not found: {config_file}, using defaults")
            return {}
        
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get_settings(self, environment: str = None) -> Settings:
        """Obtener configuración completa"""
        if self._settings is None:
            yaml_config = self.load_yaml_config(environment)
            flattened_config = self._flatten_dict(yaml_config)
            self._settings = Settings(**flattened_config)
        return self._settings
    
    def get_gpu_manager(self) -> GPUManager:
        """Obtener GPU manager"""
        if self._gpu_manager is None:
            settings = self.get_settings()
            self._gpu_manager = GPUManager(settings)
        return self._gpu_manager
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Aplanar diccionario anidado para Pydantic"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


# =============================================================================
# INSTANCIAS GLOBALES
# =============================================================================

# Singleton instances
_config_manager = ConfigManager()

def get_settings(environment: str = None) -> Settings:
    """Función principal para obtener configuración"""
    return _config_manager.get_settings(environment)

def get_gpu_manager() -> GPUManager:
    """Función principal para obtener GPU manager"""
    return _config_manager.get_gpu_manager()

def get_device_config() -> Dict:
    """Obtener configuración optimizada de device"""
    gpu_manager = get_gpu_manager()
    device_info = gpu_manager.get_device_info()
    settings = get_settings()
    
    return {
        "device": settings.embedding_device,
        "batch_size": settings.embedding_batch_size,
        "precision": settings.embedding_precision,
        "gpu_enabled": settings.gpu_enabled,
        "gpu_info": device_info,
        "faiss_gpu": settings.faiss_gpu_enabled and device_info["cuda_available"],
    }

# Instancia global para importación directa
settings = get_settings()
gpu_manager = get_gpu_manager()


# =============================================================================
# UTILITIES
# =============================================================================

def get_disease_display_name(disease_key: str) -> str:
    """Obtener nombre de display para una enfermedad"""
    return settings.diseases.get(disease_key, disease_key.replace('_', ' ').title())

def is_development() -> bool:
    """Verificar si estamos en ambiente de desarrollo"""
    return settings.environment == "development"

def is_production() -> bool:
    """Verificar si estamos en ambiente de producción"""
    return settings.environment == "production"

def setup_gpu_optimizations():
    """Setup completo de optimizaciones GPU"""
    gpu_manager.optimize_for_inference()
    
    # Print configuration summary
    device_config = get_device_config()
    print("🚀 EndoCenter MLOps - GPU Configuration")
    print("=" * 50)
    print(f"📱 App: {settings.app_name} v{settings.app_version}")
    print(f"🌍 Environment: {settings.environment}")
    print(f"🎯 Device: {device_config['device']}")
    print(f"🔢 Batch Size: {device_config['batch_size']}")
    print(f"🎮 GPU: {device_config['gpu_info']['gpu_name'] if device_config['gpu_info']['cuda_available'] else 'CPU'}")
    print(f"💾 VRAM: {device_config['gpu_info']['total_memory'] / 1024**3:.1f}GB" if device_config['gpu_info']['cuda_available'] else "N/A")
    print(f"🔍 FAISS GPU: {device_config['faiss_gpu']}")
    print(f"🏥 Diseases: {len(settings.diseases)}")
    print("=" * 50)


# =============================================================================
# DEMO/TEST FUNCTION
# =============================================================================

if __name__ == "__main__":
    # Demostración y test de configuración
    print("🔧 Testing EndoCenter Configuration...")
    
    # Setup GPU optimizations
    setup_gpu_optimizations()
    
    # Test device config
    device_config = get_device_config()
    print(f"\n📊 Device Config: {device_config}")
    
    # Test GPU memory
    if torch.cuda.is_available():
        memory_stats = gpu_manager.get_memory_stats()
        print(f"💾 GPU Memory: {memory_stats}")
    
    # Test disease mapping
    print(f"\n🏥 Disease example: {get_disease_display_name('diabetes_tipo_2')}")
    
    print("\n✅ Configuration test completed!")