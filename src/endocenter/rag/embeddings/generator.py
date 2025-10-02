"""
EndoCenter MLOps - Embedding Generator
Genera embeddings GPU-optimizados para chunks de texto médico
Optimizado para RTX 5070
"""

import torch
import numpy as np
import time
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from sentence_transformers import SentenceTransformer
from loguru import logger
from sqlalchemy.orm import Session

from endocenter.db.models import DocumentChunk, Embedding
from endocenter.config import settings, gpu_manager


class EmbeddingGenerator:
    """
    Genera embeddings vectoriales usando sentence-transformers
    Optimizado para GPU RTX 5070 con batch processing
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None
    ):
        """
        Args:
            model_name: Nombre del modelo de HuggingFace (default: desde config)
            device: 'cuda' o 'cpu' (default: desde config)
            batch_size: Tamaño de batch (default: desde config)
        """
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device
        self.batch_size = batch_size or settings.embedding_batch_size
        
        # Info GPU
        self.gpu_info = gpu_manager.get_device_info()
        
        # Cargar modelo
        self.model = None
        self.embedding_dim = None
        self._load_model()
    
    def _load_model(self):
        """Carga el modelo de embeddings"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            logger.info(f"Device: {self.device}")
            logger.info(f"Batch size: {self.batch_size}")
            
            # Cargar modelo
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # Obtener dimensión de embeddings
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
            # Optimizaciones GPU si está disponible
            if self.device == "cuda" and torch.cuda.is_available():
                # Mixed precision para RTX 5070
                if settings.embedding_precision == "fp16":
                    self.model.half()
                    logger.info("Using FP16 precision")
                
                # Optimizaciones CUDA
                torch.backends.cudnn.benchmark = True
                
                logger.success(
                    f"Model loaded on GPU: {self.gpu_info['gpu_name']}"
                )
            else:
                logger.warning("Running on CPU (GPU not available)")
            
            logger.success(
                f"Model ready - Embedding dimension: {self.embedding_dim}"
            )
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Genera embedding para un solo texto
        
        Returns:
            Vector numpy de dimensión embedding_dim
        """
        if not text or not text.strip():
            logger.warning("Empty text provided")
            return None
        
        try:
            # Generar embedding
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True  # L2 normalization para cosine similarity
            )
            
            return embedding
        
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def generate_embeddings_batch(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Genera embeddings para múltiples textos en batch
        
        Returns:
            (embeddings_array, metadata)
        """
        if not texts:
            logger.warning("No texts provided")
            return None, {}
        
        # Filtrar textos vacíos
        valid_texts = [t for t in texts if t and t.strip()]
        
        if not valid_texts:
            logger.warning("No valid texts after filtering")
            return None, {}
        
        try:
            start_time = time.time()
            
            logger.info(f"Generating embeddings for {len(valid_texts)} texts")
            
            # Generar embeddings en batch
            embeddings = self.model.encode(
                valid_texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
                normalize_embeddings=True
            )
            
            # Calcular tiempo
            elapsed_time = (time.time() - start_time) * 1000  # ms
            
            # Metadata
            metadata = {
                "num_embeddings": len(embeddings),
                "embedding_dimension": embeddings.shape[1],
                "total_time_ms": elapsed_time,
                "avg_time_per_text_ms": elapsed_time / len(valid_texts),
                "batch_size": self.batch_size,
                "device": self.device,
                "model": self.model_name,
                "gpu_used": self.device == "cuda" and torch.cuda.is_available()
            }
            
            # GPU memory info si está disponible
            if metadata["gpu_used"]:
                metadata["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
                metadata["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
            
            logger.success(
                f"Generated {len(embeddings)} embeddings in {elapsed_time:.2f}ms "
                f"({metadata['avg_time_per_text_ms']:.2f}ms per text)"
            )
            
            return embeddings, metadata
        
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return None, {}
    
    def save_embeddings_to_db(
        self,
        db: Session,
        chunk_ids: List[int],
        embeddings: np.ndarray,
        metadata: Dict
    ) -> Tuple[int, Optional[str]]:
        """
        Guarda embeddings en la base de datos
        
        Args:
            db: Database session
            chunk_ids: Lista de IDs de chunks
            embeddings: Array de embeddings
            metadata: Metadata de generación
        
        Returns:
            (saved_count, error_message)
        """
        if len(chunk_ids) != len(embeddings):
            return 0, "Mismatch between chunk_ids and embeddings length"
        
        try:
            saved_count = 0
            
            for chunk_id, embedding_vector in zip(chunk_ids, embeddings):
                # Verificar que el chunk existe
                chunk = db.query(DocumentChunk).filter(
                    DocumentChunk.id == chunk_id
                ).first()
                
                if not chunk:
                    logger.warning(f"Chunk {chunk_id} not found, skipping")
                    continue
                
                # Convertir embedding a bytes
                embedding_bytes = embedding_vector.tobytes()
                
                # Crear registro de embedding
                embedding_record = Embedding(
                    chunk_id=chunk_id,
                    model_name=self.model_name,
                    embedding_dimension=self.embedding_dim,
                    vector_data=embedding_bytes,
                    generated_with_gpu=metadata.get("gpu_used", False),
                    gpu_device_used=self.gpu_info.get("gpu_name") if metadata.get("gpu_used") else None,
                    generation_time_ms=metadata.get("avg_time_per_text_ms"),
                    batch_size_used=self.batch_size,
                    precision_used=settings.embedding_precision
                )
                
                db.add(embedding_record)
                
                # Actualizar chunk con info de embedding
                chunk.embedding_model = self.model_name
                chunk.embedding_created_at = torch.cuda.Event(enable_timing=False) if metadata.get("gpu_used") else None
                
                saved_count += 1
            
            db.commit()
            
            logger.success(f"Saved {saved_count} embeddings to database")
            
            return saved_count, None
        
        except Exception as e:
            db.rollback()
            error_msg = f"Error saving embeddings: {str(e)}"
            logger.error(error_msg)
            return 0, error_msg
    
    def process_chunks(
        self,
        db: Session,
        document_id: int,
        show_progress: bool = True
    ) -> Tuple[int, Optional[str]]:
        """
        Procesa todos los chunks de un documento y genera embeddings
        
        Returns:
            (processed_count, error_message)
        """
        try:
            # Obtener chunks sin embeddings
            chunks = db.query(DocumentChunk).filter(
                DocumentChunk.document_id == document_id,
                ~DocumentChunk.embeddings.any()  # Sin embeddings
            ).all()
            
            if not chunks:
                return 0, "No chunks found without embeddings"
            
            logger.info(f"Processing {len(chunks)} chunks from document {document_id}")
            
            # Extraer textos e IDs
            texts = [chunk.content for chunk in chunks]
            chunk_ids = [chunk.id for chunk in chunks]
            
            # Generar embeddings
            embeddings, metadata = self.generate_embeddings_batch(
                texts,
                show_progress=show_progress
            )
            
            if embeddings is None:
                return 0, "Failed to generate embeddings"
            
            # Guardar en DB
            saved_count, error = self.save_embeddings_to_db(
                db, chunk_ids, embeddings, metadata
            )
            
            return saved_count, error
        
        except Exception as e:
            error_msg = f"Error processing chunks: {str(e)}"
            logger.error(error_msg)
            return 0, error_msg
    
    def get_embedding_stats(self) -> Dict:
        """Obtiene estadísticas del generador"""
        stats = {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "device": self.device,
            "batch_size": self.batch_size,
            "precision": settings.embedding_precision,
        }
        
        if self.device == "cuda" and torch.cuda.is_available():
            stats.update({
                "gpu_name": self.gpu_info.get("gpu_name"),
                "gpu_available_memory_gb": self.gpu_info.get("available_memory", 0) / 1024**3,
                "cuda_version": torch.version.cuda,
                "pytorch_version": torch.__version__,
            })
        
        return stats


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def generate_embedding_for_text(text: str) -> Optional[np.ndarray]:
    """Helper function para generar embedding de un texto"""
    generator = EmbeddingGenerator()
    return generator.generate_embedding(text)


def generate_embeddings_for_texts(texts: List[str]) -> Optional[np.ndarray]:
    """Helper function para generar embeddings de múltiples textos"""
    generator = EmbeddingGenerator()
    embeddings, _ = generator.generate_embeddings_batch(texts)
    return embeddings


def process_document_chunks(db: Session, document_id: int) -> int:
    """Helper function para procesar chunks de un documento"""
    generator = EmbeddingGenerator()
    count, error = generator.process_chunks(db, document_id)
    
    if error:
        logger.error(f"Error processing document: {error}")
        return 0
    
    return count


# =============================================================================
# DEMO/TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing EmbeddingGenerator...")
    print("=" * 60)
    
    # Crear generador
    generator = EmbeddingGenerator()
    
    # Mostrar stats
    print("\n1. Generator stats:")
    stats = generator.get_embedding_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test embedding simple
    print("\n2. Testing single embedding...")
    text = "La diabetes mellitus tipo 2 es una enfermedad metabólica crónica."
    embedding = generator.generate_embedding(text)
    
    if embedding is not None:
        print(f"   Generated embedding with shape: {embedding.shape}")
        print(f"   First 5 values: {embedding[:5]}")
    
    # Test batch
    print("\n3. Testing batch embeddings...")
    texts = [
        "Diabetes tipo 2 es una enfermedad metabólica.",
        "El hipotiroidismo afecta la función tiroidea.",
        "La osteoporosis reduce la densidad ósea.",
    ]
    
    embeddings, metadata = generator.generate_embeddings_batch(texts, show_progress=True)
    
    if embeddings is not None:
        print(f"   Generated {len(embeddings)} embeddings")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Time: {metadata['total_time_ms']:.2f}ms")
        print(f"   Avg per text: {metadata['avg_time_per_text_ms']:.2f}ms")
        if metadata['gpu_used']:
            print(f"   GPU memory: {metadata['gpu_memory_allocated_mb']:.2f}MB")
    
    print("\n" + "=" * 60)
    print("EmbeddingGenerator ready!")