"""
EndoCenter MLOps - FAISS Index Manager
Gestiona índice FAISS GPU para búsqueda vectorial rápida
Optimizado para RTX 5070
"""

import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from loguru import logger
from sqlalchemy.orm import Session

from endocenter.db.models import Embedding, DocumentChunk, MedicalDocument
from endocenter.config import settings


class FAISSIndexManager:
    """
    Gestiona índice FAISS para búsqueda vectorial eficiente
    Soporta CPU y GPU
    """
    
    def __init__(
        self,
        embedding_dimension: int = 384,  # MiniLM default
        use_gpu: bool = True,
        index_type: str = "IndexFlatL2"
    ):
        """
        Args:
            embedding_dimension: Dimensión de los embeddings
            use_gpu: Si usar GPU para el índice
            index_type: Tipo de índice FAISS (IndexFlatL2, IndexIVFFlat, etc.)
        """
        self.embedding_dimension = embedding_dimension
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        self.index_type = index_type
        
        self.index = None
        self.chunk_ids = []  # Mapeo de índice a chunk_id
        self.metadata = {}
        
        logger.info(f"FAISS Manager initialized - GPU: {self.use_gpu}")
    
    def create_index(self) -> bool:
        """Crea un nuevo índice FAISS vacío"""
        try:
            logger.info(f"Creating FAISS index: {self.index_type}")
            
            # Crear índice según tipo
            if self.index_type == "IndexFlatL2":
                # L2 distance (Euclidean)
                cpu_index = faiss.IndexFlatL2(self.embedding_dimension)
            elif self.index_type == "IndexFlatIP":
                # Inner product (cosine similarity si embeddings normalizados)
                cpu_index = faiss.IndexFlatIP(self.embedding_dimension)
            else:
                logger.warning(f"Unknown index type {self.index_type}, using IndexFlatL2")
                cpu_index = faiss.IndexFlatL2(self.embedding_dimension)
            
            # Mover a GPU si está disponible
            if self.use_gpu:
                logger.info("Moving index to GPU...")
                gpu_resources = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(gpu_resources, 0, cpu_index)
                logger.success("Index on GPU")
            else:
                self.index = cpu_index
                logger.info("Index on CPU")
            
            return True
        
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            return False
    
    def add_embeddings(
        self,
        embeddings: np.ndarray,
        chunk_ids: List[int]
    ) -> bool:
        """
        Agrega embeddings al índice
        
        Args:
            embeddings: Array numpy de shape (n, embedding_dim)
            chunk_ids: Lista de IDs de chunks correspondientes
        """
        if self.index is None:
            logger.error("Index not created. Call create_index() first")
            return False
        
        if len(embeddings) != len(chunk_ids):
            logger.error("Embeddings and chunk_ids length mismatch")
            return False
        
        try:
            # Asegurar tipo float32
            embeddings = embeddings.astype(np.float32)
            
            # Agregar al índice
            self.index.add(embeddings)
            self.chunk_ids.extend(chunk_ids)
            
            logger.info(f"Added {len(embeddings)} embeddings to index")
            logger.info(f"Total vectors in index: {self.index.ntotal}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error adding embeddings: {e}")
            return False
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Busca los k vectores más similares
        
        Args:
            query_embedding: Vector de consulta (1D array)
            k: Número de resultados
        
        Returns:
            (distances, indices) - Arrays de shape (1, k)
        """
        if self.index is None:
            logger.error("Index not created")
            return np.array([]), np.array([])
        
        try:
            # Asegurar shape correcto (1, dim)
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Asegurar tipo float32
            query_embedding = query_embedding.astype(np.float32)
            
            # Buscar
            distances, indices = self.index.search(query_embedding, k)
            
            return distances, indices
        
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return np.array([]), np.array([])
    
    def search_with_chunk_ids(
        self,
        query_embedding: np.ndarray,
        k: int = 10
    ) -> List[Dict]:
        """
        Busca y retorna resultados con chunk_ids
        
        Returns:
            Lista de dicts con {chunk_id, distance, rank}
        """
        distances, indices = self.search(query_embedding, k)
        
        if len(indices) == 0:
            return []
        
        results = []
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
            if idx < len(self.chunk_ids):
                results.append({
                    "chunk_id": self.chunk_ids[idx],
                    "distance": float(dist),
                    "similarity": 1.0 / (1.0 + float(dist)),  # Convertir distancia a similarity
                    "rank": rank
                })
        
        return results
    
    def save(self, filepath: Path) -> bool:
        """Guarda el índice a disco"""
        if self.index is None:
            logger.error("No index to save")
            return False
        
        try:
            logger.info(f"Saving index to {filepath}")
            
            # Si está en GPU, mover a CPU para guardar
            if self.use_gpu:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
            else:
                cpu_index = self.index
            
            # Guardar índice
            faiss.write_index(cpu_index, str(filepath))
            
            # Guardar metadata (chunk_ids mapping)
            metadata_path = filepath.with_suffix('.pkl')
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'chunk_ids': self.chunk_ids,
                    'embedding_dimension': self.embedding_dimension,
                    'index_type': self.index_type,
                    'total_vectors': self.index.ntotal
                }, f)
            
            logger.success(f"Index saved: {self.index.ntotal} vectors")
            return True
        
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            return False
    
    def load(self, filepath: Path) -> bool:
        """Carga el índice desde disco"""
        if not filepath.exists():
            logger.error(f"Index file not found: {filepath}")
            return False
        
        try:
            logger.info(f"Loading index from {filepath}")
            
            # Cargar índice
            cpu_index = faiss.read_index(str(filepath))
            
            # Mover a GPU si está disponible
            if self.use_gpu:
                gpu_resources = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(gpu_resources, 0, cpu_index)
            else:
                self.index = cpu_index
            
            # Cargar metadata
            metadata_path = filepath.with_suffix('.pkl')
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                    self.chunk_ids = metadata['chunk_ids']
                    self.embedding_dimension = metadata['embedding_dimension']
                    self.metadata = metadata
            
            logger.success(f"Index loaded: {self.index.ntotal} vectors")
            return True
        
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
    
    def build_from_database(
        self,
        db: Session,
        document_ids: Optional[List[int]] = None
    ) -> bool:
        """
        Construye índice desde embeddings en la base de datos
        
        Args:
            db: Database session
            document_ids: Lista de IDs de documentos (None = todos)
        """
        try:
            # Query embeddings
            query = db.query(Embedding).join(DocumentChunk)
            
            if document_ids:
                query = query.join(MedicalDocument).filter(
                    MedicalDocument.id.in_(document_ids)
                )
            
            embeddings_records = query.all()
            
            if not embeddings_records:
                logger.warning("No embeddings found in database")
                return False
            
            logger.info(f"Loading {len(embeddings_records)} embeddings from DB")
            
            # Extraer embeddings y chunk_ids
            embeddings_list = []
            chunk_ids_list = []
            
            for emb_record in embeddings_records:
                # Convertir bytes a numpy array
                embedding_vector = np.frombuffer(
                    emb_record.vector_data,
                    dtype=np.float32
                )
                
                embeddings_list.append(embedding_vector)
                chunk_ids_list.append(emb_record.chunk_id)
            
            # Convertir a numpy array
            embeddings_array = np.vstack(embeddings_list)
            
            # Crear índice si no existe
            if self.index is None:
                self.create_index()
            
            # Agregar embeddings
            success = self.add_embeddings(embeddings_array, chunk_ids_list)
            
            if success:
                logger.success(f"Index built with {len(embeddings_list)} vectors")
            
            return success
        
        except Exception as e:
            logger.error(f"Error building index from database: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Obtiene estadísticas del índice"""
        if self.index is None:
            return {"status": "not_initialized"}
        
        return {
            "status": "ready",
            "total_vectors": self.index.ntotal,
            "embedding_dimension": self.embedding_dimension,
            "index_type": self.index_type,
            "using_gpu": self.use_gpu,
            "chunk_ids_count": len(self.chunk_ids)
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_faiss_index_from_db(
    db: Session,
    save_path: Optional[Path] = None
) -> FAISSIndexManager:
    """Helper para crear índice desde DB"""
    manager = FAISSIndexManager(use_gpu=settings.faiss_gpu_enabled)
    
    success = manager.build_from_database(db)
    
    if not success:
        logger.error("Failed to build index")
        return None
    
    if save_path:
        manager.save(save_path)
    
    return manager


# =============================================================================
# DEMO/TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing FAISSIndexManager...")
    print("=" * 60)
    
    # Test básico con embeddings dummy
    print("\n1. Creating index...")
    manager = FAISSIndexManager(embedding_dimension=384, use_gpu=True)
    manager.create_index()
    
    print(f"   Stats: {manager.get_stats()}")
    
    # Crear embeddings dummy
    print("\n2. Adding dummy embeddings...")
    dummy_embeddings = np.random.rand(100, 384).astype(np.float32)
    chunk_ids = list(range(1, 101))
    
    success = manager.add_embeddings(dummy_embeddings, chunk_ids)
    print(f"   Added: {success}")
    print(f"   Stats: {manager.get_stats()}")
    
    # Test búsqueda
    print("\n3. Testing search...")
    query = np.random.rand(384).astype(np.float32)
    results = manager.search_with_chunk_ids(query, k=5)
    
    print(f"   Found {len(results)} results:")
    for result in results:
        print(f"     Chunk {result['chunk_id']}: similarity={result['similarity']:.4f}")
    
    # Test save/load
    print("\n4. Testing save/load...")
    test_path = settings.embeddings_dir / "test_index.faiss"
    manager.save(test_path)
    
    # Cargar
    new_manager = FAISSIndexManager()
    new_manager.load(test_path)
    print(f"   Loaded stats: {new_manager.get_stats()}")
    
    print("\n" + "=" * 60)
    print("FAISSIndexManager ready!")