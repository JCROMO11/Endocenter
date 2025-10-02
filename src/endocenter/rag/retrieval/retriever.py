"""
EndoCenter MLOps - Retriever
Sistema de recuperación de documentos para RAG
"""

from typing import List, Dict, Optional
from pathlib import Path
import numpy as np
from loguru import logger
from sqlalchemy.orm import Session

from endocenter.db.models import DocumentChunk, MedicalDocument, Embedding
from endocenter.rag.embeddings.generator import EmbeddingGenerator
from endocenter.rag.retrieval.faiss_index import FAISSIndexManager
from endocenter.config import settings


class Retriever:
    """
    Sistema de recuperación para RAG
    Combina búsqueda vectorial con metadata filtering
    """
    
    def __init__(
        self,
        db: Session,
        index_manager: Optional[FAISSIndexManager] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None
    ):
        """
        Args:
            db: Database session
            index_manager: FAISS index manager (crea uno si es None)
            embedding_generator: Generator para query embeddings
        """
        self.db = db
        self.index_manager = index_manager
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        
        # Cargar o crear índice
        if self.index_manager is None:
            self._initialize_index()
    
    def _initialize_index(self):
        """Inicializa o carga índice FAISS"""
        index_path = settings.embeddings_dir / "faiss_index.faiss"
        
        self.index_manager = FAISSIndexManager(
            embedding_dimension=self.embedding_generator.embedding_dim,
            use_gpu=settings.faiss_gpu_enabled
        )
        
        # Intentar cargar índice existente
        if index_path.exists():
            logger.info("Loading existing FAISS index...")
            success = self.index_manager.load(index_path)
            if success:
                logger.success("Index loaded successfully")
                return
        
        # Construir desde DB si no existe
        logger.info("Building new FAISS index from database...")
        success = self.index_manager.build_from_database(self.db)
        
        if success:
            self.index_manager.save(index_path)
            logger.success("New index created and saved")
        else:
            logger.warning("Failed to build index")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Recupera chunks más relevantes para una consulta
        
        Args:
            query: Texto de consulta
            top_k: Número de resultados
            filters: Filtros opcionales (disease, document_type, etc.)
        
        Returns:
            Lista de chunks con metadata y scores
        """
        try:
            # 1. Generar embedding de la query
            logger.info(f"Retrieving for query: {query[:100]}...")
            query_embedding = self.embedding_generator.generate_embedding(query)
            
            if query_embedding is None:
                logger.error("Failed to generate query embedding")
                return []
            
            # 2. Buscar en FAISS
            # Solicitar más resultados para aplicar filtros después
            search_k = top_k * 3 if filters else top_k
            
            results = self.index_manager.search_with_chunk_ids(
                query_embedding,
                k=search_k
            )
            
            if not results:
                logger.warning("No results from FAISS search")
                return []
            
            # 3. Enriquecer con metadata de DB
            enriched_results = self._enrich_results(results)
            
            # 4. Aplicar filtros si existen
            if filters:
                enriched_results = self._apply_filters(enriched_results, filters)
            
            # 5. Limitar a top_k
            enriched_results = enriched_results[:top_k]
            
            logger.success(f"Retrieved {len(enriched_results)} relevant chunks")
            
            return enriched_results
        
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
    
    def _enrich_results(self, results: List[Dict]) -> List[Dict]:
        """Enriquece resultados con información de la base de datos"""
        enriched = []
        
        chunk_ids = [r["chunk_id"] for r in results]
        
        # Obtener chunks con sus documentos en una query
        chunks = self.db.query(DocumentChunk).join(MedicalDocument).filter(
            DocumentChunk.id.in_(chunk_ids)
        ).all()
        
        # Crear mapeo chunk_id -> chunk
        chunks_map = {chunk.id: chunk for chunk in chunks}
        
        # Enriquecer cada resultado
        for result in results:
            chunk = chunks_map.get(result["chunk_id"])
            
            if chunk:
                enriched.append({
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "similarity": result["similarity"],
                    "distance": result["distance"],
                    "rank": result["rank"],
                    "page_number": chunk.page_number,
                    "section_title": chunk.section_title,
                    "document_title": chunk.document.title,
                    "document_type": chunk.document.document_type,
                    "medical_specialty": chunk.document.medical_specialty,
                    "diseases_covered": chunk.document.diseases_covered,
                })
        
        return enriched
    
    def _apply_filters(
        self,
        results: List[Dict],
        filters: Dict
    ) -> List[Dict]:
        """Aplica filtros a los resultados"""
        filtered = results
        
        # Filtro por enfermedad
        if "disease" in filters:
            disease = filters["disease"]
            filtered = [
                r for r in filtered
                if r["diseases_covered"] and disease in r["diseases_covered"]
            ]
        
        # Filtro por tipo de documento
        if "document_type" in filters:
            doc_type = filters["document_type"]
            filtered = [
                r for r in filtered
                if r["document_type"] == doc_type
            ]
        
        # Filtro por especialidad
        if "medical_specialty" in filters:
            specialty = filters["medical_specialty"]
            filtered = [
                r for r in filtered
                if r["medical_specialty"] == specialty
            ]
        
        # Filtro por similarity mínima
        if "min_similarity" in filters:
            min_sim = filters["min_similarity"]
            filtered = [
                r for r in filtered
                if r["similarity"] >= min_sim
            ]
        
        return filtered
    
    def retrieve_by_disease(
        self,
        query: str,
        disease: str,
        top_k: int = 10
    ) -> List[Dict]:
        """Helper para búsqueda específica por enfermedad"""
        return self.retrieve(
            query,
            top_k=top_k,
            filters={"disease": disease}
        )
    
    def get_chunk_context(
        self,
        chunk_id: int,
        include_neighbors: bool = True
    ) -> Dict:
        """
        Obtiene un chunk con su contexto (chunks anterior y siguiente)
        Útil para mostrar contexto completo
        """
        chunk = self.db.query(DocumentChunk).filter(
            DocumentChunk.id == chunk_id
        ).first()
        
        if not chunk:
            return None
        
        result = {
            "chunk": {
                "id": chunk.id,
                "content": chunk.content,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title
            },
            "document": {
                "id": chunk.document.id,
                "title": chunk.document.title,
                "type": chunk.document.document_type
            }
        }
        
        if include_neighbors:
            # Chunk anterior
            prev_chunk = self.db.query(DocumentChunk).filter(
                DocumentChunk.document_id == chunk.document_id,
                DocumentChunk.chunk_index == chunk.chunk_index - 1
            ).first()
            
            # Chunk siguiente
            next_chunk = self.db.query(DocumentChunk).filter(
                DocumentChunk.document_id == chunk.document_id,
                DocumentChunk.chunk_index == chunk.chunk_index + 1
            ).first()
            
            result["context"] = {
                "previous": prev_chunk.content if prev_chunk else None,
                "next": next_chunk.content if next_chunk else None
            }
        
        return result
    
    def rebuild_index(self) -> bool:
        """Reconstruye el índice FAISS desde la base de datos"""
        logger.info("Rebuilding FAISS index...")
        
        # Crear nuevo índice
        self.index_manager = FAISSIndexManager(
            embedding_dimension=self.embedding_generator.embedding_dim,
            use_gpu=settings.faiss_gpu_enabled
        )
        
        # Construir desde DB
        success = self.index_manager.build_from_database(self.db)
        
        if success:
            # Guardar
            index_path = settings.embeddings_dir / "faiss_index.faiss"
            self.index_manager.save(index_path)
            logger.success("Index rebuilt successfully")
        
        return success


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def search_medical_documents(
    db: Session,
    query: str,
    top_k: int = 10
) -> List[Dict]:
    """Helper simple para búsqueda"""
    retriever = Retriever(db)
    return retriever.retrieve(query, top_k=top_k)


# =============================================================================
# DEMO/TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing Retriever...")
    print("=" * 60)
    
    from endocenter.db.database import get_db_context
    
    with get_db_context() as db:
        # Verificar que hay embeddings en DB
        emb_count = db.query(Embedding).count()
        print(f"\nEmbeddings in database: {emb_count}")
        
        if emb_count > 0:
            print("\nInitializing retriever...")
            retriever = Retriever(db)
            
            print(f"Index stats: {retriever.index_manager.get_stats()}")
            
            # Test búsqueda
            print("\n2. Testing retrieval...")
            test_query = "¿Cuáles son los síntomas de diabetes tipo 2?"
            
            results = retriever.retrieve(test_query, top_k=5)
            
            print(f"\nQuery: {test_query}")
            print(f"Results: {len(results)}")
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Similarity: {result['similarity']:.4f}")
                print(f"   Document: {result['document_title']}")
                print(f"   Content: {result['content'][:200]}...")
        else:
            print("\nNo embeddings found in database.")
            print("Run batch processor first to create embeddings.")
    
    print("\n" + "=" * 60)
    print("Retriever ready!")