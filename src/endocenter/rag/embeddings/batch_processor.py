"""
EndoCenter MLOps - Batch Processor
Procesa múltiples documentos en batch con optimización GPU
"""

from typing import List, Dict, Optional
from pathlib import Path
from loguru import logger
from sqlalchemy.orm import Session

from endocenter.db.models import MedicalDocument, DocumentStatus, DocumentChunk
from endocenter.rag.preprocessing.document_loader import DocumentLoader
from endocenter.rag.preprocessing.pdf_processor import PDFProcessor
from endocenter.rag.preprocessing.text_cleaner import TextCleaner
from endocenter.rag.preprocessing.chunker import TextChunker
from endocenter.rag.embeddings.generator import EmbeddingGenerator
from endocenter.config import settings


class BatchProcessor:
    """
    Procesa documentos completos: carga → extracción → limpieza → chunking → embeddings
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.doc_loader = DocumentLoader(db)
        self.pdf_processor = PDFProcessor()
        self.text_cleaner = TextCleaner()
        self.chunker = TextChunker(
            chunk_size=settings.max_chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        self.embedding_gen = EmbeddingGenerator()
    
    def process_single_document(
        self,
        pdf_path: Path,
        uploaded_by: Optional[int] = None,
        generate_embeddings: bool = True
    ) -> Dict:
        """
        Procesa un documento completo end-to-end
        
        Returns:
            Dict con resultados del procesamiento
        """
        result = {
            "success": False,
            "document_id": None,
            "stages": {},
            "error": None
        }
        
        try:
            # Stage 1: Cargar documento
            logger.info(f"Stage 1/5: Loading document {pdf_path.name}")
            doc, error = self.doc_loader.load_document(
                pdf_path,
                uploaded_by=uploaded_by
            )
            
            if error:
                result["error"] = f"Loading failed: {error}"
                return result
            
            result["document_id"] = doc.id
            result["stages"]["loading"] = "success"
            
            # Stage 2: Extraer texto
            logger.info(f"Stage 2/5: Extracting text from PDF")
            text, metadata, error = self.pdf_processor.extract_text_from_pdf(pdf_path)
            
            if error:
                self.doc_loader.update_document_status(
                    doc.id, DocumentStatus.ERROR, error
                )
                result["error"] = f"Extraction failed: {error}"
                return result
            
            # Actualizar metadata del documento
            doc.page_count = metadata.get("page_count")
            doc.word_count = metadata.get("word_count")
            doc.content_preview = self.pdf_processor.get_text_preview(pdf_path)
            self.db.commit()
            
            result["stages"]["extraction"] = {
                "status": "success",
                "pages": metadata.get("page_count"),
                "words": metadata.get("word_count")
            }
            
            # Stage 3: Limpiar texto
            logger.info(f"Stage 3/5: Cleaning text")
            cleaned_text = self.text_cleaner.clean_text(text)
            
            result["stages"]["cleaning"] = {
                "status": "success",
                "original_length": len(text),
                "cleaned_length": len(cleaned_text)
            }
            
            # Stage 4: Chunking
            logger.info(f"Stage 4/5: Creating chunks")
            chunks = self.chunker.chunk_text(cleaned_text, preserve_sections=True)
            
            if not chunks:
                error = "No chunks created"
                self.doc_loader.update_document_status(
                    doc.id, DocumentStatus.ERROR, error
                )
                result["error"] = error
                return result
            
            # Guardar chunks en DB
            saved_count, error = self.chunker.save_chunks_to_db(
                self.db, doc.id, chunks
            )
            
            if error:
                result["error"] = f"Chunking failed: {error}"
                return result
            
            result["stages"]["chunking"] = {
                "status": "success",
                "chunks_created": saved_count
            }
            
            # Stage 5: Generar embeddings (opcional)
            if generate_embeddings:
                logger.info(f"Stage 5/5: Generating embeddings")
                emb_count, error = self.embedding_gen.process_chunks(
                    self.db, doc.id, show_progress=True
                )
                
                if error:
                    result["error"] = f"Embedding generation failed: {error}"
                    return result
                
                # Actualizar estado del documento
                self.doc_loader.update_document_status(doc.id, DocumentStatus.INDEXED)
                
                result["stages"]["embeddings"] = {
                    "status": "success",
                    "embeddings_created": emb_count
                }
            else:
                result["stages"]["embeddings"] = {"status": "skipped"}
            
            result["success"] = True
            logger.success(f"Document {doc.id} processed successfully")
            
            return result
        
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            result["error"] = error_msg
            
            if result["document_id"]:
                self.doc_loader.update_document_status(
                    result["document_id"], DocumentStatus.ERROR, error_msg
                )
            
            return result
    
    def process_directory(
        self,
        directory_path: Path,
        uploaded_by: Optional[int] = None,
        generate_embeddings: bool = True,
        recursive: bool = True
    ) -> Dict:
        """
        Procesa todos los PDFs de un directorio
        
        Returns:
            Dict con resumen del procesamiento
        """
        summary = {
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "results": []
        }
        
        if not directory_path.exists():
            logger.error(f"Directory not found: {directory_path}")
            return summary
        
        # Buscar PDFs
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = list(directory_path.glob(pattern))
        summary["total_files"] = len(pdf_files)
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for i, pdf_file in enumerate(pdf_files, 1):
            logger.info(f"Processing file {i}/{len(pdf_files)}: {pdf_file.name}")
            
            result = self.process_single_document(
                pdf_file,
                uploaded_by=uploaded_by,
                generate_embeddings=generate_embeddings
            )
            
            summary["results"].append({
                "filename": pdf_file.name,
                "document_id": result.get("document_id"),
                "success": result["success"],
                "error": result.get("error")
            })
            
            if result["success"]:
                summary["successful"] += 1
            else:
                summary["failed"] += 1
        
        logger.info(
            f"Batch processing complete: "
            f"{summary['successful']} successful, "
            f"{summary['failed']} failed"
        )
        
        return summary
    
    def reprocess_failed_documents(
        self,
        generate_embeddings: bool = True
    ) -> Dict:
        """
        Reintenta procesar documentos que fallaron
        """
        # Obtener documentos con estado ERROR
        failed_docs = self.db.query(MedicalDocument).filter(
            MedicalDocument.status == DocumentStatus.ERROR
        ).all()
        
        if not failed_docs:
            logger.info("No failed documents to reprocess")
            return {"reprocessed": 0, "successful": 0, "failed": 0}
        
        logger.info(f"Reprocessing {len(failed_docs)} failed documents")
        
        summary = {
            "reprocessed": len(failed_docs),
            "successful": 0,
            "failed": 0,
            "results": []
        }
        
        for doc in failed_docs:
            pdf_path = Path(doc.file_path)
            
            if not pdf_path.exists():
                logger.warning(f"File not found: {pdf_path}")
                summary["failed"] += 1
                continue
            
            # Limpiar chunks anteriores
            self.db.query(DocumentChunk).filter(
                DocumentChunk.document_id == doc.id
            ).delete()
            self.db.commit()
            
            # Reintentar procesamiento desde extracción
            result = self._reprocess_existing_document(
                doc, pdf_path, generate_embeddings
            )
            
            summary["results"].append(result)
            
            if result["success"]:
                summary["successful"] += 1
            else:
                summary["failed"] += 1
        
        return summary
    
    def _reprocess_existing_document(
        self,
        doc: MedicalDocument,
        pdf_path: Path,
        generate_embeddings: bool
    ) -> Dict:
        """Helper para reprocesar documento existente"""
        result = {
            "document_id": doc.id,
            "filename": pdf_path.name,
            "success": False,
            "error": None
        }
        
        try:
            # Extraer texto
            text, metadata, error = self.pdf_processor.extract_text_from_pdf(pdf_path)
            if error:
                result["error"] = error
                return result
            
            # Limpiar
            cleaned_text = self.text_cleaner.clean_text(text)
            
            # Chunking
            chunks = self.chunker.chunk_text(cleaned_text)
            saved_count, error = self.chunker.save_chunks_to_db(
                self.db, doc.id, chunks
            )
            
            if error:
                result["error"] = error
                return result
            
            # Embeddings
            if generate_embeddings:
                emb_count, error = self.embedding_gen.process_chunks(
                    self.db, doc.id
                )
                
                if error:
                    result["error"] = error
                    return result
                
                self.doc_loader.update_document_status(doc.id, DocumentStatus.INDEXED)
            
            result["success"] = True
            return result
        
        except Exception as e:
            result["error"] = str(e)
            return result


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def process_pdf_file(
    db: Session,
    pdf_path: str,
    uploaded_by: Optional[int] = None
) -> Dict:
    """Helper function para procesar un archivo PDF"""
    processor = BatchProcessor(db)
    return processor.process_single_document(
        Path(pdf_path),
        uploaded_by=uploaded_by
    )


def process_pdf_directory(
    db: Session,
    directory_path: str,
    uploaded_by: Optional[int] = None
) -> Dict:
    """Helper function para procesar directorio de PDFs"""
    processor = BatchProcessor(db)
    return processor.process_directory(
        Path(directory_path),
        uploaded_by=uploaded_by
    )


# =============================================================================
# DEMO/TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing BatchProcessor...")
    print("=" * 60)
    
    from endocenter.db.database import get_db_context
    
    with get_db_context() as db:
        processor = BatchProcessor(db)
        
        # Test con directorio data/raw
        raw_dir = settings.raw_dir
        
        print(f"\nProcessing directory: {raw_dir}")
        
        if raw_dir.exists():
            # Buscar PDFs
            pdf_files = list(raw_dir.glob("*.pdf"))
            
            if pdf_files:
                print(f"Found {len(pdf_files)} PDF files")
                
                # Procesar primer archivo como test
                test_file = pdf_files[0]
                print(f"\nTest processing: {test_file.name}")
                
                result = processor.process_single_document(
                    test_file,
                    generate_embeddings=True
                )
                
                print("\nResult:")
                print(f"  Success: {result['success']}")
                print(f"  Document ID: {result.get('document_id')}")
                
                if result['success']:
                    print("\nStages:")
                    for stage, info in result['stages'].items():
                        print(f"  {stage}: {info}")
                else:
                    print(f"  Error: {result.get('error')}")
            else:
                print("No PDF files found")
                print(f"Add PDFs to {raw_dir} to test")
        else:
            print(f"Directory not found: {raw_dir}")
            print(f"Create with: mkdir -p {raw_dir}")
    
    print("\n" + "=" * 60)
    print("BatchProcessor ready!")