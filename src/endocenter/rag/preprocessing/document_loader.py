"""
EndoCenter MLOps - Document Loader
Carga y valida documentos médicos (PDFs) para el sistema RAG
"""

import hashlib
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from sqlalchemy.orm import Session
from loguru import logger

from endocenter.db.models import (
    MedicalDocument, 
    DocumentType, 
    DocumentStatus,
    MedicalSpecialty
)
from endocenter.config import settings


class DocumentLoader:
    """
    Carga documentos médicos y guarda metadata en la base de datos
    Valida archivos, calcula hashes, y previene duplicados
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.supported_extensions = ['.pdf']
        self.max_file_size_mb = 100  # Máximo 100MB por archivo
    
    def validate_file(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Valida que el archivo sea adecuado para procesamiento
        Returns: (is_valid, error_message)
        """
        # Verificar que existe
        if not file_path.exists():
            return False, f"File not found: {file_path}"
        
        # Verificar que es archivo (no directorio)
        if not file_path.is_file():
            return False, f"Path is not a file: {file_path}"
        
        # Verificar extensión
        if file_path.suffix.lower() not in self.supported_extensions:
            return False, f"Unsupported file type: {file_path.suffix}. Supported: {self.supported_extensions}"
        
        # Verificar tamaño
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            return False, f"File too large: {file_size_mb:.2f}MB (max: {self.max_file_size_mb}MB)"
        
        # Verificar permisos de lectura
        if not file_path.is_file() or not file_path.stat().st_mode & 0o400:
            return False, f"File not readable: {file_path}"
        
        return True, None
    
    def calculate_file_hash(self, file_path: Path) -> str:
        """
        Calcula SHA-256 hash del archivo para detectar duplicados
        """
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            # Leer en chunks para archivos grandes
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    def check_duplicate(self, file_hash: str) -> Optional[MedicalDocument]:
        """
        Verifica si ya existe un documento con el mismo hash
        Returns el documento existente o None
        """
        return self.db.query(MedicalDocument).filter(
            MedicalDocument.file_hash == file_hash
        ).first()
    
    def extract_metadata_from_filename(self, filename: str) -> Dict:
        """
        Intenta extraer metadata del nombre del archivo
        Ejemplo: "diabetes_tipo_2_guia_clinica.pdf" -> {"disease": "diabetes", "type": "guideline"}
        """
        metadata = {
            "document_type": DocumentType.OTHER,
            "medical_specialty": None,
            "diseases_covered": []
        }
        
        filename_lower = filename.lower()
        
        # Detectar tipo de documento
        if any(word in filename_lower for word in ['guia', 'guideline', 'protocolo']):
            metadata["document_type"] = DocumentType.CLINICAL_GUIDELINE
        elif any(word in filename_lower for word in ['textbook', 'libro', 'greenspan']):
            metadata["document_type"] = DocumentType.TEXTBOOK
        elif any(word in filename_lower for word in ['research', 'estudio', 'paper']):
            metadata["document_type"] = DocumentType.RESEARCH_PAPER
        elif any(word in filename_lower for word in ['caso', 'case']):
            metadata["document_type"] = DocumentType.CASE_STUDY
        
        # Detectar especialidad
        specialty_keywords = {
            MedicalSpecialty.DIABETES: ['diabetes', 'glucosa', 'insulina'],
            MedicalSpecialty.THYROID: ['tiroides', 'thyroid', 'hipotiroidismo', 'hipertiroidismo'],
            MedicalSpecialty.ADRENAL: ['suprarrenal', 'adrenal', 'cushing', 'addison'],
            MedicalSpecialty.BONE_METABOLISM: ['osteoporosis', 'bone', 'calcio', 'hueso'],
            MedicalSpecialty.REPRODUCTIVE_ENDOCRINOLOGY: ['sop', 'ovario', 'poliquistico', 'pcos'],
        }
        
        for specialty, keywords in specialty_keywords.items():
            if any(keyword in filename_lower for keyword in keywords):
                metadata["medical_specialty"] = specialty
                break
        
        # Detectar enfermedades mencionadas
        disease_keywords = {
            'diabetes': ['diabetes', 'diabetico'],
            'hipotiroidismo': ['hipotiroidismo', 'hypothyroid'],
            'cushing': ['cushing'],
            'osteoporosis': ['osteoporosis'],
            'sop': ['sop', 'pcos', 'ovario_poliquistico']
        }
        
        for disease, keywords in disease_keywords.items():
            if any(keyword in filename_lower for keyword in keywords):
                metadata["diseases_covered"].append(disease)
        
        return metadata
    
    def load_document(
        self,
        file_path: Path,
        title: Optional[str] = None,
        document_type: Optional[DocumentType] = None,
        medical_specialty: Optional[MedicalSpecialty] = None,
        diseases_covered: Optional[List[str]] = None,
        uploaded_by: Optional[int] = None,
        auto_metadata: bool = True
    ) -> Tuple[Optional[MedicalDocument], Optional[str]]:
        """
        Carga un documento médico y guarda metadata en la base de datos
        
        Args:
            file_path: Ruta al archivo PDF
            title: Título del documento (si None, usa filename)
            document_type: Tipo de documento (si None y auto_metadata=True, detecta automáticamente)
            medical_specialty: Especialidad médica
            diseases_covered: Lista de enfermedades cubiertas
            uploaded_by: ID del usuario que sube el documento
            auto_metadata: Si True, intenta extraer metadata del filename
        
        Returns:
            (MedicalDocument, error_message)
        """
        try:
            # 1. Validar archivo
            is_valid, error = self.validate_file(file_path)
            if not is_valid:
                logger.error(f"Validation failed: {error}")
                return None, error
            
            # 2. Calcular hash
            file_hash = self.calculate_file_hash(file_path)
            logger.info(f"File hash: {file_hash}")
            
            # 3. Verificar duplicado
            existing_doc = self.check_duplicate(file_hash)
            if existing_doc:
                logger.warning(f"Duplicate document found: {existing_doc.id}")
                return None, f"Document already exists with ID: {existing_doc.id}"
            
            # 4. Extraer metadata automática si está habilitado
            auto_meta = {}
            if auto_metadata:
                auto_meta = self.extract_metadata_from_filename(file_path.name)
                logger.info(f"Auto-detected metadata: {auto_meta}")
            
            # 5. Usar metadata proporcionada o auto-detectada
            final_title = title or file_path.stem.replace('_', ' ').title()
            final_doc_type = document_type or auto_meta.get("document_type", DocumentType.OTHER)
            final_specialty = medical_specialty or auto_meta.get("medical_specialty")
            final_diseases = diseases_covered or auto_meta.get("diseases_covered", [])
            
            # 6. Obtener información del archivo
            file_size = file_path.stat().st_size
            
            # 7. Crear registro en base de datos
            doc = MedicalDocument(
                title=final_title,
                filename=file_path.name,
                file_path=str(file_path.absolute()),
                file_size_bytes=file_size,
                file_hash=file_hash,
                document_type=final_doc_type,
                medical_specialty=final_specialty,
                diseases_covered=final_diseases,
                status=DocumentStatus.UPLOADED,
                uploaded_by=uploaded_by,
                language="es",  # Default español
                version="1.0"
            )
            
            self.db.add(doc)
            self.db.commit()
            self.db.refresh(doc)
            
            logger.success(f"Document loaded successfully: {doc.id} - {doc.title}")
            return doc, None
        
        except Exception as e:
            self.db.rollback()
            error_msg = f"Error loading document: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
    
    def load_directory(
        self,
        directory_path: Path,
        recursive: bool = False,
        uploaded_by: Optional[int] = None
    ) -> Dict[str, List]:
        """
        Carga todos los PDFs de un directorio
        
        Returns:
            {
                "loaded": [MedicalDocument, ...],
                "failed": [(file_path, error_message), ...],
                "duplicates": [file_path, ...]
            }
        """
        results = {
            "loaded": [],
            "failed": [],
            "duplicates": []
        }
        
        if not directory_path.exists():
            logger.error(f"Directory not found: {directory_path}")
            return results
        
        # Obtener archivos PDF
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = list(directory_path.glob(pattern))
        
        logger.info(f"Found {len(pdf_files)} PDF files in {directory_path}")
        
        for pdf_file in pdf_files:
            doc, error = self.load_document(
                pdf_file,
                uploaded_by=uploaded_by,
                auto_metadata=True
            )
            
            if doc:
                results["loaded"].append(doc)
            elif error and "already exists" in error:
                results["duplicates"].append(pdf_file)
            else:
                results["failed"].append((pdf_file, error))
        
        logger.info(
            f"Directory loading complete: "
            f"{len(results['loaded'])} loaded, "
            f"{len(results['duplicates'])} duplicates, "
            f"{len(results['failed'])} failed"
        )
        
        return results
    
    def get_document_by_hash(self, file_hash: str) -> Optional[MedicalDocument]:
        """Obtener documento por hash"""
        return self.check_duplicate(file_hash)
    
    def get_document_by_id(self, doc_id: int) -> Optional[MedicalDocument]:
        """Obtener documento por ID"""
        return self.db.query(MedicalDocument).filter(
            MedicalDocument.id == doc_id
        ).first()
    
    def update_document_status(
        self,
        doc_id: int,
        status: DocumentStatus,
        error_message: Optional[str] = None
    ) -> bool:
        """Actualizar estado del documento"""
        try:
            doc = self.get_document_by_id(doc_id)
            if not doc:
                logger.error(f"Document {doc_id} not found")
                return False
            
            doc.status = status
            if error_message:
                doc.processing_error = error_message
            
            if status == DocumentStatus.INDEXED:
                doc.processed_at = datetime.utcnow()
            
            self.db.commit()
            logger.info(f"Document {doc_id} status updated to {status}")
            return True
        
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating document status: {e}")
            return False


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_single_document(
    db: Session,
    file_path: str,
    title: Optional[str] = None,
    uploaded_by: Optional[int] = None
) -> Optional[MedicalDocument]:
    """
    Helper function para cargar un solo documento
    """
    loader = DocumentLoader(db)
    doc, error = loader.load_document(
        Path(file_path),
        title=title,
        uploaded_by=uploaded_by
    )
    
    if error:
        logger.error(f"Failed to load document: {error}")
        return None
    
    return doc


def load_documents_from_directory(
    db: Session,
    directory_path: str,
    recursive: bool = True,
    uploaded_by: Optional[int] = None
) -> Dict:
    """
    Helper function para cargar directorio completo
    """
    loader = DocumentLoader(db)
    return loader.load_directory(
        Path(directory_path),
        recursive=recursive,
        uploaded_by=uploaded_by
    )


# =============================================================================
# DEMO/TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing DocumentLoader...")
    print("=" * 60)
    
    from endocenter.db.database import get_db_context
    
    # Test con archivo de ejemplo
    print("\n1. Testing document validation...")
    
    with get_db_context() as db:
        loader = DocumentLoader(db)
        
        # Test directorio data/raw
        raw_dir = settings.raw_dir
        print(f"\n2. Checking directory: {raw_dir}")
        
        if raw_dir.exists():
            results = loader.load_directory(raw_dir, recursive=True)
            
            print(f"\nResults:")
            print(f"  Loaded: {len(results['loaded'])}")
            print(f"  Duplicates: {len(results['duplicates'])}")
            print(f"  Failed: {len(results['failed'])}")
            
            if results['loaded']:
                doc = results['loaded'][0]
                print(f"\nFirst document:")
                print(f"  ID: {doc.id}")
                print(f"  Title: {doc.title}")
                print(f"  Type: {doc.document_type}")
                print(f"  Specialty: {doc.medical_specialty}")
                print(f"  Diseases: {doc.diseases_covered}")
        else:
            print(f"  Directory not found. Create it with:")
            print(f"  mkdir -p {raw_dir}")
            print(f"  Then add PDF files to test")
    
    print("\n" + "=" * 60)
    print("DocumentLoader ready!")