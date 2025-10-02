"""
EndoCenter MLOps - Text Chunker
Divide texto médico en chunks semánticamente coherentes para RAG
"""

import re
from typing import List, Dict, Optional, Tuple
from loguru import logger
from sqlalchemy.orm import Session

from endocenter.db.models import MedicalDocument, DocumentChunk, DocumentStatus
from endocenter.config import settings


class TextChunker:
    """
    Divide texto en chunks optimizados para embeddings y retrieval
    Respeta párrafos, secciones y contexto médico
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        min_chunk_size: int = 50
    ):
        """
        Args:
            chunk_size: Tamaño máximo de chunk en caracteres
            chunk_overlap: Solapamiento entre chunks para mantener contexto
            min_chunk_size: Tamaño mínimo para considerar chunk válido
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # Marcadores de secciones en textos médicos
        self.section_markers = [
            r'^#+\s+',  # Markdown headers
            r'^Capítulo\s+\d+',
            r'^Chapter\s+\d+',
            r'^\d+\.\s+[A-Z]',  # 1. Título
            r'^[A-ZÁÉÍÓÚÑ\s]{3,30}$',  # TÍTULOS EN MAYÚSCULAS
        ]
    
    def split_into_paragraphs(self, text: str) -> List[str]:
        """
        Divide texto en párrafos respetando estructura
        """
        # Dividir por doble salto de línea o más
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Filtrar párrafos vacíos y muy cortos
        paragraphs = [
            p.strip() for p in paragraphs
            if len(p.strip()) >= self.min_chunk_size
        ]
        
        return paragraphs
    
    def is_section_header(self, text: str) -> bool:
        """
        Detecta si un texto es un header de sección
        """
        text = text.strip()
        
        for pattern in self.section_markers:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        
        # Headers son típicamente cortos
        if len(text) < 100 and text[0].isupper():
            return True
        
        return False
    
    def extract_section_title(self, text: str) -> Optional[str]:
        """
        Extrae el título de una sección si existe
        """
        lines = text.split('\n')
        
        for line in lines[:3]:  # Revisar primeras 3 líneas
            line = line.strip()
            if self.is_section_header(line):
                return line
        
        return None
    
    def create_chunks_from_paragraphs(
        self,
        paragraphs: List[str],
        section_title: Optional[str] = None
    ) -> List[Dict]:
        """
        Crea chunks a partir de párrafos, respetando tamaño límite
        """
        chunks = []
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            # Si el párrafo solo cabe en nuevo chunk
            if current_size + para_size > self.chunk_size and current_chunk:
                # Guardar chunk actual
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "size": len(chunk_text),
                    "section_title": section_title
                })
                
                # Iniciar nuevo chunk con overlap
                # Tomar últimas palabras del chunk anterior
                overlap_text = self._get_overlap_text(chunk_text)
                current_chunk = [overlap_text, para] if overlap_text else [para]
                current_size = len('\n\n'.join(current_chunk))
            else:
                # Agregar párrafo al chunk actual
                current_chunk.append(para)
                current_size += para_size + 2  # +2 por \n\n
        
        # Agregar último chunk si tiene contenido
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append({
                    "text": chunk_text,
                    "size": len(chunk_text),
                    "section_title": section_title
                })
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """
        Obtiene las últimas palabras para overlap
        """
        if len(text) <= self.chunk_overlap:
            return text
        
        # Tomar últimos N caracteres, cortando en espacio
        overlap = text[-self.chunk_overlap:]
        first_space = overlap.find(' ')
        
        if first_space > 0:
            return overlap[first_space:].strip()
        
        return overlap
    
    def chunk_text(
        self,
        text: str,
        preserve_sections: bool = True
    ) -> List[Dict]:
        """
        Divide texto completo en chunks
        
        Args:
            text: Texto a dividir
            preserve_sections: Si True, intenta mantener secciones juntas
        
        Returns:
            Lista de chunks con metadata
        """
        if not text or len(text) < self.min_chunk_size:
            logger.warning("Text too short to chunk")
            return []
        
        logger.info(f"Chunking text of {len(text)} characters")
        
        # Dividir en párrafos
        paragraphs = self.split_into_paragraphs(text)
        
        if not paragraphs:
            logger.warning("No valid paragraphs found")
            return []
        
        # Si no se preservan secciones, procesar todo junto
        if not preserve_sections:
            chunks = self.create_chunks_from_paragraphs(paragraphs)
        else:
            # Agrupar por secciones si es posible
            chunks = self._chunk_with_sections(paragraphs)
        
        # Agregar índices
        for i, chunk in enumerate(chunks):
            chunk['chunk_index'] = i
            chunk['total_chunks'] = len(chunks)
        
        logger.success(f"Created {len(chunks)} chunks")
        
        return chunks
    
    def _chunk_with_sections(self, paragraphs: List[str]) -> List[Dict]:
        """
        Agrupa párrafos por secciones antes de hacer chunks
        """
        sections = []
        current_section = []
        current_title = None
        
        for para in paragraphs:
            # Verificar si es un header
            if self.is_section_header(para):
                # Guardar sección anterior
                if current_section:
                    sections.append({
                        "title": current_title,
                        "paragraphs": current_section
                    })
                
                # Iniciar nueva sección
                current_title = para
                current_section = []
            else:
                current_section.append(para)
        
        # Agregar última sección
        if current_section:
            sections.append({
                "title": current_title,
                "paragraphs": current_section
            })
        
        # Crear chunks por sección
        all_chunks = []
        for section in sections:
            section_chunks = self.create_chunks_from_paragraphs(
                section["paragraphs"],
                section_title=section["title"]
            )
            all_chunks.extend(section_chunks)
        
        return all_chunks
    
    def chunk_document_pages(
        self,
        pages: List[Dict]
    ) -> List[Dict]:
        """
        Procesa páginas de documento y genera chunks con info de página
        
        Args:
            pages: Lista de {"page_number": X, "text": "..."}
        
        Returns:
            Lista de chunks con metadata de página
        """
        all_chunks = []
        
        for page in pages:
            page_text = page["text"]
            page_number = page["page_number"]
            
            # Obtener chunks de esta página
            page_chunks = self.chunk_text(page_text, preserve_sections=False)
            
            # Agregar info de página
            for chunk in page_chunks:
                chunk["page_number"] = page_number
            
            all_chunks.extend(page_chunks)
        
        # Re-indexar todos los chunks
        for i, chunk in enumerate(all_chunks):
            chunk['chunk_index'] = i
            chunk['total_chunks'] = len(all_chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(pages)} pages")
        
        return all_chunks
    
    def save_chunks_to_db(
        self,
        db: Session,
        document_id: int,
        chunks: List[Dict]
    ) -> Tuple[int, Optional[str]]:
        """
        Guarda chunks en la base de datos
        
        Returns:
            (chunks_saved, error_message)
        """
        try:
            # Verificar que el documento existe
            doc = db.query(MedicalDocument).filter(
                MedicalDocument.id == document_id
            ).first()
            
            if not doc:
                return 0, f"Document {document_id} not found"
            
            # Crear chunks en DB
            saved_count = 0
            
            for chunk_data in chunks:
                chunk = DocumentChunk(
                    document_id=document_id,
                    chunk_index=chunk_data["chunk_index"],
                    content=chunk_data["text"],
                    content_length=len(chunk_data["text"]),
                    page_number=chunk_data.get("page_number"),
                    section_title=chunk_data.get("section_title"),
                    chunk_type="paragraph"
                )
                
                db.add(chunk)
                saved_count += 1
            
            # Actualizar documento
            doc.total_chunks = saved_count
            doc.status = DocumentStatus.CHUNKED
            
            db.commit()
            
            logger.success(f"Saved {saved_count} chunks for document {document_id}")
            
            return saved_count, None
        
        except Exception as e:
            db.rollback()
            error_msg = f"Error saving chunks: {str(e)}"
            logger.error(error_msg)
            return 0, error_msg


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def chunk_text_simple(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Helper function para chunking simple
    Returns lista de strings
    """
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = chunker.chunk_text(text)
    return [chunk["text"] for chunk in chunks]


def process_and_save_chunks(
    db: Session,
    document_id: int,
    text: str
) -> Tuple[int, Optional[str]]:
    """
    Helper function para procesar y guardar chunks de un documento
    """
    chunker = TextChunker(
        chunk_size=settings.max_chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    
    # Crear chunks
    chunks = chunker.chunk_text(text)
    
    if not chunks:
        return 0, "No chunks created"
    
    # Guardar en DB
    return chunker.save_chunks_to_db(db, document_id, chunks)


# =============================================================================
# DEMO/TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing TextChunker...")
    print("=" * 60)
    
    chunker = TextChunker(chunk_size=300, chunk_overlap=50)
    
    # Texto de prueba médico
    test_text = """
Capítulo 5: Diabetes Mellitus Tipo 2

La diabetes mellitus tipo 2 es una enfermedad metabólica crónica caracterizada por hiperglucemia, que resulta de defectos en la secreción de insulina, la acción de la insulina, o ambos.

Epidemiología

La prevalencia global de diabetes tipo 2 ha aumentado dramáticamente en las últimas décadas. Se estima que más de 400 millones de personas en el mundo padecen esta condición.

Los factores de riesgo incluyen obesidad, sedentarismo, historia familiar, y edad avanzada.

Fisiopatología

La resistencia a la insulina es el defecto fisiopatológico principal en la diabetes tipo 2. Los tejidos periféricos, especialmente el músculo esquelético y el tejido adiposo, presentan una respuesta disminuida a la insulina.

Con el tiempo, las células beta pancreáticas no pueden mantener la hipersecreción compensatoria de insulina, lo que lleva a hiperglucemia progresiva.

Diagnóstico

Los criterios diagnósticos incluyen:
- Glucosa en ayunas ≥126 mg/dL
- HbA1c ≥6.5%
- Glucosa 2 horas post-carga ≥200 mg/dL

Tratamiento

El manejo de la diabetes tipo 2 es multifactorial e incluye modificaciones del estilo de vida, terapia farmacológica, y monitoreo continuo de la glucemia.
"""
    
    print("\n1. Testing basic chunking...")
    chunks = chunker.chunk_text(test_text, preserve_sections=True)
    
    print(f"\nCreated {len(chunks)} chunks:")
    for chunk in chunks:
        print(f"\n  Chunk {chunk['chunk_index']}:")
        print(f"    Size: {chunk['size']} chars")
        print(f"    Section: {chunk.get('section_title', 'None')}")
        print(f"    Preview: {chunk['text'][:100]}...")
    
    # Test con páginas
    print("\n2. Testing page-based chunking...")
    pages = [
        {"page_number": 1, "text": test_text[:500]},
        {"page_number": 2, "text": test_text[500:]}
    ]
    
    page_chunks = chunker.chunk_document_pages(pages)
    print(f"\nCreated {len(page_chunks)} chunks from {len(pages)} pages")
    
    for chunk in page_chunks:
        print(f"  Chunk {chunk['chunk_index']} (Page {chunk['page_number']}): {chunk['size']} chars")
    
    # Test guardado en DB
    print("\n3. Testing database save...")
    print("  (Skipped - requires database connection)")
    print("  Use process_and_save_chunks() with active DB session")
    
    print("\n" + "=" * 60)
    print("TextChunker ready!")