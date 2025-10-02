"""
EndoCenter MLOps - PDF Processor
Extrae texto de documentos PDF médicos usando PyMuPDF
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from loguru import logger

from endocenter.db.models import MedicalDocument, DocumentStatus
from endocenter.config import settings


class PDFProcessor:
    """
    Procesa archivos PDF y extrae texto limpio
    Maneja tablas, imágenes con texto, y estructura del documento
    """
    
    def __init__(self):
        self.min_text_length = 50  # Mínimo de caracteres para considerar página válida
    
    def extract_text_from_pdf(
        self,
        pdf_path: Path,
        extract_images: bool = False
    ) -> Tuple[Optional[str], Optional[Dict], Optional[str]]:
        """
        Extrae texto completo de un PDF
        
        Returns:
            (full_text, metadata, error_message)
        """
        try:
            # Abrir PDF
            doc = fitz.open(pdf_path)
            
            # Metadata del PDF
            metadata = {
                "page_count": len(doc),
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "keywords": doc.metadata.get("keywords", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "creation_date": doc.metadata.get("creationDate", ""),
                "modification_date": doc.metadata.get("modDate", ""),
            }
            
            # Extraer texto página por página
            full_text = []
            page_texts = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extraer texto de la página
                text = page.get_text("text")
                
                # Validar que tenga contenido suficiente
                if len(text.strip()) >= self.min_text_length:
                    page_texts.append({
                        "page_number": page_num + 1,
                        "text": text,
                        "char_count": len(text)
                    })
                    full_text.append(text)
            
            doc.close()
            
            # Combinar todo el texto
            combined_text = "\n\n".join(full_text)
            
            # Calcular estadísticas
            word_count = len(combined_text.split())
            metadata["word_count"] = word_count
            metadata["pages_with_text"] = len(page_texts)
            metadata["average_words_per_page"] = word_count // len(page_texts) if page_texts else 0
            
            logger.success(
                f"Extracted {word_count} words from {len(doc)} pages in {pdf_path.name}"
            )
            
            return combined_text, metadata, None
        
        except Exception as e:
            error_msg = f"Error extracting text from PDF: {str(e)}"
            logger.error(error_msg)
            return None, None, error_msg
    
    def extract_text_by_page(
        self,
        pdf_path: Path
    ) -> Tuple[Optional[List[Dict]], Optional[Dict], Optional[str]]:
        """
        Extrae texto página por página (útil para mantener contexto)
        
        Returns:
            (pages_list, metadata, error_message)
            pages_list: [{"page_number": 1, "text": "...", "char_count": 123}, ...]
        """
        try:
            doc = fitz.open(pdf_path)
            
            metadata = {
                "page_count": len(doc),
                "title": doc.metadata.get("title", ""),
            }
            
            pages = []
            total_words = 0
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                
                if len(text.strip()) >= self.min_text_length:
                    word_count = len(text.split())
                    total_words += word_count
                    
                    pages.append({
                        "page_number": page_num + 1,
                        "text": text,
                        "char_count": len(text),
                        "word_count": word_count
                    })
            
            doc.close()
            
            metadata["word_count"] = total_words
            metadata["pages_with_text"] = len(pages)
            
            logger.info(f"Extracted {len(pages)} pages from {pdf_path.name}")
            
            return pages, metadata, None
        
        except Exception as e:
            error_msg = f"Error extracting pages: {str(e)}"
            logger.error(error_msg)
            return None, None, error_msg
    
    def extract_sections(
        self,
        pdf_path: Path
    ) -> Tuple[Optional[List[Dict]], Optional[str]]:
        """
        Intenta identificar secciones del documento (headers, capítulos)
        Útil para documentos estructurados como libros o guidelines
        
        Returns:
            (sections_list, error_message)
        """
        try:
            doc = fitz.open(pdf_path)
            sections = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Obtener texto con información de formato
                blocks = page.get_text("dict")["blocks"]
                
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                # Detectar posibles headers por tamaño de fuente
                                if span["size"] > 12:  # Font size mayor = posible header
                                    sections.append({
                                        "page": page_num + 1,
                                        "text": span["text"],
                                        "font_size": span["size"],
                                        "is_bold": "bold" in span["font"].lower()
                                    })
            
            doc.close()
            
            logger.info(f"Found {len(sections)} potential sections in {pdf_path.name}")
            
            return sections, None
        
        except Exception as e:
            error_msg = f"Error extracting sections: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
    
    def get_text_preview(self, pdf_path: Path, max_chars: int = 500) -> Optional[str]:
        """
        Obtiene un preview del contenido del PDF
        """
        try:
            doc = fitz.open(pdf_path)
            
            # Obtener texto de la primera página con contenido
            for page_num in range(min(3, len(doc))):  # Revisar primeras 3 páginas
                page = doc[page_num]
                text = page.get_text("text")
                
                if len(text.strip()) >= self.min_text_length:
                    doc.close()
                    return text[:max_chars].strip() + "..."
            
            doc.close()
            return None
        
        except Exception as e:
            logger.error(f"Error getting preview: {e}")
            return None


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def process_pdf(pdf_path: str) -> Optional[Dict]:
    """
    Helper function para procesar un PDF
    Returns metadata y texto extraído
    """
    processor = PDFProcessor()
    text, metadata, error = processor.extract_text_from_pdf(Path(pdf_path))
    
    if error:
        logger.error(f"Failed to process PDF: {error}")
        return None
    
    return {
        "text": text,
        "metadata": metadata
    }


def process_pdf_by_pages(pdf_path: str) -> Optional[List[Dict]]:
    """
    Helper function para extraer texto por páginas
    """
    processor = PDFProcessor()
    pages, metadata, error = processor.extract_text_by_page(Path(pdf_path))
    
    if error:
        logger.error(f"Failed to process PDF: {error}")
        return None
    
    return pages


# =============================================================================
# DEMO/TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing PDFProcessor...")
    print("=" * 60)
    
    processor = PDFProcessor()
    
    # Buscar un PDF de prueba
    raw_dir = settings.raw_dir
    
    if raw_dir.exists():
        pdf_files = list(raw_dir.glob("*.pdf"))
        
        if pdf_files:
            test_pdf = pdf_files[0]
            print(f"\nTesting with: {test_pdf.name}")
            
            # Test extracción completa
            print("\n1. Full text extraction...")
            text, metadata, error = processor.extract_text_from_pdf(test_pdf)
            
            if text:
                print(f"   Pages: {metadata['page_count']}")
                print(f"   Words: {metadata['word_count']}")
                print(f"   Preview: {text[:200]}...")
            else:
                print(f"   Error: {error}")
            
            # Test preview
            print("\n2. Text preview...")
            preview = processor.get_text_preview(test_pdf)
            if preview:
                print(f"   {preview}")
        else:
            print(f"\nNo PDF files found in {raw_dir}")
            print("Add some PDF files to test")
    else:
        print(f"\nDirectory not found: {raw_dir}")
        print("Create it with: mkdir -p data/raw")
    
    print("\n" + "=" * 60)
    print("PDFProcessor ready!")