"""
EndoCenter MLOps - Text Cleaner
Limpia y normaliza texto médico extraído de PDFs
"""

import re
from typing import List, Optional
from loguru import logger


class TextCleaner:
    """
    Limpia texto médico removiendo ruido y normalizando formato
    Preserva terminología médica importante
    """
    
    def __init__(self):
        # Patrones de texto a remover
        self.noise_patterns = [
            r'\f',  # Form feed
            r'\x0c',  # Page break
            r'\u200b',  # Zero-width space
            r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]',  # Control characters
        ]
        
        # Headers/footers comunes en PDFs médicos
        self.header_footer_patterns = [
            r'Página\s+\d+\s+de\s+\d+',
            r'Page\s+\d+\s+of\s+\d+',
            r'Copyright\s+©\s+\d{4}',
            r'©\s+\d{4}',
        ]
    
    def remove_noise(self, text: str) -> str:
        """
        Remueve caracteres de control y ruido
        """
        cleaned = text
        
        for pattern in self.noise_patterns:
            cleaned = re.sub(pattern, '', cleaned)
        
        return cleaned
    
    def remove_headers_footers(self, text: str) -> str:
        """
        Remueve headers y footers comunes
        """
        cleaned = text
        
        for pattern in self.header_footer_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        return cleaned
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Normaliza espacios en blanco y saltos de línea
        """
        # Remover espacios múltiples
        text = re.sub(r' +', ' ', text)
        
        # Normalizar saltos de línea múltiples a máximo 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remover espacios al inicio/fin de líneas
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        return text.strip()
    
    def fix_hyphenation(self, text: str) -> str:
        """
        Une palabras separadas por guiones al final de línea
        Común en PDFs donde palabras se dividen entre líneas
        """
        # Patrón: palabra- \n palabra
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        return text
    
    def remove_references(self, text: str, aggressive: bool = False) -> str:
        """
        Remueve secciones de referencias bibliográficas
        
        Args:
            aggressive: Si True, remueve más agresivamente
        """
        # Patrones de inicio de sección de referencias
        reference_markers = [
            r'\n\s*Referencias?\s*\n',
            r'\n\s*References?\s*\n',
            r'\n\s*Bibliografía\s*\n',
            r'\n\s*Bibliography\s*\n',
        ]
        
        for marker in reference_markers:
            match = re.search(marker, text, flags=re.IGNORECASE)
            if match:
                # Cortar el texto antes de las referencias
                text = text[:match.start()]
                logger.debug(f"Removed references section starting at position {match.start()}")
                break
        
        return text
    
    def preserve_medical_terms(self, text: str) -> str:
        """
        Asegura que términos médicos importantes se preserven correctamente
        """
        # Preservar abreviaciones médicas comunes
        medical_abbrev = {
            'TSH': 'TSH',
            'T3': 'T3',
            'T4': 'T4',
            'HbA1c': 'HbA1c',
            'IMC': 'IMC',
            'BMI': 'BMI',
            'SOP': 'SOP',
            'PCOS': 'PCOS',
        }
        
        # Por ahora solo retornamos el texto
        # En versión avanzada, podríamos normalizar variaciones
        return text
    
    def remove_page_numbers(self, text: str) -> str:
        """
        Remueve números de página sueltos
        """
        # Líneas que son solo números
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        return text
    
    def clean_text(
        self,
        text: str,
        remove_refs: bool = True,
        fix_hyphens: bool = True
    ) -> str:
        """
        Pipeline completo de limpieza de texto
        
        Args:
            text: Texto a limpiar
            remove_refs: Si remover sección de referencias
            fix_hyphens: Si unir palabras con guión
        
        Returns:
            Texto limpio
        """
        if not text:
            return ""
        
        logger.debug(f"Cleaning text: {len(text)} characters")
        
        # 1. Remover ruido
        text = self.remove_noise(text)
        
        # 2. Remover headers/footers
        text = self.remove_headers_footers(text)
        
        # 3. Remover números de página
        text = self.remove_page_numbers(text)
        
        # 4. Fix hyphenation si está habilitado
        if fix_hyphens:
            text = self.fix_hyphenation(text)
        
        # 5. Remover referencias si está habilitado
        if remove_refs:
            text = self.remove_references(text)
        
        # 6. Preservar términos médicos
        text = self.preserve_medical_terms(text)
        
        # 7. Normalizar whitespace (siempre al final)
        text = self.normalize_whitespace(text)
        
        logger.debug(f"Cleaned text: {len(text)} characters")
        
        return text
    
    def clean_pages(self, pages: List[dict]) -> List[dict]:
        """
        Limpia una lista de páginas
        
        Args:
            pages: Lista de dicts con {"page_number": X, "text": "..."}
        
        Returns:
            Lista de páginas con texto limpio
        """
        cleaned_pages = []
        
        for page in pages:
            cleaned_text = self.clean_text(
                page["text"],
                remove_refs=False,  # No remover refs por página
                fix_hyphens=True
            )
            
            if cleaned_text:  # Solo agregar si tiene contenido
                cleaned_pages.append({
                    **page,
                    "text": cleaned_text,
                    "original_length": len(page["text"]),
                    "cleaned_length": len(cleaned_text)
                })
        
        logger.info(f"Cleaned {len(cleaned_pages)} pages")
        
        return cleaned_pages


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clean_medical_text(text: str) -> str:
    """
    Helper function para limpiar texto médico
    """
    cleaner = TextCleaner()
    return cleaner.clean_text(text)


def clean_document_pages(pages: List[dict]) -> List[dict]:
    """
    Helper function para limpiar páginas de documento
    """
    cleaner = TextCleaner()
    return cleaner.clean_pages(pages)


# =============================================================================
# DEMO/TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing TextCleaner...")
    print("=" * 60)
    
    cleaner = TextCleaner()
    
    # Texto de prueba con ruido común en PDFs
    test_text = """
    Página 1 de 10
    
    Capítulo 5: Diabetes Mellitus Tipo 2
    
    La diabetes mellitus tipo 2 es una enfer-
    medad metabólica caracterizada por hiper-
    glucemia crónica.
    
    
    
    Los niveles de HbA1c deben mantenerse
    por debajo del 7%.
    
    Copyright © 2023
    """
    
    print("\nOriginal text:")
    print(test_text)
    print(f"\nLength: {len(test_text)} characters")
    
    # Limpiar
    cleaned = cleaner.clean_text(test_text)
    
    print("\nCleaned text:")
    print(cleaned)
    print(f"\nLength: {len(cleaned)} characters")
    
    # Test con lista de páginas
    print("\n" + "=" * 60)
    print("Testing page cleaning...")
    
    pages = [
        {"page_number": 1, "text": test_text},
        {"page_number": 2, "text": "Página 2\n\nMás contenido médico."}
    ]
    
    cleaned_pages = cleaner.clean_pages(pages)
    
    print(f"\nCleaned {len(cleaned_pages)} pages")
    for page in cleaned_pages:
        print(f"  Page {page['page_number']}: {page['cleaned_length']} chars")
    
    print("\n" + "=" * 60)
    print("TextCleaner ready!")