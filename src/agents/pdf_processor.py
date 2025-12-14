"""PDF processing agent - converts PDF to images and extracts text."""

import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Tuple
import io


class PDFProcessor:
    """Processes PDF files into images and text for downstream analysis."""
    
    def __init__(self, dpi: int = 200):
        """
        Initialize PDF processor.
        
        Args:
            dpi: Resolution for rendering pages as images
        """
        self.dpi = dpi
        self.zoom = dpi / 72  # 72 is the base PDF resolution
    
    def process(self, pdf_path: str) -> Tuple[List[bytes], str, int]:
        """
        Process a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (page_images, full_text, page_count)
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        doc = fitz.open(str(pdf_path))
        
        page_images = []
        full_text = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract text
            text = page.get_text()
            full_text.append(f"--- Page {page_num + 1} ---\n{text}")
            
            # Render page as image
            mat = fitz.Matrix(self.zoom, self.zoom)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            page_images.append(img_bytes)
        
        doc.close()
        
        return page_images, "\n\n".join(full_text), len(page_images)
    
    def get_page_image(self, pdf_path: str, page_num: int) -> bytes:
        """Get a single page as an image."""
        doc = fitz.open(str(pdf_path))
        
        if page_num < 0 or page_num >= len(doc):
            raise ValueError(f"Invalid page number: {page_num}")
        
        page = doc[page_num]
        mat = fitz.Matrix(self.zoom, self.zoom)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        
        doc.close()
        return img_bytes
    
    def get_page_dimensions(self, pdf_path: str, page_num: int) -> Tuple[int, int]:
        """Get dimensions of a rendered page in pixels."""
        doc = fitz.open(str(pdf_path))
        page = doc[page_num]
        mat = fitz.Matrix(self.zoom, self.zoom)
        pix = page.get_pixmap(matrix=mat)
        width, height = pix.width, pix.height
        doc.close()
        return width, height
