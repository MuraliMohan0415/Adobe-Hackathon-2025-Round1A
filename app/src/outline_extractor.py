# outline_extractor.py
# This module will contain logic for extracting structured outlines (Title, H1, H2, H3) from PDFs. 
import fitz  # PyMuPDF
import json
import re
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import logging
import pytesseract
import easyocr
from PIL import Image
import io
import base64
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedPDFExtractor:
    def __init__(self):
        # Initialize OCR readers
        self.tesseract_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?;:()[]{}-\'\" '
        self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
        
        # Heading detection patterns
        self.heading_patterns = [
            r'^[A-Z][A-Z\s]+$',  # ALL CAPS
            r'^\d+\.\s+[A-Z]',   # Numbered headings
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*$',  # Title Case
            r'^[IVX]+\.\s+[A-Z]',  # Roman numerals
            r'^[A-Z]\.[A-Z]',     # A.B format
            r'^Chapter\s+\d+',    # Chapter headings
            r'^Section\s+\d+',    # Section headings
            r'^[A-Z][A-Z\s]{2,}$',  # Short ALL CAPS
        ]
        
        self.title_patterns = [
            r'^[A-Z][A-Z\s]{3,}$',  # ALL CAPS with minimum length
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){2,}$',  # Title Case with multiple words
        ]
        
        self.exclude_patterns = [
            r'^\d+$',  # Just numbers
            r'^[a-z]+$',  # Just lowercase
            r'^[A-Z]$',   # Single letter
            r'^\s*$',     # Empty or whitespace only
            r'^[^\w\s]*$',  # Only special characters
        ]

    def preprocess_image(self, image: np.ndarray) -> List[np.ndarray]:
        """Advanced image preprocessing for better OCR results."""
        processed_images = []
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. Original grayscale
        processed_images.append(gray)
        
        # 2. Resize for better OCR (but keep original too)
        height, width = gray.shape
        if width > 2000:
            scale = 2000 / width
            resized = cv2.resize(gray, None, fx=scale, fy=scale)
            processed_images.append(resized)
        
        # 3. Denoise
        denoised = cv2.fastNlMeansDenoising(gray)
        processed_images.append(denoised)
        
        # 4. Adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        processed_images.append(adaptive_thresh)
        
        # 5. Otsu thresholding
        _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed_images.append(otsu_thresh)
        
        # 6. Morphological operations
        kernel = np.ones((1, 1), np.uint8)
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        processed_images.append(morph)
        
        # 7. Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        processed_images.append(enhanced)
        
        # 8. Gaussian blur for noise reduction
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        processed_images.append(blurred)
        
        # 9. Edge enhancement
        kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel_sharpen)
        processed_images.append(sharpened)
        
        return processed_images

    def extract_text_with_tesseract(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract text using Tesseract OCR with detailed positioning."""
        return self.extract_text_with_tesseract_custom(image, self.tesseract_config)

    def extract_text_with_tesseract_custom(self, image: np.ndarray, config: str) -> List[Dict[str, Any]]:
        """Extract text using Tesseract OCR with custom configuration."""
        results = []
        
        # Get detailed OCR data
        data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
        
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            if not text or int(data['conf'][i]) < 20:  # Lower confidence threshold for better coverage
                continue
                
            # Get bounding box
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            
            # Calculate font size (height of bounding box)
            font_size = h
            
            # Determine if bold (based on confidence and width)
            is_bold = int(data['conf'][i]) > 70 and w > h * 0.8
            
            # Calculate position
            center_x = x + w // 2
            center_y = y + h // 2
            
            result = {
                'text': text,
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'center_x': center_x,
                'center_y': center_y,
                'font_size': font_size,
                'confidence': int(data['conf'][i]),
                'is_bold': is_bold,
                'block_num': data['block_num'][i],
                'line_num': data['line_num'][i],
                'word_num': data['word_num'][i],
                'source': 'tesseract'
            }
            results.append(result)
        
        return results

    def extract_text_with_easyocr(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract text using EasyOCR with detailed positioning."""
        results = []
        
        # Run EasyOCR
        ocr_results = self.easyocr_reader.readtext(image)
        
        for (bbox, text, confidence) in ocr_results:
            if not text.strip() or confidence < 0.3:  # Filter low confidence
                continue
                
            # Parse bounding box
            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = bbox
            x, y = int(x1), int(y1)
            w, h = int(x2 - x1), int(y2 - y1)
            
            # Calculate center
            center_x = x + w // 2
            center_y = y + h // 2
            
            result = {
                'text': text.strip(),
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'center_x': center_x,
                'center_y': center_y,
                'font_size': h,
                'confidence': confidence,
                'is_bold': confidence > 0.7 and w > h * 0.8,
                'source': 'easyocr'
            }
            results.append(result)
        
        return results

    def merge_ocr_results(self, tesseract_results: List[Dict], easyocr_results: List[Dict]) -> List[Dict[str, Any]]:
        """Merge and deduplicate OCR results from multiple engines."""
        all_results = []
        
        # Add Tesseract results
        for result in tesseract_results:
            result['source'] = 'tesseract'
            all_results.append(result)
        
        # Add EasyOCR results
        all_results.extend(easyocr_results)
        
        # Remove duplicates based on position and text similarity
        unique_results = []
        for result in all_results:
            is_duplicate = False
            for existing in unique_results:
                # Check if texts are similar and positions overlap
                text_similarity = self.calculate_text_similarity(result['text'], existing['text'])
                position_overlap = self.calculate_position_overlap(result, existing)
                
                if text_similarity > 0.8 and position_overlap > 0.5:
                    # Keep the one with higher confidence
                    if result['confidence'] > existing['confidence']:
                        unique_results.remove(existing)
                        unique_results.append(result)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_results.append(result)
        
        return unique_results

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        if not text1 or not text2:
            return 0.0
        
        # Simple character-based similarity
        set1 = set(text1.lower())
        set2 = set(text2.lower())
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0

    def calculate_position_overlap(self, result1: Dict, result2: Dict) -> float:
        """Calculate overlap between two bounding boxes."""
        # Calculate intersection area
        x1 = max(result1['x'], result2['x'])
        y1 = max(result1['y'], result2['y'])
        x2 = min(result1['x'] + result1['width'], result2['x'] + result2['width'])
        y2 = min(result1['y'] + result1['height'], result2['y'] + result2['height'])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = result1['width'] * result1['height']
        area2 = result2['width'] * result2['height']
        
        return intersection / min(area1, area2)

    def group_text_by_lines(self, ocr_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group OCR results into lines and paragraphs."""
        if not ocr_results:
            return []
        
        # Sort by y-coordinate, then by x-coordinate
        sorted_results = sorted(ocr_results, key=lambda x: (x['y'], x['x']))
        
        # Group into lines using clustering
        positions = np.array([[r['center_y']] for r in sorted_results])
        scaler = StandardScaler()
        positions_scaled = scaler.fit_transform(positions)
        
        # Use DBSCAN to cluster lines
        clustering = DBSCAN(eps=0.5, min_samples=1).fit(positions_scaled)
        
        # Group results by cluster
        lines = defaultdict(list)
        for i, result in enumerate(sorted_results):
            cluster = clustering.labels_[i]
            lines[cluster].append(result)
        
        # Sort lines by y-coordinate
        line_groups = []
        for cluster_id, line_results in lines.items():
            # Sort within line by x-coordinate
            line_results.sort(key=lambda x: x['x'])
            
            # Combine text in line
            line_text = ' '.join([r['text'] for r in line_results])
            
            # Calculate line properties
            min_x = min(r['x'] for r in line_results)
            max_x = max(r['x'] + r['width'] for r in line_results)
            min_y = min(r['y'] for r in line_results)
            max_y = max(r['y'] + r['height'] for r in line_results)
            
            # Calculate average font size
            avg_font_size = np.mean([r['font_size'] for r in line_results])
            
            # Determine if line is bold
            is_bold = any(r['is_bold'] for r in line_results)
            
            line_group = {
                'text': line_text,
                'x': min_x,
                'y': min_y,
                'width': max_x - min_x,
                'height': max_y - min_y,
                'font_size': avg_font_size,
                'is_bold': is_bold,
                'confidence': np.mean([r['confidence'] for r in line_results]),
                'words': line_results
            }
            line_groups.append(line_group)
        
        # Sort lines by y-coordinate
        line_groups.sort(key=lambda x: x['y'])
        
        return line_groups

    def is_heading_pattern(self, text: str) -> bool:
        """Check if text matches heading patterns."""
        if not text or len(text) < 2:
            return False
            
        # Check exclusion patterns first
        for pattern in self.exclude_patterns:
            if re.match(pattern, text):
                return False
        
        # Check heading patterns
        for pattern in self.heading_patterns:
            if re.match(pattern, text):
                return True
                
        return False

    def is_title_pattern(self, text: str) -> bool:
        """Check if text matches title patterns."""
        if not text or len(text) < 5:
            return False
            
        for pattern in self.title_patterns:
            if re.match(pattern, text):
                return True
                
        return False

    def determine_text_level(self, text: str, font_size: float, is_bold: bool, 
                           y_position: float, page_width: float) -> str:
        """Determine the level of text (Title, H1, H2, H3, Paragraph)."""
        
        # Clean text for analysis
        clean_text = text.strip()
        if not clean_text:
            return "Paragraph"
        
        # Check exclusion patterns
        for pattern in self.exclude_patterns:
            if re.match(pattern, clean_text):
                return "Paragraph"
        
        # Title detection
        for pattern in self.title_patterns:
            if re.match(pattern, clean_text):
                if font_size > 16 or (font_size > 14 and is_bold):
                    return "Title"
        
        # Heading detection
        for pattern in self.heading_patterns:
            if re.match(pattern, clean_text):
                if font_size >= 14 or (font_size >= 12 and is_bold):
                    return "H1"
                elif font_size >= 12 or (font_size >= 10 and is_bold):
                    return "H2"
                elif font_size >= 10:
                    return "H3"
        
        # Additional heuristics
        if len(clean_text) > 3:
            # ALL CAPS with good size
            if clean_text.isupper() and font_size >= 12:
                return "H1"
            
            # Title case with good size
            if clean_text.istitle() and font_size >= 12:
                return "H2"
            
            # Bold text with good size
            if is_bold and font_size >= 11:
                return "H2"
        
        return "Paragraph"

    def extract_from_pdf_page(self, page: fitz.Page) -> List[Dict[str, Any]]:
        """Extract text from a single PDF page using multiple methods."""
        results = []
        
        # Method 1: Try native text extraction first
        try:
            text_dict = page.get_text("dict")
            for block in text_dict.get("blocks", []):
                if "lines" not in block:
                    continue
                    
                for line in block["lines"]:
                    line_text = ""
                    line_font_size = 0
                    line_is_bold = False
                    line_is_italic = False
                    line_font_name = ""
                    line_x = float('inf')
                    line_y = float('inf')
                    line_width = 0
                    line_height = 0
                    
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text:
                            line_text += text + " "
                            line_font_size = max(line_font_size, span["size"])
                            line_is_bold = line_is_bold or "bold" in span["font"].lower()
                            line_is_italic = line_is_italic or "italic" in span["font"].lower()
                            line_font_name = span["font"]  # Keep the last font name
                            
                            bbox = span["bbox"]
                            line_x = min(line_x, bbox[0])
                            line_y = min(line_y, bbox[1])
                            line_width = max(line_width, bbox[2] - bbox[0])
                            line_height = max(line_height, bbox[3] - bbox[1])
                    
                    if line_text.strip():
                        result = {
                            'text': line_text.strip(),
                            'x': line_x,
                            'y': line_y,
                            'width': line_width,
                            'height': line_height,
                            'font_size': line_font_size,
                            'font': line_font_name,
                            'is_bold': line_is_bold,
                            'is_italic': line_is_italic,
                            'confidence': 1.0,
                            'source': 'native'
                        }
                        results.append(result)
        except Exception as e:
            logger.warning(f"Native text extraction failed: {e}")
        
        # Method 2: Use EasyOCR only if native extraction found very little text
        if len(results) < 3:  # Only use OCR if we found very little text
            try:
                # Convert page to image with moderate resolution for speed
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes("png")
                
                # Convert to OpenCV format
                nparr = np.frombuffer(img_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is not None:
                    # Use only the best preprocessing techniques for speed
                    processed_images = [image]  # Original image
                    
                    # Add one enhanced version
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    enhanced = clahe.apply(gray)
                    processed_images.append(enhanced)
                    
                    # Extract text with EasyOCR
                    all_ocr_results = []
                    
                    for processed_img in processed_images:
                        try:
                            easyocr_results = self.extract_text_with_easyocr(processed_img)
                            all_ocr_results.extend(easyocr_results)
                        except Exception as e:
                            logger.warning(f"EasyOCR failed: {e}")
                    
                    # Group into lines
                    line_groups = self.group_text_by_lines(all_ocr_results)
                    
                    # Add OCR results, but don't overwrite native results
                    for line in line_groups:
                        # Check if this text is already found by native extraction
                        is_duplicate = False
                        for native_result in results:
                            if self.calculate_text_similarity(line['text'], native_result['text']) > 0.8:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            result = {
                                'text': line['text'],
                                'x': line['x'],
                                'y': line['y'],
                                'width': line['width'],
                                'height': line['height'],
                                'font_size': line['font_size'],
                                'font': 'Unknown',  # OCR doesn't provide font info
                                'is_bold': line['is_bold'],
                                'is_italic': False,  # OCR doesn't provide italic info
                                'confidence': line['confidence'],
                                'source': 'ocr'
                            }
                            results.append(result)
                            
            except Exception as e:
                logger.warning(f"OCR extraction failed: {e}")
        
        return results

    def extract_pdf_outline(self, pdf_path: str) -> Dict[str, Any]:
        """Main function to extract PDF outline with maximum accuracy."""
        doc = None
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Open PDF
            doc = fitz.open(pdf_path)
            
            all_text_blocks = []
            page_width = 0
            
            # Process each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get page dimensions
                if page_num == 0:
                    page_width = page.rect.width
                
                # Extract text from page
                page_results = self.extract_from_pdf_page(page)
                
                # Add page number and determine levels
                for result in page_results:
                    result['page'] = page_num + 1
                    result['level'] = self.determine_text_level(
                        result['text'], 
                        result['font_size'], 
                        result['is_bold'],
                        result['y'],
                        page_width
                    )
                    
                    # Calculate alignment
                    result['left_aligned'] = result['x'] < page_width * 0.1
                    result['centered'] = abs(result['x'] - (page_width / 2)) < 50
                    
                    # Add additional properties
                    result = self.add_advanced_properties(result)
                    
                    all_text_blocks.append(result)
            
            # Find title (usually the first large, centered text)
            title = self.find_title(all_text_blocks)
            
            # Extract headings
            headings = [block for block in all_text_blocks if block['level'] in ['Title', 'H1', 'H2', 'H3']]
            
            # Sort headings by page, then by position
            headings.sort(key=lambda x: (x['page'], x['y'], x['x']))
            
            # Create outline structure
            outline = []
            for heading in headings:
                outline.append({
                    "level": heading["level"],
                    "text": heading["text"],
                    "page": heading["page"]
                })
            
            result = {
                "title": title,
                "outline": outline,
                "total_pages": len(doc),
                "total_blocks": len(all_text_blocks),
                "headings_found": len(headings),
                "text_blocks": all_text_blocks  # Include all detailed text blocks
            }
            
            logger.info(f"Extracted {len(headings)} headings from {len(doc)} pages")
            logger.info(f"Total text blocks found: {len(all_text_blocks)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return {
                "title": "Error processing PDF",
                "outline": [],
                "error": str(e),
                "total_pages": 0,
                "total_blocks": 0,
                "headings_found": 0,
                "text_blocks": []
            }
        finally:
            if doc:
                doc.close()

    def add_advanced_properties(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Add advanced properties to text block."""
        text = result['text']
        
        # Text analysis
        result['heading_pattern'] = self.is_heading_pattern(text)
        result['title_case'] = self.is_title_pattern(text)
        result['all_caps'] = text.isupper() and len(text) > 2
        result['heading_punct'] = bool(re.search(r'[.!?]$', text))
        
        # Text type classification
        result['text_type'] = self.classify_text_type(text)
        
        # Word count and character analysis
        result['word_count'] = len(text.split())
        result['char_count'] = len(text)
        result['has_numbers'] = bool(re.search(r'\d', text))
        result['has_special_chars'] = bool(re.search(r'[^\w\s]', text))
        
        # Position analysis
        result['is_top_of_page'] = result['y'] < 100
        result['is_bottom_of_page'] = result['y'] > 700  # Assuming standard page height
        
        # Font analysis
        result['font_weight'] = 'bold' if result['is_bold'] else 'normal'
        result['font_style'] = 'italic' if result.get('is_italic', False) else 'normal'
        
        # Semantic similarity placeholder (can be enhanced with actual ML model)
        result['semantic_sim'] = 0.0
        
        # Confidence and quality metrics
        result['quality_score'] = self.calculate_quality_score(result)
        
        return result

    def classify_text_type(self, text: str) -> str:
        """Classify text as word, sentence, heading, etc."""
        if not text:
            return "empty"
        
        # Check if it's a heading
        if self.is_heading_pattern(text):
            return "heading"
        
        # Check if it's a sentence (ends with punctuation and has multiple words)
        if re.search(r'[.!?]$', text) and len(text.split()) > 3:
            return "sentence"
        
        # Check if it's a single word or short phrase
        if len(text.split()) <= 2:
            return "word"
        
        # Check if it's a paragraph (multiple sentences)
        if len(text.split()) > 10:
            return "paragraph"
        
        # Default to phrase
        return "phrase"

    def calculate_quality_score(self, result: Dict[str, Any]) -> float:
        """Calculate a quality score for the text block."""
        score = 0.0
        
        # Base score from confidence
        score += result.get('confidence', 0.0) * 0.4
        
        # Font size bonus
        font_size = result.get('font_size', 0)
        if font_size > 12:
            score += 0.2
        elif font_size > 10:
            score += 0.1
        
        # Bold text bonus
        if result.get('is_bold', False):
            score += 0.1
        
        # Text length bonus (not too short, not too long)
        text_length = len(result.get('text', ''))
        if 3 <= text_length <= 100:
            score += 0.1
        
        # Position bonus (not at edges)
        if 50 < result.get('x', 0) < 500:
            score += 0.1
        
        return min(score, 1.0)

    def find_title(self, text_blocks: List[Dict[str, Any]]) -> str:
        """Find the document title from text blocks."""
        # Look for title on first page
        first_page_blocks = [b for b in text_blocks if b["page"] == 1]
        
        # Sort by font size (largest first)
        first_page_blocks.sort(key=lambda x: x["font_size"], reverse=True)
        
        for block in first_page_blocks[:10]:  # Check top 10 largest
            if block["level"] == "Title" or (block["font_size"] > 14 and block["centered"]):
                return block["text"]
        
        # Fallback: return the first large text
        for block in first_page_blocks:
            if block["font_size"] > 12:
                return block["text"]
        
        return "Untitled Document"

# Global instance
extractor = AdvancedPDFExtractor()

def extract_outline(pdf_path: str) -> Dict[str, Any]:
    """Main function to extract PDF outline."""
    return extractor.extract_pdf_outline(pdf_path) 