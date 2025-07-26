# Adobe Hackathon: Connecting the Dots - Round 1A

## Overview
This project implements an advanced PDF outline extraction system for the Adobe India Hackathon 2025. The solution extracts structured outlines (Title, H1, H2, H3) from PDFs with comprehensive text analysis and positioning information.

## Round 1A: PDF Outline Extraction

### Features
- **Comprehensive Text Extraction**: Extracts every text element with detailed properties
- **Advanced OCR Integration**: Uses EasyOCR for image-based PDFs
- **Smart Heading Detection**: Pattern-based and ML-enhanced heading classification
- **Detailed Output**: Each text block includes coordinates, font properties, and classification
- **High Accuracy**: Works with both digital and image-based PDFs

### Technical Approach

#### 1. Multi-Method Text Extraction
- **Native PDF Extraction**: Uses PyMuPDF for digital PDFs
- **OCR Processing**: EasyOCR for image-based PDFs with preprocessing
- **Smart Deduplication**: Merges results from multiple methods

#### 2. Advanced Text Analysis
- **Pattern Recognition**: Detects ALL CAPS, Title Case, numbered headings
- **Font Analysis**: Extracts font name, size, bold, italic properties
- **Position Analysis**: Calculates alignment and positioning
- **Text Classification**: Categorizes as word, sentence, heading, or paragraph

#### 3. Comprehensive Output Format
Each text block includes:
- Basic properties: text, page, level, coordinates
- Font properties: font name, size, bold, italic
- Analysis: heading patterns, text type, quality score
- Position: alignment, page position indicators

### Libraries Used
- **PyMuPDF**: Native PDF text extraction
- **EasyOCR**: Optical Character Recognition
- **OpenCV**: Image preprocessing
- **scikit-learn**: Text clustering and analysis
- **NumPy**: Numerical operations

### Performance
- **Execution Time**: < 10 seconds per PDF
- **Model Size**: < 200MB (optimized dependencies)
- **CPU-only**: No GPU required
- **Offline Operation**: No internet access needed

## Setup and Usage

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app/src/main.py
```

### Docker Deployment
```bash
# Build the Docker image
docker build --platform linux/amd64 -t pdf-extractor:latest .

# Run the container
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none pdf-extractor:latest
```

### Input/Output
- **Input**: PDF files in `/app/input` directory
- **Output**: JSON files in `/app/output` directory
- **Format**: Comprehensive JSON with outline and detailed text blocks

## Output Format
```json
{
  "title": "Document Title",
  "outline": [
    {
      "level": "H1",
      "text": "Heading Text",
      "page": 1
    }
  ],
  "total_pages": 14,
  "total_blocks": 550,
  "headings_found": 37,
  "text_blocks": [
    {
      "text": "Sample Text",
      "page": 1,
      "level": "H1",
      "font": "Helvetica-Bold",
      "is_bold": true,
      "is_italic": false,
      "x": 100.0,
      "y": 200.0,
      "left_aligned": false,
      "centered": true,
      "heading_pattern": true,
      "title_case": false,
      "all_caps": true,
      "text_type": "heading",
      "quality_score": 0.9
    }
  ]
}
```

## Project Structure
```
├── app/
│   ├── src/
│   │   ├── main.py              # Main entry point
│   │   ├── outline_extractor.py # Core extraction engine
│   │   └── ...
│   ├── input/                   # Input PDF files
│   └── output/                  # Output JSON files
├── Dockerfile                   # Docker configuration
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Round 1A Requirements Compliance
✅ **PDF Input**: Accepts PDFs up to 50 pages  
✅ **Text Extraction**: Title, H1, H2, H3 with levels and page numbers  
✅ **JSON Output**: Valid JSON format  
✅ **CPU-only**: No GPU dependencies  
✅ **Model Size**: < 200MB  
✅ **Execution Time**: < 10 seconds per PDF  
✅ **Offline Operation**: No internet access required  
✅ **Docker Support**: Fully containerized solution  

## Next Steps
- **Round 1B**: Persona-driven document intelligence
- **Round 2**: Futuristic webapp with Adobe PDF Embed API 