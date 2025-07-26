# main.py
# Entry point for PDF processing pipeline

import os
import json
from outline_extractor import extract_outline

def process_all_pdfs(input_dir: str, output_dir: str):
    """Process all PDFs in input directory and save comprehensive results to output directory."""
    print(f"Processing PDFs from: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all PDF files
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in input directory!")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        output_file = os.path.join(output_dir, pdf_file.replace('.pdf', '.json'))
        
        print(f"\nProcessing: {pdf_file}")
        
        try:
            # Extract comprehensive outline with all text blocks
            result = extract_outline(pdf_path)
            
            # Save to JSON file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print(f"✓ Saved results to: {output_file}")
            print(f"  - Title: {result.get('title', 'N/A')}")
            print(f"  - Headings found: {result.get('headings_found', 0)}")
            print(f"  - Total pages: {result.get('total_pages', 0)}")
            print(f"  - Total text blocks: {result.get('total_blocks', 0)}")
            
            # Show some sample headings
            outline = result.get('outline', [])
            if outline:
                print(f"  - Sample headings:")
                for i, heading in enumerate(outline[:3]):  # Show first 3 headings
                    print(f"    {i+1}. {heading['level']}: {heading['text']} (Page {heading['page']})")
                if len(outline) > 3:
                    print(f"    ... and {len(outline) - 3} more headings")
            
        except Exception as e:
            print(f"✗ Error processing {pdf_file}: {str(e)}")
    
    print(f"\nProcessing complete! Check {output_dir} for results.")

if __name__ == "__main__":
    # Use relative paths for local development
    input_dir = "app/input"
    output_dir = "app/output"
    
    # Check if we're running in Docker (absolute paths)
    if os.path.exists("/app/input"):
        input_dir = "/app/input"
        output_dir = "/app/output"
    
    # Process PDFs with advanced extraction
    process_all_pdfs(input_dir, output_dir)