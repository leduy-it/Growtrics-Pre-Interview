#!/usr/bin/env python3
"""
OCR Question Extraction Pipeline - Main Entry Point

Usage:
    python src/main.py --pdf path/to/exam.pdf --name math_exam
    python src/main.py --pdf math.pdf --name math_exam --output outputs
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.ocr_workflow import run_ocr_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract questions from exam PDFs using AI"
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default="./data",  
        help="Path to the PDF file or folder to process"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="exam",
        help="Name of the exam (used for output folder name)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Base output directory (default: outputs)"
    )
    return parser.parse_args()


async def process_single_pdf(pdf_path: Path, output_base: Path, exam_name: str):
    """Process a single PDF file."""
    output_dir = output_base / exam_name
    
    print("=" * 60)
    print(f"Processing: {pdf_path.name}")
    print("=" * 60)
    print(f"Exam Name: {exam_name}")
    print(f"Output Dir: {output_dir}")
    print("=" * 60)
    print()
    
    try:
        # Run pipeline
        result = await run_ocr_pipeline(
            pdf_path=str(pdf_path),
            output_dir=str(output_dir),
            exam_name=exam_name
        )
        
        # Print summary
        print()
        print("=" * 60)
        print(f"[OK] COMPLETED: {pdf_path.name}")
        print("=" * 60)
        print(f"Total Questions: {result.total_questions}")
        
        for q in result.questions:
            print(f"\n  Question {q.question_number}:")
            for part in q.question_parts:
                diagrams = f" [{len(part.diagrams)} diagram(s)]" if part.diagrams else ""
                print(f"    - {part.part_id}: {part.answer_type.value}{diagrams}")
        
        print()
        print(f"Output saved to: {output_dir}")
        print("=" * 60)
        print()
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Error processing {pdf_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    args = parse_args()
    
    # Validate path exists
    input_path = Path(args.pdf)
    if not input_path.exists():
        print(f"Error: Path not found: {input_path}")
        sys.exit(1)
    
    output_base = Path(args.output)
    
    # Determine if it's a file or directory
    if input_path.is_file():
        # Single PDF file
        if not input_path.suffix.lower() == '.pdf':
            print(f"Error: File must be a PDF: {input_path}")
            sys.exit(1)
        
        pdfs_to_process = [(input_path, args.name)]
        
    elif input_path.is_dir():
        # Directory - find all PDFs
        pdf_files = sorted(input_path.glob("*.pdf"))
        
        if not pdf_files:
            print(f"Error: No PDF files found in directory: {input_path}")
            sys.exit(1)
        
        print(f"\n[INFO] Found {len(pdf_files)} PDF(s) in {input_path}")
        for pdf in pdf_files:
            print(f"   - {pdf.name}")
        print()
        
        # Generate exam names from filenames
        pdfs_to_process = [
            (pdf, pdf.stem.replace(" ", "_"))  # Use filename without extension as exam name
            for pdf in pdf_files
        ]
    else:
        print(f"Error: Invalid path: {input_path}")
        sys.exit(1)
    
    # Process all PDFs
    print("\n" + "=" * 60)
    print("OCR QUESTION EXTRACTION PIPELINE")
    print("=" * 60)
    print(f"Total PDFs to process: {len(pdfs_to_process)}")
    print("=" * 60)
    print()
    
    results = []
    for i, (pdf_path, exam_name) in enumerate(pdfs_to_process, 1):
        print(f"\n[PDF] [{i}/{len(pdfs_to_process)}] Processing: {pdf_path.name}")
        success = await process_single_pdf(pdf_path, output_base, exam_name)
        results.append((pdf_path.name, success))
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    successful = sum(1 for _, success in results if success)
    failed = len(results) - successful
    
    print(f"[OK] Successful: {successful}/{len(results)}")
    if failed > 0:
        print(f"[FAIL] Failed: {failed}/{len(results)}")
        print("\nFailed files:")
        for name, success in results:
            if not success:
                print(f"   - {name}")
    
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
