#!/usr/bin/env python3
"""
Script to convert Office files (DOCX, PPTX) to PDF to avoid EMF/WMF image format issues.
This can be run as a preprocessing step before indexing.

Requirements:
- LibreOffice (for headless conversion)
  Install: sudo amazon-linux-extras install libreoffice

Usage:
    python3 scripts/convert_office_to_pdf.py --file document.docx
    python3 scripts/convert_office_to_pdf.py --file presentation.pptx
    python3 scripts/convert_office_to_pdf.py --directory <directory>
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from loguru import logger

# Supported file extensions
SUPPORTED_EXTENSIONS = {'.docx', '.pptx'}

def check_libreoffice_installed():
    """Check if LibreOffice is installed"""
    # Try different command names (libreoffice on Linux, soffice on macOS)
    commands = ['soffice', 'libreoffice']
    
    for cmd in commands:
        try:
            result = subprocess.run([cmd, '--version'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            if result.returncode == 0:
                logger.info(f"LibreOffice found: {result.stdout.strip()}")
                return cmd  # Return the working command
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    
    logger.error("LibreOffice not found. Please install it:")
    logger.error("  macOS: brew install --cask libreoffice")
    logger.error("  Linux: sudo amazon-linux-extras install libreoffice")
    return None

def convert_office_to_pdf(office_path: str, output_dir: str = None, libreoffice_cmd: str = None) -> str:
    """
    Convert an Office file (DOCX, PPTX) to PDF using LibreOffice headless mode.
    
    Args:
        office_path: Path to the Office file (DOCX or PPTX)
        output_dir: Directory for output PDF (default: same as input)
        libreoffice_cmd: LibreOffice command to use (default: auto-detect)
        
    Returns:
        Path to the generated PDF file
    """
    office_path = Path(office_path).resolve()
    
    if not office_path.exists():
        raise FileNotFoundError(f"File not found: {office_path}")
    
    if office_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {office_path.suffix}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")
    
    # Auto-detect LibreOffice command if not provided
    if libreoffice_cmd is None:
        libreoffice_cmd = check_libreoffice_installed()
        if not libreoffice_cmd:
            raise RuntimeError("LibreOffice not found")
    
    # Determine output directory
    if output_dir is None:
        output_dir = office_path.parent
    else:
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Expected PDF output path
    pdf_path = output_dir / f"{office_path.stem}.pdf"
    
    file_type = "Word document" if office_path.suffix.lower() == '.docx' else "PowerPoint presentation"
    logger.info(f"Converting {file_type}: {office_path.name} to PDF...")
    
    try:
        # Run LibreOffice in headless mode to convert
        cmd = [
            libreoffice_cmd,
            '--headless',
            '--convert-to', 'pdf',
            '--outdir', str(output_dir),
            str(office_path)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 120 second timeout (PPTX can take longer)
        )
        
        if result.returncode != 0:
            logger.error(f"Conversion failed: {result.stderr}")
            raise RuntimeError(f"LibreOffice conversion failed: {result.stderr}")
        
        if not pdf_path.exists():
            raise RuntimeError(f"PDF was not created: {pdf_path}")
        
        logger.success(f"✓ Created: {pdf_path}")
        return str(pdf_path)
        
    except subprocess.TimeoutExpired:
        logger.error(f"Conversion timed out for {office_path.name}")
        raise
    except Exception as e:
        logger.error(f"Error converting {office_path.name}: {e}")
        raise

def convert_directory(directory: str, output_dir: str = None, recursive: bool = False, file_types: set = None, libreoffice_cmd: str = None):
    """
    Convert all Office files in a directory to PDF.
    
    Args:
        directory: Directory containing Office files
        output_dir: Directory for output PDFs (default: same as input)
        recursive: Whether to search subdirectories
        file_types: Set of file extensions to convert (default: all supported)
        libreoffice_cmd: LibreOffice command to use (default: auto-detect)
    """
    directory = Path(directory).resolve()
    
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if file_types is None:
        file_types = SUPPORTED_EXTENSIONS
    
    # Find all Office files
    office_files = []
    for ext in file_types:
        if recursive:
            office_files.extend(directory.rglob(f'*{ext}'))
        else:
            office_files.extend(directory.glob(f'*{ext}'))
    
    # Filter out temporary files (starting with ~$)
    office_files = [f for f in office_files if not f.name.startswith('~$')]
    
    if not office_files:
        logger.warning(f"No Office files found in {directory}")
        return
    
    # Count by type
    docx_count = sum(1 for f in office_files if f.suffix.lower() == '.docx')
    pptx_count = sum(1 for f in office_files if f.suffix.lower() == '.pptx')
    
    logger.info(f"Found {len(office_files)} Office files to convert:")
    if docx_count > 0:
        logger.info(f"  - {docx_count} Word documents (.docx)")
    if pptx_count > 0:
        logger.info(f"  - {pptx_count} PowerPoint presentations (.pptx)")
    
    success_count = 0
    error_count = 0
    
    for office_file in office_files:
        try:
            # Maintain directory structure if recursive
            if recursive and output_dir:
                relative_path = office_file.parent.relative_to(directory)
                file_output_dir = Path(output_dir) / relative_path
            else:
                file_output_dir = output_dir
            
            convert_office_to_pdf(str(office_file), str(file_output_dir) if file_output_dir else None, libreoffice_cmd)
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to convert {office_file.name}: {e}")
            error_count += 1
    
    logger.info(f"\nConversion complete:")
    logger.info(f"  ✓ Success: {success_count}")
    if error_count > 0:
        logger.warning(f"  ✗ Errors: {error_count}")

def main():
    parser = argparse.ArgumentParser(
        description='Convert Office files (DOCX, PPTX) to PDF to avoid EMF/WMF image issues'
    )
    
    parser.add_argument('--check',
                       action='store_true',
                       help='Only check if LibreOffice is installed')
    
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--file', '-f', 
                      help='Single Office file to convert (DOCX or PPTX)')
    group.add_argument('--directory', '-d',
                      help='Directory containing Office files')
    
    parser.add_argument('--output', '-o',
                       help='Output directory for PDFs (default: same as input)')
    parser.add_argument('--recursive', '-r',
                       action='store_true',
                       help='Search subdirectories recursively')
    parser.add_argument('--type', '-t',
                       choices=['docx', 'pptx', 'all'],
                       default='all',
                       help='File type to convert (default: all)')
    
    args = parser.parse_args()
    
    # Check LibreOffice installation and get the command
    libreoffice_cmd = check_libreoffice_installed()
    if not libreoffice_cmd:
        sys.exit(1)
    
    if args.check:
        logger.success("LibreOffice is installed and ready")
        sys.exit(0)
    
    # Require file or directory if not just checking
    if not args.file and not args.directory:
        parser.error("one of the arguments --file/-f --directory/-d is required")
    
    # Determine file types to process
    if args.type == 'docx':
        file_types = {'.docx'}
    elif args.type == 'pptx':
        file_types = {'.pptx'}
    else:
        file_types = SUPPORTED_EXTENSIONS
    
    try:
        if args.file:
            # Convert single file
            pdf_path = convert_office_to_pdf(args.file, args.output, libreoffice_cmd)
            logger.success(f"PDF created: {pdf_path}")
        else:
            # Convert directory (pass libreoffice_cmd through convert_office_to_pdf calls)
            convert_directory(args.directory, args.output, args.recursive, file_types, libreoffice_cmd)
            
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
