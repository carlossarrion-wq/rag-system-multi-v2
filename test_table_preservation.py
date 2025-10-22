#!/usr/bin/env python3
"""
Test script for table preservation functionality
Tests the SemanticChunker with various table formats
"""

import sys
import os
sys.path.append('src')

from src.indexing.semantic_chunker import SemanticChunker
from loguru import logger

def test_pipe_separated_table():
    """Test with pipe-separated table"""
    logger.info("Testing pipe-separated table...")
    
    text = """
Este documento contiene información importante sobre códigos de sistema.

| Código | Descripción | Módulo | Estado |
|--------|-------------|---------|---------|
| AC01   | Acceso principal | GADEA | Activo |
| AC02   | Acceso secundario | GADEA | Inactivo |
| Z001   | Configuración base | SAP | Activo |
| Z002   | Configuración avanzada | SAP | Activo |

La tabla anterior muestra los códigos más importantes del sistema.
Estos códigos son fundamentales para el funcionamiento correcto.

Información adicional:
- Los códigos AC están relacionados con accesos
- Los códigos Z están relacionados con configuraciones
"""
    
    chunker = SemanticChunker(chunk_size=500, chunk_overlap=100)
    chunks = chunker.chunk_with_table_preservation(text)
    
    logger.info(f"Generated {len(chunks)} chunks")
    
    for i, chunk in enumerate(chunks):
        logger.info(f"\n--- Chunk {i} ---")
        logger.info(f"Type: {chunk['chunk_type']}")
        logger.info(f"Content type: {chunk['metadata'].get('content_type', 'unknown')}")
        logger.info(f"Contains codes: {chunk['metadata'].get('contains_codes', False)}")
        logger.info(f"Technical codes: {chunk['metadata'].get('technical_codes', [])}")
        
        if chunk['chunk_type'] == 'table':
            logger.info(f"Table rows: {chunk['metadata'].get('table_rows_count', 0)}")
            logger.info(f"Table columns: {chunk['metadata'].get('table_columns_count', 0)}")
            logger.info(f"Table headers: {chunk['metadata'].get('table_headers', [])}")
        
        logger.info(f"Text preview: {chunk['text'][:200]}...")
    
    return chunks

def test_space_separated_table():
    """Test with space-separated table"""
    logger.info("\nTesting space-separated table...")
    
    text = """
Configuración de módulos del sistema DARWIN:

Módulo          Estado      Versión     Responsable
DARWIN_CORE     Activo      2.1.3       Admin
DARWIN_UI       Activo      1.8.2       Frontend
DARWIN_API      Inactivo    1.5.1       Backend
PDS_INTEGRATION Activo      3.2.0       Integration

Esta configuración debe mantenerse actualizada.
"""
    
    chunker = SemanticChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk_with_table_preservation(text)
    
    logger.info(f"Generated {len(chunks)} chunks")
    
    for i, chunk in enumerate(chunks):
        logger.info(f"\n--- Chunk {i} ---")
        logger.info(f"Type: {chunk['chunk_type']}")
        logger.info(f"Module detected: {chunk['metadata'].get('module', 'None')}")
        logger.info(f"Technical codes: {chunk['metadata'].get('technical_codes', [])}")
        logger.info(f"Text: {chunk['text']}")
    
    return chunks

def test_mixed_content():
    """Test with mixed content (text + table + text)"""
    logger.info("\nTesting mixed content...")
    
    text = """
# Manual de Usuario - Sistema GADEA

## Introducción
El sistema GADEA es una plataforma integral para la gestión empresarial.
Incluye módulos para contabilidad, recursos humanos y gestión de proyectos.

## Códigos de Error Principales

| Código | Tipo | Descripción | Solución |
|--------|------|-------------|----------|
| ERR001 | Sistema | Error de conexión | Verificar red |
| ERR002 | Usuario | Credenciales inválidas | Resetear password |
| ERR003 | Base de datos | Timeout de consulta | Optimizar query |
| SAP_001 | Integración | Error SAP | Revisar configuración |

## Procedimientos de Resolución

Para resolver los errores mencionados en la tabla anterior:

1. **ERR001**: Verificar la conectividad de red
   - Ping al servidor
   - Verificar firewall
   - Comprobar DNS

2. **ERR002**: Gestión de credenciales
   - Verificar usuario existe
   - Resetear contraseña si es necesario
   - Comprobar permisos

3. **ERR003**: Optimización de base de datos
   - Revisar índices
   - Analizar plan de ejecución
   - Considerar particionado

## Contacto
Para más información, contactar al equipo de soporte técnico.
"""
    
    chunker = SemanticChunker(chunk_size=800, chunk_overlap=150)
    chunks = chunker.chunk_with_table_preservation(text)
    
    logger.info(f"Generated {len(chunks)} chunks")
    
    table_chunks = 0
    text_chunks = 0
    total_codes = 0
    
    for i, chunk in enumerate(chunks):
        logger.info(f"\n--- Chunk {i} ---")
        logger.info(f"Type: {chunk['chunk_type']}")
        logger.info(f"Content type: {chunk['metadata'].get('content_type', 'unknown')}")
        logger.info(f"Contains codes: {chunk['metadata'].get('contains_codes', False)}")
        
        codes = chunk['metadata'].get('technical_codes', [])
        total_codes += len(codes)
        logger.info(f"Technical codes: {codes}")
        
        if chunk['chunk_type'] == 'table':
            table_chunks += 1
            logger.info(f"Table preserved with {chunk['metadata'].get('table_rows_count', 0)} rows")
        else:
            text_chunks += 1
        
        logger.info(f"Length: {len(chunk['text'])} chars")
        logger.info(f"Text preview: {chunk['text'][:150]}...")
    
    logger.info(f"\nSummary:")
    logger.info(f"- Total chunks: {len(chunks)}")
    logger.info(f"- Table chunks: {table_chunks}")
    logger.info(f"- Text chunks: {text_chunks}")
    logger.info(f"- Total technical codes found: {total_codes}")
    
    return chunks

def test_no_table_content():
    """Test with content that has no tables"""
    logger.info("\nTesting content without tables...")
    
    text = """
Este es un documento de texto normal sin tablas.
Contiene algunos códigos técnicos como AC01 y Z001.
También menciona el sistema SAP y GADEA.

El documento tiene varios párrafos con información importante.
Incluye procedimientos y configuraciones del sistema.
Los códigos SAP_MODULE_CONFIG y DARWIN_CORE son relevantes.

Esta es información adicional que debe ser procesada
correctamente por el sistema de chunking semántico.
"""
    
    chunker = SemanticChunker(chunk_size=200, chunk_overlap=50)
    chunks = chunker.chunk_with_table_preservation(text)
    
    logger.info(f"Generated {len(chunks)} chunks")
    
    for i, chunk in enumerate(chunks):
        logger.info(f"\n--- Chunk {i} ---")
        logger.info(f"Type: {chunk['chunk_type']}")
        logger.info(f"Content type: {chunk['metadata'].get('content_type', 'unknown')}")
        logger.info(f"Module: {chunk['metadata'].get('module', 'None')}")
        logger.info(f"Technical codes: {chunk['metadata'].get('technical_codes', [])}")
        logger.info(f"Text: {chunk['text']}")
    
    return chunks

def main():
    """Run all tests"""
    logger.info("🚀 Starting Table Preservation Tests")
    logger.info("=" * 60)
    
    try:
        # Test 1: Pipe-separated table
        chunks1 = test_pipe_separated_table()
        
        # Test 2: Space-separated table
        chunks2 = test_space_separated_table()
        
        # Test 3: Mixed content
        chunks3 = test_mixed_content()
        
        # Test 4: No tables
        chunks4 = test_no_table_content()
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ All tests completed successfully!")
        logger.info(f"Total chunks generated across all tests: {len(chunks1) + len(chunks2) + len(chunks3) + len(chunks4)}")
        
        # Verify table preservation
        table_chunks_found = 0
        for test_chunks in [chunks1, chunks2, chunks3, chunks4]:
            table_chunks_found += sum(1 for chunk in test_chunks if chunk['chunk_type'] == 'table')
        
        logger.info(f"Table chunks preserved: {table_chunks_found}")
        
        if table_chunks_found > 0:
            logger.info("🎉 Table preservation is working correctly!")
        else:
            logger.warning("⚠️  No table chunks found - check table detection logic")
            
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
