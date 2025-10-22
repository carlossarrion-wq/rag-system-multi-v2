#!/usr/bin/env python3
"""
Script de pruebas para el parser JSON mejorado
Procesa todos los archivos de logs y extrae las respuestas JSON entre las secciones especificadas
"""

import os
import sys
import json
import glob
import re
from datetime import datetime

# Añadir el directorio src al path
sys.path.append('src')

try:
    from generation.structured_response_parser import StructuredResponseParser
    PARSER_AVAILABLE = True
except ImportError:
    print("⚠️ Parser no disponible, usando solo validación JSON básica")
    PARSER_AVAILABLE = False


def extract_json_from_log(file_path):
    """
    Extrae el JSON entre las secciones === RAW LLM RESPONSE === y === RESPONSE ANALYSIS ===
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Buscar la sección RAW LLM RESPONSE
        start_marker = "=== RAW LLM RESPONSE ==="
        end_marker = "=== RESPONSE ANALYSIS ==="
        
        start_idx = content.find(start_marker)
        end_idx = content.find(end_marker)
        
        if start_idx == -1 or end_idx == -1:
            return None, "Marcadores de sección no encontrados"
        
        # Extraer el contenido entre los marcadores
        raw_response = content[start_idx + len(start_marker):end_idx].strip()
        
        return raw_response, None
        
    except Exception as e:
        return None, f"Error leyendo archivo: {e}"


def test_json_parsing(raw_response, file_name):
    """
    Prueba el parseo del JSON extraído
    """
    result = {
        'file': file_name,
        'raw_length': len(raw_response),
        'has_json': False,
        'json_valid': False,
        'parser_success': False,
        'parser_structured': False,
        'errors': []
    }
    
    # Verificar si contiene JSON
    if '{' in raw_response and '}' in raw_response:
        result['has_json'] = True
        
        # Intentar parseo JSON directo
        try:
            # Buscar el JSON en la respuesta
            json_start = raw_response.find('{')
            if json_start != -1:
                # Encontrar el JSON balanceado
                brace_count = 0
                json_end = json_start
                for i, char in enumerate(raw_response[json_start:], json_start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if brace_count == 0:
                    json_str = raw_response[json_start:json_end]
                    parsed = json.loads(json_str)
                    result['json_valid'] = True
                    result['json_keys'] = list(parsed.keys()) if isinstance(parsed, dict) else []
        except json.JSONDecodeError as e:
            result['errors'].append(f"JSON directo inválido: {e}")
        except Exception as e:
            result['errors'].append(f"Error extrayendo JSON: {e}")
    
    # Probar con el parser mejorado si está disponible
    if PARSER_AVAILABLE:
        try:
            parser = StructuredResponseParser()
            parsed_data, is_structured = parser.parse_response(raw_response)
            
            result['parser_success'] = True
            result['parser_structured'] = is_structured
            result['parser_keys'] = list(parsed_data.keys()) if isinstance(parsed_data, dict) else []
            
            if is_structured:
                result['confidence_score'] = parsed_data.get('confidence', {}).get('score', 'N/A')
                result['sources_count'] = len(parsed_data.get('sources', []))
                result['key_points_count'] = len(parsed_data.get('key_points', []))
            
        except Exception as e:
            result['errors'].append(f"Parser mejorado falló: {e}")
    
    return result


def main():
    """
    Función principal que ejecuta las pruebas
    """
    print("=== BATERÍA DE PRUEBAS DEL PARSER JSON MEJORADO ===")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Buscar archivos de logs
    log_pattern = "logs/llm_response_*.txt"
    log_files = glob.glob(log_pattern)
    
    if not log_files:
        print(f"❌ No se encontraron archivos con el patrón: {log_pattern}")
        return
    
    print(f"📁 Encontrados {len(log_files)} archivos de logs")
    print()
    
    results = []
    successful_parses = 0
    json_valid_count = 0
    parser_structured_count = 0
    
    # Procesar cada archivo
    for i, log_file in enumerate(sorted(log_files), 1):
        file_name = os.path.basename(log_file)
        print(f"[{i:2d}/{len(log_files)}] Procesando: {file_name}")
        
        # Extraer JSON del log
        raw_response, error = extract_json_from_log(log_file)
        
        if error:
            print(f"    ❌ {error}")
            continue
        
        # Probar el parseo
        result = test_json_parsing(raw_response, file_name)
        results.append(result)
        
        # Mostrar resultado
        status_icons = []
        if result['has_json']:
            status_icons.append("📄")
        if result['json_valid']:
            status_icons.append("✅")
            json_valid_count += 1
        if result['parser_success']:
            status_icons.append("🔧")
            successful_parses += 1
        if result['parser_structured']:
            status_icons.append("📊")
            parser_structured_count += 1
        
        status = " ".join(status_icons) if status_icons else "❌"
        print(f"    {status} Longitud: {result['raw_length']} chars")
        
        if result['errors']:
            for error in result['errors']:
                print(f"    ⚠️  {error}")
    
    # Generar reporte final
    print()
    print("=== REPORTE FINAL ===")
    print(f"📊 Archivos procesados: {len(results)}")
    print(f"📄 Con JSON: {sum(1 for r in results if r['has_json'])}")
    print(f"✅ JSON válido directo: {json_valid_count}")
    print(f"🔧 Parser exitoso: {successful_parses}")
    print(f"📊 Parseado estructurado: {parser_structured_count}")
    
    # Calcular estadísticas
    if results:
        success_rate = (successful_parses / len(results)) * 100
        structured_rate = (parser_structured_count / len(results)) * 100
        
        print(f"📈 Tasa de éxito del parser: {success_rate:.1f}%")
        print(f"📈 Tasa de parseo estructurado: {structured_rate:.1f}%")
    
    # Guardar reporte detallado
    report_file = f"parser_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_files': len(results),
                'files_with_json': sum(1 for r in results if r['has_json']),
                'json_valid_direct': json_valid_count,
                'parser_successful': successful_parses,
                'parser_structured': parser_structured_count,
                'success_rate': success_rate if results else 0,
                'structured_rate': structured_rate if results else 0
            },
            'detailed_results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Reporte detallado guardado en: {report_file}")
    
    # Mostrar algunos ejemplos de errores comunes
    error_types = {}
    for result in results:
        for error in result['errors']:
            error_type = error.split(':')[0]
            error_types[error_type] = error_types.get(error_type, 0) + 1
    
    if error_types:
        print()
        print("=== TIPOS DE ERRORES MÁS COMUNES ===")
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {error_type}: {count} veces")


if __name__ == "__main__":
    main()
