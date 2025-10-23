#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para instalar y configurar NLTK en EC2
Este script verifica e instala NLTK con los datos necesarios para stop words
"""

import sys
import subprocess
import os
from pathlib import Path

def run_command(command, description):
    """Ejecuta un comando y maneja errores"""
    print(f"\n🔄 {description}...")
    print(f"Ejecutando: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        print(f"✅ {description} - Exitoso")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Error")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Verifica la versión de Python"""
    print(f"🐍 Python version: {sys.version}")
    if sys.version_info < (3, 6):
        print("❌ Se requiere Python 3.6 o superior")
        return False
    return True

def check_nltk_installation():
    """Verifica si NLTK está instalado"""
    try:
        import nltk
        print(f"✅ NLTK ya está instalado - Versión: {nltk.__version__}")
        return True
    except ImportError:
        print("❌ NLTK no está instalado")
        return False

def install_nltk():
    """Instala NLTK usando pip"""
    commands = [
        "pip3 install --user nltk",
        "python3 -m pip install --user nltk"
    ]
    
    for cmd in commands:
        if run_command(cmd, f"Instalando NLTK con: {cmd}"):
            return True
    
    print("❌ No se pudo instalar NLTK con ningún método")
    return False

def download_nltk_data():
    """Descarga los datos necesarios de NLTK"""
    print("\n🔄 Descargando datos de NLTK...")
    
    try:
        import nltk
        
        # Crear directorio para datos de NLTK si no existe
        nltk_data_dir = os.path.expanduser('~/nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        
        # Descargar stopwords
        print("📥 Descargando stopwords...")
        nltk.download('stopwords', download_dir=nltk_data_dir, quiet=False)
        
        # Verificar que se descargaron correctamente
        from nltk.corpus import stopwords
        
        # Probar idiomas
        english_words = set(stopwords.words('english'))
        spanish_words = set(stopwords.words('spanish'))
        
        print(f"✅ Stopwords descargadas correctamente:")
        print(f"   - Inglés: {len(english_words)} palabras")
        print(f"   - Español: {len(spanish_words)} palabras")
        
        return True
        
    except Exception as e:
        print(f"❌ Error descargando datos de NLTK: {e}")
        return False

def test_stop_words_manager():
    """Prueba el stop words manager del proyecto"""
    print("\n🧪 Probando Stop Words Manager del proyecto...")
    
    try:
        # Agregar el directorio del proyecto al path
        project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(project_root))
        
        from src.utils.stop_words_manager import StopWordsManager
        
        # Crear instancia del manager
        manager = StopWordsManager()
        
        # Obtener estadísticas
        stats = manager.get_stats()
        print("📊 Estadísticas del Stop Words Manager:")
        for key, value in stats.items():
            print(f"   - {key}: {value}")
        
        # Probar extracción de términos clave
        test_text = """
        This is a test document about SAP HANA database system.
        The system provides advanced analytics and real-time processing capabilities.
        Users can configure the database settings through various configuration files.
        """
        
        key_terms = manager.extract_key_terms(test_text, application='sap')
        print(f"\n🔑 Términos clave extraídos: {key_terms}")
        
        # Probar stop words para diferentes configuraciones
        en_stop_words = manager.get_stop_words(['english'])
        es_stop_words = manager.get_stop_words(['spanish'])
        combined_stop_words = manager.get_stop_words(['english', 'spanish'], 'sap')
        
        print(f"\n📝 Stop words configuradas:")
        print(f"   - Solo inglés: {len(en_stop_words)} palabras")
        print(f"   - Solo español: {len(es_stop_words)} palabras")
        print(f"   - Combinadas (EN+ES+SAP): {len(combined_stop_words)} palabras")
        
        print("✅ Stop Words Manager funciona correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error probando Stop Words Manager: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal"""
    print("🚀 INSTALACIÓN Y CONFIGURACIÓN DE NLTK EN EC2")
    print("=" * 60)
    
    # Verificar versión de Python
    if not check_python_version():
        sys.exit(1)
    
    # Verificar si NLTK ya está instalado
    nltk_installed = check_nltk_installation()
    
    # Instalar NLTK si no está instalado
    if not nltk_installed:
        print("\n📦 Instalando NLTK...")
        if not install_nltk():
            print("❌ No se pudo instalar NLTK")
            sys.exit(1)
        
        # Verificar instalación
        if not check_nltk_installation():
            print("❌ NLTK no se instaló correctamente")
            sys.exit(1)
    
    # Descargar datos de NLTK
    if not download_nltk_data():
        print("❌ No se pudieron descargar los datos de NLTK")
        sys.exit(1)
    
    # Probar el stop words manager del proyecto
    if not test_stop_words_manager():
        print("❌ El Stop Words Manager no funciona correctamente")
        sys.exit(1)
    
    print("\n🎉 ¡INSTALACIÓN COMPLETADA EXITOSAMENTE!")
    print("✅ NLTK está instalado y configurado")
    print("✅ Stop words descargadas")
    print("✅ Stop Words Manager del proyecto funciona")
    print("\nEl sistema RAG ahora puede usar NLTK para mejorar el filtrado de stop words.")

if __name__ == "__main__":
    main()
