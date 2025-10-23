#!/bin/bash
# Script para verificar dependencias en EC2
# Uso: ./check_dependencies_ec2.sh

echo "🔍 VERIFICACIÓN DE DEPENDENCIAS EN EC2"
echo "======================================"

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para mostrar estado
show_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
    else
        echo -e "${RED}❌ $2${NC}"
    fi
}

# Verificar Python
echo -e "\n${BLUE}🐍 Verificando Python...${NC}"
python3 --version
show_status $? "Python 3 disponible"

# Verificar pip
echo -e "\n${BLUE}📦 Verificando pip...${NC}"
pip3 --version
show_status $? "pip3 disponible"

# Verificar si NLTK está instalado
echo -e "\n${BLUE}📚 Verificando NLTK...${NC}"
python3 -c "import nltk; print(f'NLTK versión: {nltk.__version__}')" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ NLTK está instalado${NC}"
    
    # Verificar datos de NLTK
    echo -e "\n${BLUE}📥 Verificando datos de NLTK...${NC}"
    python3 -c "
import nltk
try:
    from nltk.corpus import stopwords
    en_words = stopwords.words('english')
    es_words = stopwords.words('spanish')
    print(f'✅ Stopwords disponibles - Inglés: {len(en_words)}, Español: {len(es_words)}')
except:
    print('❌ Datos de stopwords no disponibles')
    exit(1)
" 2>/dev/null
    show_status $? "Datos de stopwords disponibles"
else
    echo -e "${RED}❌ NLTK no está instalado${NC}"
fi

# Verificar otras dependencias del requirements.txt
echo -e "\n${BLUE}📋 Verificando dependencias del proyecto...${NC}"

# Lista de dependencias críticas
dependencies=("boto3" "pyyaml" "opensearch-py" "loguru" "pandas" "Pillow" "requests" "regex")

for dep in "${dependencies[@]}"; do
    python3 -c "import $dep" 2>/dev/null
    if [ $? -eq 0 ]; then
        version=$(python3 -c "import $dep; print(getattr($dep, '__version__', 'unknown'))" 2>/dev/null)
        echo -e "${GREEN}✅ $dep ($version)${NC}"
    else
        echo -e "${RED}❌ $dep no disponible${NC}"
    fi
done

# Verificar estructura del proyecto
echo -e "\n${BLUE}📁 Verificando estructura del proyecto...${NC}"

required_dirs=("src" "config" "scripts" "data")
for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo -e "${GREEN}✅ Directorio $dir existe${NC}"
    else
        echo -e "${RED}❌ Directorio $dir no encontrado${NC}"
    fi
done

# Verificar archivos críticos
required_files=("requirements.txt" "config/stop_words_config.yaml" "src/utils/stop_words_manager.py")
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅ Archivo $file existe${NC}"
    else
        echo -e "${RED}❌ Archivo $file no encontrado${NC}"
    fi
done

# Verificar permisos de ejecución
echo -e "\n${BLUE}🔐 Verificando permisos...${NC}"
if [ -x "scripts/install_nltk_ec2.py" ]; then
    echo -e "${GREEN}✅ Script de instalación ejecutable${NC}"
else
    echo -e "${YELLOW}⚠️  Haciendo ejecutable el script de instalación...${NC}"
    chmod +x scripts/install_nltk_ec2.py
    show_status $? "Permisos de ejecución configurados"
fi

# Verificar conectividad (opcional)
echo -e "\n${BLUE}🌐 Verificando conectividad...${NC}"
ping -c 1 pypi.org > /dev/null 2>&1
show_status $? "Conectividad a PyPI disponible"

echo -e "\n${BLUE}📊 RESUMEN DE VERIFICACIÓN${NC}"
echo "================================"
echo "Si ves errores arriba, ejecuta el script de instalación:"
echo -e "${YELLOW}python3 scripts/install_nltk_ec2.py${NC}"
echo ""
echo "Para instalar todas las dependencias:"
echo -e "${YELLOW}pip3 install --user -r requirements.txt${NC}"
