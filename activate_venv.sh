#!/bin/bash
# Script para activar el entorno virtual

echo "Activando entorno virtual..."
source venv/bin/activate

echo ""
echo "✓ Entorno virtual activado"
echo "Python version: $(python3 --version)"
echo "Pip version: $(pip3 --version)"
echo ""
echo "Para desactivar el entorno virtual, ejecuta: deactivate"
