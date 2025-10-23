# Guía de Instalación de NLTK en EC2

Esta guía te ayudará a instalar y configurar NLTK en tu instancia EC2 para mejorar el sistema de stop words del proyecto RAG.

## 📋 Información de Conexión

**Servidor EC2:**
- **IP:** 52.18.245.120
- **Usuario:** ec2-user
- **Clave SSH:** ~/.ssh/ec2_new_key
- **Directorio del proyecto:** /home/ec2-user/RAG_SYSTEM_MULTI_v3

## 🚀 Pasos de Instalación

### 1. Conectar a EC2

```bash
# Desde tu máquina local
ssh -i ~/.ssh/ec2_new_key ec2-user@52.18.245.120
```

### 2. Navegar al Directorio del Proyecto

```bash
cd /home/ec2-user/RAG_SYSTEM_MULTI_v3
```

### 3. Verificar Estado Actual

```bash
# Hacer ejecutable el script de verificación
chmod +x scripts/check_dependencies_ec2.sh

# Ejecutar verificación
./scripts/check_dependencies_ec2.sh
```

### 4. Instalar NLTK (si es necesario)

```bash
# Hacer ejecutable el script de instalación
chmod +x scripts/install_nltk_ec2.py

# Ejecutar instalación
python3 scripts/install_nltk_ec2.py
```

### 5. Instalar Todas las Dependencias

```bash
# Instalar dependencias del proyecto
pip3 install --user -r requirements.txt
```

## 🔧 Comandos de Verificación Manual

### Verificar Python y pip
```bash
python3 --version
pip3 --version
```

### Verificar NLTK
```bash
python3 -c "import nltk; print(f'NLTK versión: {nltk.__version__}')"
```

### Verificar Stop Words
```bash
python3 -c "
from nltk.corpus import stopwords
print(f'Inglés: {len(stopwords.words(\"english\"))} palabras')
print(f'Español: {len(stopwords.words(\"spanish\"))} palabras')
"
```

### Probar Stop Words Manager del Proyecto
```bash
python3 -c "
from src.utils.stop_words_manager import StopWordsManager
manager = StopWordsManager()
stats = manager.get_stats()
print('Stop Words Manager funcionando:', stats['config_loaded'])
print('NLTK disponible:', stats['nltk_available'])
"
```

## 🛠️ Solución de Problemas

### Si NLTK no se instala con pip3
```bash
# Intentar con python3 -m pip
python3 -m pip install --user nltk

# O instalar en el directorio local
pip3 install --user --upgrade nltk
```

### Si faltan datos de NLTK
```bash
python3 -c "
import nltk
nltk.download('stopwords')
"
```

### Si hay problemas de permisos
```bash
# Verificar permisos del directorio home
ls -la ~/
mkdir -p ~/.local/lib/python3.*/site-packages
```

### Si falta alguna dependencia
```bash
# Instalar dependencias una por una
pip3 install --user boto3
pip3 install --user pyyaml
pip3 install --user opensearch-py
pip3 install --user loguru
pip3 install --user pandas
pip3 install --user Pillow
pip3 install --user requests
pip3 install --user regex
```

## 📊 Verificación Final

Después de la instalación, ejecuta:

```bash
# Verificar que todo funciona
python3 scripts/install_nltk_ec2.py

# Debería mostrar:
# ✅ NLTK está instalado y configurado
# ✅ Stop words descargadas
# ✅ Stop Words Manager del proyecto funciona
```

## 🎯 Beneficios Esperados

Una vez instalado NLTK, el sistema tendrá:

- **486+ stop words** (inglés + español + técnicas + específicas por aplicación)
- **Mejor calidad de términos clave** en resúmenes
- **Filtrado más preciso** de palabras irrelevantes
- **Integración automática** con el sistema existente

## 📝 Notas Importantes

1. **Fallback automático:** Si NLTK no está disponible, el sistema usa las stop words configuradas en `config/stop_words_config.yaml`

2. **Sin interrupciones:** La instalación no afecta el funcionamiento actual del sistema

3. **Mejora gradual:** Los beneficios se verán en nuevos resúmenes generados después de la instalación

## 🔄 Comandos de Conexión Rápida

```bash
# Comando completo para conectar y verificar
ssh -i ~/.ssh/ec2_new_key ec2-user@52.18.245.120 "cd /home/ec2-user/RAG_SYSTEM_MULTI_v3 && ./scripts/check_dependencies_ec2.sh"

# Comando completo para conectar e instalar
ssh -i ~/.ssh/ec2_new_key ec2-user@52.18.245.120 "cd /home/ec2-user/RAG_SYSTEM_MULTI_v3 && python3 scripts/install_nltk_ec2.py"
```

## ✅ Lista de Verificación

- [ ] Conectar a EC2
- [ ] Navegar al directorio del proyecto
- [ ] Ejecutar script de verificación
- [ ] Instalar NLTK si es necesario
- [ ] Verificar funcionamiento
- [ ] Probar Stop Words Manager
- [ ] Confirmar mejoras en el sistema

¿Necesitas ayuda con algún paso específico?
