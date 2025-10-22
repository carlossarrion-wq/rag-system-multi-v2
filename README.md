# RAG System Multi-Application

## 📋 Descripción

Sistema RAG (Retrieval-Augmented Generation) multi-aplicación que soporta múltiples sistemas empresariales con búsqueda híbrida, chunking semántico y capacidades multimodales.

## 🏗️ Arquitectura

### Aplicaciones Soportadas
- **GADEA**: Sistema de gestión empresarial GADEA
- **PDS**: Plataforma Digital de Servicios UFD
- **DARWIN**: Sistema de gestión empresarial DARWIN
- **SAP**: Sistema SAP empresarial

### Componentes Principales

#### 🔍 Retrieval (Búsqueda)
- **Hybrid Search**: Combinación de búsqueda vectorial y por palabras clave
- **RRF (Reciprocal Rank Fusion)**: Algoritmo de fusión de rankings
- **Specialized Retrievers**: Recuperadores especializados por tipo de contenido

#### 📚 Indexing (Indexación)
- **OpenSearch Integration**: Integración con Amazon OpenSearch
- **Multi-App Indexer**: Indexador multi-aplicación
- **Multimodal Support**: Soporte para texto e imágenes

#### 📄 Ingestion (Ingesta)
- **Document Loader**: Cargador de documentos múltiples formatos
- **Supported Formats**: PDF, DOCX, XLSX, TXT, PNG, JPG, XML

#### 🤖 Generation (Generación)
- **LLM Client**: Cliente para modelos de lenguaje (Claude 3 Haiku)
- **Citation Manager**: Gestor de citas y referencias
- **Structured Response**: Respuestas estructuradas con esquemas

#### 🧠 Agent (Agente)
- **Conversational Agent**: Agente conversacional avanzado
- **Memory System**: Sistema de memoria conversacional
- **Context Enhancement**: Mejora de contexto documental

## 🛠️ Tecnologías

- **Python 3.8+**
- **Amazon Bedrock**: Modelos de IA (Claude 3, Titan Embeddings)
- **Amazon OpenSearch**: Motor de búsqueda y vectores
- **PostgreSQL**: Base de datos relacional
- **AWS S3**: Almacenamiento de documentos

## 📁 Estructura del Proyecto

```
RAG_SYSTEM_MULTI/
├── src/
│   ├── agent/              # Agente conversacional
│   ├── generation/         # Generación de respuestas
│   ├── indexing/          # Indexación de documentos
│   ├── ingestion/         # Ingesta de documentos
│   ├── retrieval/         # Búsqueda y recuperación
│   └── utils/             # Utilidades comunes
├── config/
│   ├── multi_app_config.yaml    # Configuración principal
│   └── system_prompts/          # Prompts del sistema
├── scripts/               # Scripts de utilidad
├── data/                 # Datos y memoria
└── logs/                 # Logs del sistema
```

## ⚙️ Configuración

### Variables de Entorno
```bash
# AWS Configuration
AWS_REGION=eu-west-1
AWS_PROFILE=default

# OpenSearch
OPENSEARCH_ENDPOINT=vpc-rag-opensearch-clean-xxx.eu-west-1.es.amazonaws.com

# PostgreSQL
POSTGRES_HOST=rag-postgres.xxx.eu-west-1.rds.amazonaws.com
POSTGRES_PORT=5432
POSTGRES_DB=ragdb
POSTGRES_USER=raguser
```

### Configuración por Aplicación
Cada aplicación tiene su propia configuración en `config/multi_app_config.yaml`:
- Bucket S3 dedicado
- Índice OpenSearch específico
- Configuración RAG personalizada
- System prompt especializado

## 🚀 Uso

### Ingesta de Documentos
```bash
python3 scripts/multi_app_aws_ingestion_manager_with_summarization.py --app gadea
```

### Chat Interactivo
```bash
python3 scripts/multi_app_chat.py --app gadea
```

## 📊 Mejoras Planificadas

### 🥇 Chunking Semántico + Metadata (30% mejora esperada)
- ✅ Overlap configurado: 150-225 tokens
- ✅ Metadata básica implementada
- 🔄 **EN DESARROLLO**: Chunking semántico inteligente
- 🔄 **EN DESARROLLO**: Preservación de tablas completas
- 🔄 **EN DESARROLLO**: Metadata enriquecida (`contains_codes`, `content_type`, `module`)

### 🥈 Hybrid Search Optimizado (25% mejora esperada)
- ✅ RRF implementado
- ✅ Búsqueda híbrida básica
- 🔄 **EN DESARROLLO**: Pesos 50% keyword / 50% vector
- 🔄 **EN DESARROLLO**: Re-ranking por términos técnicos
- 🔄 **EN DESARROLLO**: Boost para códigos exactos (AC01, Z001, etc.)

## 🔧 Instalación

```bash
# Clonar repositorio
git clone https://github.com/csarrion/rag-system-multi.git
cd rag-system-multi

# Instalar dependencias
pip3 install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales
```

## 📝 Logs y Monitoreo

- **Logs del sistema**: `logs/`
- **Memoria conversacional**: `data/memory/`
- **Versiones de documentos**: `data/s3_document_versions*.json`

## 🤝 Contribución

1. Fork del proyecto
2. Crear rama feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit cambios (`git commit -am 'Añadir nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 🆘 Soporte

Para soporte técnico o preguntas:
- Crear un issue en GitHub
- Contactar al equipo de desarrollo

---

**Estado del Proyecto**: 🔄 En desarrollo activo
**Última actualización**: Octubre 2024
