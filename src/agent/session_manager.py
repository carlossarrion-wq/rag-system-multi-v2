"""
Session Manager - Gestión de sesiones para conversaciones concurrentes

Este módulo proporciona funcionalidades para gestionar sesiones de usuario
independientes, permitiendo que múltiples usuarios chateen concurrentemente
sin que sus conversaciones se entremezclen.
"""

import logging
import uuid
import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import json
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    """Información de una sesión de usuario"""
    session_id: str
    user_id: Optional[str]
    app_name: str
    created_at: datetime
    last_activity: datetime
    metadata: Dict[str, Any]
    
    def is_expired(self, timeout_hours: int = 24) -> bool:
        """Verifica si la sesión ha expirado"""
        return datetime.now() - self.last_activity > timedelta(hours=timeout_hours)
    
    def update_activity(self):
        """Actualiza la última actividad de la sesión"""
        self.last_activity = datetime.now()


class SessionManager:
    """
    Gestor de sesiones para conversaciones concurrentes.
    
    Características:
    - Generación automática de IDs de sesión únicos
    - Gestión de archivos de memoria por sesión
    - Limpieza automática de sesiones expiradas
    - Thread-safe para uso concurrente
    """
    
    def __init__(
        self,
        base_memory_dir: str = "data/memory",
        session_timeout_hours: int = 24,
        cleanup_interval_hours: int = 6
    ):
        """
        Inicializa el gestor de sesiones.
        
        Args:
            base_memory_dir: Directorio base para archivos de memoria
            session_timeout_hours: Horas después de las cuales una sesión expira
            cleanup_interval_hours: Intervalo para limpieza automática
        """
        self.base_memory_dir = Path(base_memory_dir)
        self.session_timeout_hours = session_timeout_hours
        self.cleanup_interval_hours = cleanup_interval_hours
        
        # Crear directorio base si no existe
        self.base_memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Registro de sesiones activas
        self.active_sessions: Dict[str, SessionInfo] = {}
        
        # Lock para operaciones thread-safe
        self._lock = threading.RLock()
        
        # Cargar sesiones existentes
        self._load_existing_sessions()
        
        logger.info(f"SessionManager initialized with {len(self.active_sessions)} existing sessions")
    
    def create_session(
        self,
        app_name: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Crea una nueva sesión de usuario.
        
        Args:
            app_name: Nombre de la aplicación
            user_id: ID del usuario (opcional)
            metadata: Metadatos adicionales de la sesión
            
        Returns:
            ID de la sesión creada
        """
        with self._lock:
            # Generar ID único de sesión
            session_id = self._generate_session_id(app_name, user_id)
            
            # Crear información de sesión
            session_info = SessionInfo(
                session_id=session_id,
                user_id=user_id,
                app_name=app_name,
                created_at=datetime.now(),
                last_activity=datetime.now(),
                metadata=metadata or {}
            )
            
            # Registrar sesión
            self.active_sessions[session_id] = session_info
            
            # Crear directorio de sesión
            session_dir = self._get_session_directory(session_id)
            session_dir.mkdir(parents=True, exist_ok=True)
            
            # Guardar información de sesión
            self._save_session_info(session_info)
            
            logger.info(f"Created new session: {session_id} for app: {app_name}, user: {user_id}")
            return session_id
    
    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """
        Obtiene información de una sesión.
        
        Args:
            session_id: ID de la sesión
            
        Returns:
            Información de la sesión o None si no existe
        """
        with self._lock:
            session_info = self.active_sessions.get(session_id)
            
            if session_info:
                # Verificar si la sesión ha expirado
                if session_info.is_expired(self.session_timeout_hours):
                    logger.warning(f"Session {session_id} has expired, removing")
                    self.remove_session(session_id)
                    return None
                
                # Actualizar actividad
                session_info.update_activity()
                self._save_session_info(session_info)
            
            return session_info
    
    def get_memory_file_path(self, session_id: str) -> str:
        """
        Obtiene la ruta del archivo de memoria para una sesión.
        
        Args:
            session_id: ID de la sesión
            
        Returns:
            Ruta del archivo de memoria
        """
        session_dir = self._get_session_directory(session_id)
        return str(session_dir / "conversation_memory.json")
    
    def list_user_sessions(self, user_id: str, app_name: Optional[str] = None) -> List[SessionInfo]:
        """
        Lista las sesiones de un usuario.
        
        Args:
            user_id: ID del usuario
            app_name: Filtrar por aplicación (opcional)
            
        Returns:
            Lista de sesiones del usuario
        """
        with self._lock:
            user_sessions = []
            
            for session_info in self.active_sessions.values():
                if session_info.user_id == user_id:
                    if app_name is None or session_info.app_name == app_name:
                        if not session_info.is_expired(self.session_timeout_hours):
                            user_sessions.append(session_info)
            
            return sorted(user_sessions, key=lambda s: s.last_activity, reverse=True)
    
    def remove_session(self, session_id: str) -> bool:
        """
        Elimina una sesión y sus archivos asociados.
        
        Args:
            session_id: ID de la sesión
            
        Returns:
            True si la sesión fue eliminada, False si no existía
        """
        with self._lock:
            if session_id not in self.active_sessions:
                return False
            
            # Eliminar del registro
            del self.active_sessions[session_id]
            
            # Eliminar archivos de la sesión
            session_dir = self._get_session_directory(session_id)
            if session_dir.exists():
                try:
                    # Eliminar archivos de memoria
                    for file_path in session_dir.glob("*.json"):
                        file_path.unlink()
                    
                    # Eliminar directorio si está vacío
                    if not any(session_dir.iterdir()):
                        session_dir.rmdir()
                    
                    logger.info(f"Removed session files for: {session_id}")
                except Exception as e:
                    logger.error(f"Error removing session files for {session_id}: {e}")
            
            logger.info(f"Session removed: {session_id}")
            return True
    
    def cleanup_expired_sessions(self) -> int:
        """
        Limpia sesiones expiradas.
        
        Returns:
            Número de sesiones eliminadas
        """
        with self._lock:
            expired_sessions = []
            
            for session_id, session_info in self.active_sessions.items():
                if session_info.is_expired(self.session_timeout_hours):
                    expired_sessions.append(session_id)
            
            # Eliminar sesiones expiradas
            removed_count = 0
            for session_id in expired_sessions:
                if self.remove_session(session_id):
                    removed_count += 1
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} expired sessions")
            
            return removed_count
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de las sesiones.
        
        Returns:
            Diccionario con estadísticas
        """
        with self._lock:
            stats = {
                'total_sessions': len(self.active_sessions),
                'sessions_by_app': {},
                'sessions_by_user': {},
                'oldest_session': None,
                'newest_session': None
            }
            
            if self.active_sessions:
                # Estadísticas por aplicación
                for session_info in self.active_sessions.values():
                    app_name = session_info.app_name
                    stats['sessions_by_app'][app_name] = stats['sessions_by_app'].get(app_name, 0) + 1
                    
                    if session_info.user_id:
                        user_id = session_info.user_id
                        stats['sessions_by_user'][user_id] = stats['sessions_by_user'].get(user_id, 0) + 1
                
                # Sesión más antigua y más nueva
                sessions_by_time = sorted(self.active_sessions.values(), key=lambda s: s.created_at)
                stats['oldest_session'] = sessions_by_time[0].created_at.isoformat()
                stats['newest_session'] = sessions_by_time[-1].created_at.isoformat()
            
            return stats
    
    @contextmanager
    def session_context(self, session_id: str):
        """
        Context manager para operaciones con sesión.
        
        Args:
            session_id: ID de la sesión
            
        Yields:
            SessionInfo si la sesión existe y es válida
            
        Raises:
            ValueError: Si la sesión no existe o ha expirado
        """
        session_info = self.get_session(session_id)
        if not session_info:
            raise ValueError(f"Session {session_id} not found or expired")
        
        try:
            yield session_info
        finally:
            # Actualizar actividad al finalizar
            session_info.update_activity()
            self._save_session_info(session_info)
    
    def _generate_session_id(self, app_name: str, user_id: Optional[str] = None) -> str:
        """Genera un ID único de sesión"""
        # Usar timestamp + UUID + hash para garantizar unicidad
        timestamp = datetime.now().isoformat()
        unique_id = str(uuid.uuid4())
        
        # Incluir user_id si está disponible
        base_string = f"{app_name}_{timestamp}_{unique_id}"
        if user_id:
            base_string = f"{user_id}_{base_string}"
        
        # Crear hash corto pero único
        session_hash = hashlib.sha256(base_string.encode()).hexdigest()[:12]
        
        return f"session_{app_name}_{session_hash}"
    
    def _get_session_directory(self, session_id: str) -> Path:
        """Obtiene el directorio de una sesión"""
        return self.base_memory_dir / "sessions" / session_id
    
    def _save_session_info(self, session_info: SessionInfo) -> None:
        """Guarda información de sesión en archivo"""
        try:
            session_dir = self._get_session_directory(session_info.session_id)
            session_dir.mkdir(parents=True, exist_ok=True)
            
            info_file = session_dir / "session_info.json"
            
            session_data = {
                'session_id': session_info.session_id,
                'user_id': session_info.user_id,
                'app_name': session_info.app_name,
                'created_at': session_info.created_at.isoformat(),
                'last_activity': session_info.last_activity.isoformat(),
                'metadata': session_info.metadata
            }
            
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving session info for {session_info.session_id}: {e}")
    
    def _load_existing_sessions(self) -> None:
        """Carga sesiones existentes desde archivos"""
        sessions_dir = self.base_memory_dir / "sessions"
        if not sessions_dir.exists():
            return
        
        loaded_count = 0
        
        for session_dir in sessions_dir.iterdir():
            if not session_dir.is_dir():
                continue
            
            info_file = session_dir / "session_info.json"
            if not info_file.exists():
                continue
            
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                
                session_info = SessionInfo(
                    session_id=session_data['session_id'],
                    user_id=session_data.get('user_id'),
                    app_name=session_data['app_name'],
                    created_at=datetime.fromisoformat(session_data['created_at']),
                    last_activity=datetime.fromisoformat(session_data['last_activity']),
                    metadata=session_data.get('metadata', {})
                )
                
                # Solo cargar si no ha expirado
                if not session_info.is_expired(self.session_timeout_hours):
                    self.active_sessions[session_info.session_id] = session_info
                    loaded_count += 1
                else:
                    # Eliminar sesión expirada
                    logger.info(f"Removing expired session during load: {session_info.session_id}")
                    self.remove_session(session_info.session_id)
                
            except Exception as e:
                logger.error(f"Error loading session from {session_dir}: {e}")
        
        if loaded_count > 0:
            logger.info(f"Loaded {loaded_count} existing sessions")


# Instancia global del gestor de sesiones (singleton)
_session_manager_instance: Optional[SessionManager] = None
_session_manager_lock = threading.Lock()


def get_session_manager() -> SessionManager:
    """
    Obtiene la instancia global del gestor de sesiones (singleton).
    
    Returns:
        Instancia del SessionManager
    """
    global _session_manager_instance
    
    if _session_manager_instance is None:
        with _session_manager_lock:
            if _session_manager_instance is None:
                _session_manager_instance = SessionManager()
    
    return _session_manager_instance


def create_user_session(
    app_name: str,
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Función de conveniencia para crear una sesión de usuario.
    
    Args:
        app_name: Nombre de la aplicación
        user_id: ID del usuario (opcional)
        metadata: Metadatos adicionales
        
    Returns:
        ID de la sesión creada
    """
    session_manager = get_session_manager()
    return session_manager.create_session(app_name, user_id, metadata)


def get_memory_file_for_session(session_id: str) -> str:
    """
    Función de conveniencia para obtener el archivo de memoria de una sesión.
    
    Args:
        session_id: ID de la sesión
        
    Returns:
        Ruta del archivo de memoria
    """
    session_manager = get_session_manager()
    return session_manager.get_memory_file_path(session_id)
