"""
EndoCenter MLOps - Database Connection Management
Configuración optimizada de SQLAlchemy para PostgreSQL con connection pooling
"""

from sqlalchemy import create_engine, event, pool
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine
from contextlib import contextmanager
from typing import Generator
import time
from loguru import logger

from endocenter.config import settings


# =============================================================================
# DATABASE ENGINE CONFIGURATION
# =============================================================================

def get_database_url() -> str:
    """
    Construir database URL desde settings
    Soporta tanto DATABASE_URL directo como componentes individuales
    """
    if settings.database_url:
        return settings.database_url
    
    # Construir desde componentes
    return f"postgresql://{settings.db_user}:{settings.db_password}@{settings.db_host}:{settings.db_port}/{settings.db_name}"


# Engine Configuration - Optimizado para producción
engine = create_engine(
    get_database_url(),
    
    # Connection Pool Settings - Optimizado para múltiples workers
    poolclass=pool.QueuePool,           # Pool de conexiones reutilizables
    pool_size=10,                        # 10 conexiones permanentes
    max_overflow=20,                     # +20 conexiones temporales si necesario
    pool_timeout=30,                     # Timeout para obtener conexión del pool
    pool_recycle=3600,                   # Reciclar conexiones cada hora
    pool_pre_ping=True,                  # Verificar conexión antes de usar
    
    # Statement execution settings
    echo=settings.db_echo if hasattr(settings, 'db_echo') else False,  # Log SQL queries en dev
    echo_pool=False,                     # Log pool events (muy verbose)
    
    # Performance settings
    connect_args={
        "connect_timeout": 10,           # Timeout de conexión inicial
        "options": "-c timezone=utc",    # Forzar UTC
    },
    
    # Execution options
    execution_options={
        "isolation_level": "READ COMMITTED"  # Nivel de aislamiento por defecto
    }
)


# Session Factory - Configuración de sesiones
SessionLocal = sessionmaker(
    autocommit=False,      # No auto-commit (control manual de transacciones)
    autoflush=False,       # No auto-flush (mejor performance)
    bind=engine,
    expire_on_commit=True  # Refresh objects después de commit
)


# =============================================================================
# DATABASE EVENTS - Logging y Monitoring
# =============================================================================

@event.listens_for(Engine, "connect")
def receive_connect(dbapi_conn, connection_record):
    """Log cuando se crea una nueva conexión"""
    logger.debug(f"New database connection established: {id(dbapi_conn)}")


@event.listens_for(Engine, "checkout")
def receive_checkout(dbapi_conn, connection_record, connection_proxy):
    """Log cuando se obtiene una conexión del pool"""
    logger.trace(f"Connection checked out from pool: {id(dbapi_conn)}")


@event.listens_for(Engine, "checkin")
def receive_checkin(dbapi_conn, connection_record):
    """Log cuando se devuelve una conexión al pool"""
    logger.trace(f"Connection returned to pool: {id(dbapi_conn)}")


# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

def get_db() -> Generator[Session, None, None]:
    """
    Dependency injection para FastAPI
    Proporciona una sesión de base de datos que se cierra automáticamente
    
    Usage en FastAPI:
        @app.get("/patients")
        def get_patients(db: Session = Depends(get_db)):
            return db.query(Patient).all()
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    Context manager para uso fuera de FastAPI
    
    Usage:
        with get_db_context() as db:
            patient = db.query(Patient).first()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        logger.error(f"Database transaction failed: {e}")
        db.rollback()
        raise
    finally:
        db.close()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def test_connection() -> bool:
    """
    Probar conexión a la base de datos
    Returns True si la conexión es exitosa
    """
    try:
        with engine.connect() as conn:
            result = conn.execute("SELECT 1")
            logger.success("Database connection test successful")
            return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


def get_pool_status() -> dict:
    """
    Obtener estado del connection pool
    Útil para monitoring y debugging
    """
    pool = engine.pool
    return {
        "pool_size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "total_connections": pool.size() + pool.overflow(),
    }


def close_all_connections():
    """
    Cerrar todas las conexiones del pool
    Útil para shutdown graceful
    """
    logger.info("Closing all database connections...")
    engine.dispose()
    logger.success("All database connections closed")


# =============================================================================
# DATABASE INITIALIZATION
# =============================================================================

def init_db():
    """
    Inicializar base de datos
    Crear todas las tablas si no existen
    """
    from endocenter.db.models import Base
    
    logger.info("Initializing database...")
    try:
        Base.metadata.create_all(bind=engine)
        logger.success("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


def drop_db():
    """
    PELIGRO: Eliminar todas las tablas
    Solo para desarrollo/testing
    """
    from endocenter.db.models import Base
    
    if settings.environment == "production":
        raise RuntimeError("Cannot drop database in production!")
    
    logger.warning("Dropping all database tables...")
    Base.metadata.drop_all(bind=engine)
    logger.warning("All tables dropped")


# =============================================================================
# TRANSACTION UTILITIES
# =============================================================================

class TransactionContext:
    """
    Context manager avanzado para transacciones con retry logic
    """
    def __init__(self, db: Session, max_retries: int = 3):
        self.db = db
        self.max_retries = max_retries
        self.retries = 0
    
    def __enter__(self):
        return self.db
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            try:
                self.db.commit()
            except Exception as e:
                self.db.rollback()
                logger.error(f"Commit failed: {e}")
                raise
        else:
            self.db.rollback()
            logger.error(f"Transaction rolled back: {exc_val}")
        return False


def with_retry(func):
    """
    Decorator para reintentar operaciones de base de datos
    """
    def wrapper(*args, **kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Operation failed after {max_retries} attempts: {e}")
                    raise
                logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                time.sleep(0.5 * (attempt + 1))  # Exponential backoff
    return wrapper


# =============================================================================
# HEALTH CHECK
# =============================================================================

def health_check() -> dict:
    """
    Health check completo de la base de datos
    Returns información de estado y performance
    """
    start_time = time.time()
    
    try:
        # Test basic connection
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        
        # Get pool status
        pool_status = get_pool_status()
        
        # Calculate response time
        response_time = (time.time() - start_time) * 1000  # ms
        
        return {
            "status": "healthy",
            "response_time_ms": round(response_time, 2),
            "database": settings.db_name,
            "host": settings.db_host,
            "pool_status": pool_status,
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "database": settings.db_name,
            "host": settings.db_host,
        }


# =============================================================================
# DEMO/TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing EndoCenter Database Configuration...")
    print("=" * 60)
    
    # Test connection
    print("\n1. Testing database connection...")
    if test_connection():
        print("   ✅ Connection successful")
    else:
        print("   ❌ Connection failed")
    
    # Pool status
    print("\n2. Connection pool status:")
    status = get_pool_status()
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    # Health check
    print("\n3. Health check:")
    health = health_check()
    for key, value in health.items():
        print(f"   {key}: {value}")
    
    # Test session
    print("\n4. Testing session management...")
    try:
        with get_db_context() as db:
            from endocenter.db.models import User
            count = db.query(User).count()
            print(f"   ✅ Session working - Users in DB: {count}")
    except Exception as e:
        print(f"   ⚠️ Session test: {e}")
    
    print("\n" + "=" * 60)
    print("Database configuration ready!")