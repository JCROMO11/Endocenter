"""
Alembic Environment Configuration for EndoCenter MLOps
Optimizado para PostgreSQL con SQLAlchemy 2.0
"""

from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
import os

# Importar la configuración de EndoCenter
import sys
from pathlib import Path

# Añadir el directorio src al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# IMPORTANTE: Cargar variables de entorno desde .env
from dotenv import load_dotenv
load_dotenv()

from endocenter.config import settings
from endocenter.db.models import Base  # Aquí importarás tus modelos

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def get_url():
    """Obtener URL de base de datos desde settings o variable de entorno"""
    # Primero intentar desde variable de entorno directamente
    database_url = os.getenv("DATABASE_URL")
    
    if database_url:
        print(f"✅ Using DATABASE_URL from environment")
        return database_url
    
    # Si no, usar settings
    if hasattr(settings, 'database_url') and settings.database_url:
        print(f"✅ Using database_url from settings")
        return settings.database_url
    
    # Si no, construir desde componentes
    db_user = os.getenv("DB_USER", settings.db_user)
    db_password = os.getenv("DB_PASSWORD", settings.db_password)
    db_host = os.getenv("DB_HOST", settings.db_host)
    db_port = os.getenv("DB_PORT", settings.db_port)
    db_name = os.getenv("DB_NAME", settings.db_name)
    
    url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    print(f"✅ Constructed database URL: postgresql://{db_user}:***@{db_host}:{db_port}/{db_name}")
    return url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    configuration = config.get_section(config.config_ini_section)
    
    # IMPORTANTE: Sobrescribir la URL con la correcta
    configuration["sqlalchemy.url"] = get_url()
    
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()