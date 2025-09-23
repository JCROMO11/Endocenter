from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from endocenter.config import settings

# Crear engine
engine = create_engine(settings.database_url, pool_pre_ping=True)

# Crear session factory
SessionLocal = sessionmaker(bind=engine)

# Crear Base para modelos
Base = declarative_base()

# Dependency para FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()