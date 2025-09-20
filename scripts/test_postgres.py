# scripts/test_postgres.py
from endocenter.config import settings
from sqlalchemy import create_engine, text

def test_postgres_connection():
    """Test conexión a PostgreSQL"""
    try:
        engine = create_engine(settings.database_url)
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version();"))
            version = result.fetchone()[0]
            print(f"✅ PostgreSQL conectado: {version}")
            
            # Test básico
            conn.execute(text("SELECT 1;"))
            print("✅ Queries funcionando")
            
    except Exception as e:
        print(f"❌ Error conectando a PostgreSQL: {e}")
        print("💡 Asegúrate de que PostgreSQL esté corriendo")

if __name__ == "__main__":
    test_postgres_connection()