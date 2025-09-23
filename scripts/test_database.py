# test_database.py (en la raíz de tu proyecto)
from endocenter.db.database import engine, SessionLocal, Base, get_db
from sqlalchemy import text

def test_database():
    print("🧪 Probando database.py...")
    
    # Test 1: Engine
    print(f"✅ Engine creado: {engine}")
    
    # Test 2: Conexión
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print(f"✅ Conexión exitosa: {result.fetchone()}")
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
    
    # Test 3: SessionLocal
    print(f"✅ SessionLocal creado: {SessionLocal}")
    
    # Test 4: Base
    print(f"✅ Base creado: {Base}")
    
    # Test 5: get_db generator
    db_gen = get_db()
    print(f"✅ get_db generator: {db_gen}")
    
    print("🎉 database.py funciona correctamente!")

if __name__ == "__main__":
    test_database()