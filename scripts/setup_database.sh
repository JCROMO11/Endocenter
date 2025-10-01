#!/bin/bash
# EndoCenter MLOps - Complete Database Setup

set -e  # Exit on error

echo "🗄️ EndoCenter Database Setup"
echo "=============================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# =============================================================================
# PASO 1: Verificar PostgreSQL
# =============================================================================

echo -e "${YELLOW}📡 Verificando PostgreSQL...${NC}"

if ! command -v psql &> /dev/null; then
    echo -e "${RED}❌ PostgreSQL no encontrado${NC}"
    echo "Instala PostgreSQL o usa Docker:"
    echo "  docker-compose -f infrastructure/docker/docker-compose.dev.yml up -d"
    exit 1
fi

# Test connection
if docker ps | grep -q endocenter_postgres; then
    echo -e "${GREEN}✅ PostgreSQL (Docker) está corriendo${NC}"
elif pg_isready -h localhost -p 5432 > /dev/null 2>&1; then
    echo -e "${GREEN}✅ PostgreSQL (Local) está corriendo${NC}"
else
    echo -e "${RED}❌ PostgreSQL no está corriendo${NC}"
    echo "Inicia PostgreSQL con:"
    echo "  docker-compose -f infrastructure/docker/docker-compose.dev.yml up -d"
    exit 1
fi

# =============================================================================
# PASO 2: Crear Base de Datos
# =============================================================================

echo ""
echo -e "${YELLOW}🏗️ Creando base de datos...${NC}"

# Si está en Docker
if docker ps | grep -q endocenter_postgres; then
    docker exec endocenter_postgres psql -U endocenter_user -d postgres -c "CREATE DATABASE endocenter;" 2>/dev/null || echo "Base de datos ya existe"
    echo -e "${GREEN}✅ Base de datos verificada (Docker)${NC}"
else
    # Si es local
    createdb endocenter 2>/dev/null || echo "Base de datos ya existe"
    echo -e "${GREEN}✅ Base de datos verificada (Local)${NC}"
fi

# =============================================================================
# PASO 3: Verificar conexión con Python
# =============================================================================

echo ""
echo -e "${YELLOW}🐍 Verificando conexión con Python...${NC}"

python << EOF
from endocenter.config import settings
from sqlalchemy import create_engine, text

try:
    engine = create_engine(settings.database_url)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT version();"))
        version = result.fetchone()[0]
        print(f"✅ Conectado a: {version}")
except Exception as e:
    print(f"❌ Error de conexión: {e}")
    exit(1)
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Conexión Python-PostgreSQL verificada${NC}"
else
    echo -e "${RED}❌ Error de conexión${NC}"
    exit 1
fi

# =============================================================================
# PASO 4: Inicializar Alembic (si no existe)
# =============================================================================

echo ""
echo -e "${YELLOW}🔧 Configurando Alembic...${NC}"

if [ ! -d "alembic" ]; then
    echo "Inicializando Alembic..."
    alembic init alembic
    echo -e "${GREEN}✅ Alembic inicializado${NC}"
    
    # Copiar archivos de configuración
    echo "Copiando configuraciones..."
    # Los archivos ya están en los artifacts
    echo -e "${YELLOW}⚠️ Recuerda copiar alembic.ini y alembic/env.py de los artifacts${NC}"
else
    echo -e "${GREEN}✅ Alembic ya está configurado${NC}"
fi

# =============================================================================
# PASO 5: Crear primera migración
# =============================================================================

echo ""
echo -e "${YELLOW}📝 Creando migración inicial...${NC}"

if alembic revision --autogenerate -m "initial_schema"; then
    echo -e "${GREEN}✅ Migración inicial creada${NC}"
else
    echo -e "${YELLOW}⚠️ La migración puede ya existir${NC}"
fi

# =============================================================================
# PASO 6: Aplicar migraciones
# =============================================================================

echo ""
echo -e "${YELLOW}⬆️ Aplicando migraciones...${NC}"

alembic upgrade head

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Migraciones aplicadas exitosamente${NC}"
else
    echo -e "${RED}❌ Error aplicando migraciones${NC}"
    exit 1
fi

# =============================================================================
# PASO 7: Verificar tablas creadas
# =============================================================================

echo ""
echo -e "${YELLOW}🔍 Verificando tablas creadas...${NC}"

python << EOF
from sqlalchemy import create_engine, inspect
from endocenter.config import settings

engine = create_engine(settings.database_url)
inspector = inspect(engine)

tables = inspector.get_table_names()
print(f"\n📊 Tablas creadas ({len(tables)}):")
for table in sorted(tables):
    print(f"  ✓ {table}")
EOF

# =============================================================================
# PASO 8: Seed data (opcional)
# =============================================================================

echo ""
echo -e "${YELLOW}🌱 ¿Quieres cargar datos de ejemplo? (y/n)${NC}"
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "Cargando datos de ejemplo..."
    python << EOF
from endocenter.db.engine import get_db_context
from endocenter.db.models import Patient, Doctor, Appointment
from datetime import datetime, timedelta

with get_db_context() as db:
    # Crear médico de ejemplo
    doctor = Doctor(
        first_name="María",
        last_name="González",
        email="maria.gonzalez@endocenter.com",
        phone="+57 300 123 4567",
        license_number="MED-12345",
        specialty="Endocrinología",
        years_experience=10
    )
    db.add(doctor)
    
    # Crear paciente de ejemplo
    patient = Patient(
        first_name="Juan",
        last_name="Pérez",
        email="juan.perez@email.com",
        phone="+57 300 765 4321",
        date_of_birth=datetime(1985, 5, 15),
        gender="Masculino",
        blood_type="O+"
    )
    db.add(patient)
    
    db.commit()
    
    # Crear cita de ejemplo
    appointment = Appointment(
        patient_id=patient.id,
        doctor_id=doctor.id,
        scheduled_at=datetime.now() + timedelta(days=7),
        duration_minutes=30,
        appointment_type="consulta",
        reason="Control de tiroides",
        status="scheduled"
    )
    db.add(appointment)
    
    db.commit()
    
    print("✅ Datos de ejemplo cargados:")
    print(f"  - 1 Doctor: Dr. {doctor.first_name} {doctor.last_name}")
    print(f"  - 1 Paciente: {patient.first_name} {patient.last_name}")
    print(f"  - 1 Cita programada")
EOF
    
    echo -e "${GREEN}✅ Datos de ejemplo cargados${NC}"
fi

# =============================================================================
# RESUMEN FINAL
# =============================================================================

echo ""
echo "========================================"
echo -e "${GREEN}🎉 Setup completado exitosamente!${NC}"
echo "========================================"
echo ""
echo "📊 Base de datos lista para usar"
echo ""
echo "Comandos útiles:"
echo "  • Ver migraciones:      alembic history"
echo "  • Crear migración:      alembic revision --autogenerate -m 'descripcion'"
echo "  • Aplicar migraciones:  alembic upgrade head"
echo "  • Revertir migración:   alembic downgrade -1"
echo "  • Estado actual:        alembic current"
echo ""
echo "🚀 Inicia el servidor:"
echo "  make run"
echo ""
EOF