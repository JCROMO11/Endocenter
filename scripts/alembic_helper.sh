#!/bin/bash
# EndoCenter MLOps - Alembic Helper Scripts
# Facilita el uso de comandos comunes de Alembic

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🗄️ EndoCenter Alembic Helper${NC}"
echo ""

function show_help() {
    echo "Comandos disponibles:"
    echo ""
    echo -e "  ${GREEN}./alembic_helper.sh init${NC}              - Inicializar Alembic (primera vez)"
    echo -e "  ${GREEN}./alembic_helper.sh create [msg]${NC}      - Crear nueva migración autogenerada"
    echo -e "  ${GREEN}./alembic_helper.sh upgrade${NC}           - Aplicar todas las migraciones pendientes"
    echo -e "  ${GREEN}./alembic_helper.sh downgrade${NC}         - Revertir última migración"
    echo -e "  ${GREEN}./alembic_helper.sh current${NC}           - Ver migración actual"
    echo -e "  ${GREEN}./alembic_helper.sh history${NC}           - Ver historial de migraciones"
    echo -e "  ${GREEN}./alembic_helper.sh heads${NC}             - Ver heads de migraciones"
    echo -e "  ${GREEN}./alembic_helper.sh stamp${NC}             - Marcar base de datos en versión actual"
    echo ""
    echo "Ejemplos:"
    echo -e "  ${YELLOW}./alembic_helper.sh create 'add user table'${NC}"
    echo -e "  ${YELLOW}./alembic_helper.sh upgrade${NC}"
    echo ""
}

# Verificar si Alembic está instalado
if ! command -v alembic &> /dev/null; then
    echo -e "${RED}❌ Error: Alembic no está instalado${NC}"
    echo ""
    echo "Instala con:"
    echo "  pip install alembic"
    echo "  # o"
    echo "  pip install -r requirements/dev.txt"
    exit 1
fi

case "$1" in
    init)
        echo -e "${YELLOW}🚀 Inicializando Alembic...${NC}"
        
        if [ -d "alembic" ]; then
            echo -e "${RED}❌ Error: El directorio 'alembic' ya existe${NC}"
            echo "Si quieres reiniciar, primero elimina el directorio:"
            echo "  rm -rf alembic alembic.ini"
            exit 1
        fi
        
        alembic init alembic
        
        echo ""
        echo -e "${GREEN}✅ Alembic inicializado exitosamente!${NC}"
        echo ""
        echo -e "${YELLOW}📋 Próximos pasos:${NC}"
        echo "  1. Configura 'alembic.ini' con tu database URL"
        echo "  2. Configura 'alembic/env.py' para importar tus modelos"
        echo "  3. Crea tu primera migración:"
        echo "     ./alembic_helper.sh create 'initial schema'"
        echo ""
        ;;
    
    create)
        if [ -z "$2" ]; then
            echo -e "${RED}❌ Error: Debes proporcionar un mensaje para la migración${NC}"
            echo ""
            echo "Uso:"
            echo "  ./alembic_helper.sh create 'descripcion_de_la_migracion'"
            echo ""
            echo "Ejemplos:"
            echo "  ./alembic_helper.sh create 'add email field to users'"
            echo "  ./alembic_helper.sh create 'create appointments table'"
            exit 1
        fi
        
        echo -e "${YELLOW}📝 Creando migración: $2${NC}"
        echo ""
        
        alembic revision --autogenerate -m "$2"
        
        echo ""
        echo -e "${GREEN}✅ Migración creada exitosamente!${NC}"
        echo ""
        echo -e "${YELLOW}💡 Recuerda:${NC}"
        echo "  1. Revisa el archivo generado en 'alembic/versions/'"
        echo "  2. Aplica la migración con:"
        echo "     ./alembic_helper.sh upgrade"
        echo ""
        ;;
    
    upgrade)
        echo -e "${YELLOW}⬆️  Aplicando migraciones pendientes...${NC}"
        echo ""
        
        alembic upgrade head
        
        echo ""
        echo -e "${GREEN}✅ Migraciones aplicadas exitosamente!${NC}"
        echo ""
        echo "Ver estado actual:"
        echo "  ./alembic_helper.sh current"
        echo ""
        ;;
    
    downgrade)
        echo -e "${YELLOW}⬇️  Revirtiendo última migración...${NC}"
        echo ""
        echo -e "${RED}⚠️  ADVERTENCIA: Esto revertirá cambios en la base de datos${NC}"
        read -p "¿Estás seguro? (y/n): " confirm
        
        if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
            echo -e "${GREEN}❌ Operación cancelada${NC}"
            exit 0
        fi
        
        alembic downgrade -1
        
        echo ""
        echo -e "${GREEN}✅ Migración revertida exitosamente!${NC}"
        echo ""
        ;;
    
    current)
        echo -e "${YELLOW}📍 Migración actual:${NC}"
        echo ""
        alembic current --verbose
        echo ""
        ;;
    
    history)
        echo -e "${YELLOW}📜 Historial de migraciones:${NC}"
        echo ""
        alembic history --verbose
        echo ""
        ;;
    
    heads)
        echo -e "${YELLOW}🎯 Heads de migraciones:${NC}"
        echo ""
        alembic heads --verbose
        echo ""
        ;;
    
    stamp)
        if [ -z "$2" ]; then
            echo -e "${YELLOW}Marcando base de datos en 'head'...${NC}"
            alembic stamp head
        else
            echo -e "${YELLOW}Marcando base de datos en '$2'...${NC}"
            alembic stamp "$2"
        fi
        
        echo ""
        echo -e "${GREEN}✅ Base de datos marcada exitosamente!${NC}"
        echo ""
        ;;
    
    help|--help|-h)
        show_help
        ;;
    
    "")
        echo -e "${RED}❌ Error: No se proporcionó ningún comando${NC}"
        echo ""
        show_help
        exit 1
        ;;
    
    *)
        echo -e "${RED}❌ Error: Comando no reconocido: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac

exit 0