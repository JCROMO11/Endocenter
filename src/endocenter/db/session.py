"""
EndoCenter MLOps - Database Session Management
Dependency injection y utilidades para manejo de sesiones SQLAlchemy
"""

from typing import Generator, Optional, Type, TypeVar, Generic, List
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from fastapi import Depends, HTTPException, status
from loguru import logger

from endocenter.db.database import SessionLocal, get_db as _get_db


# =============================================================================
# TYPE VARIABLES FOR GENERIC REPOSITORY
# =============================================================================

ModelType = TypeVar("ModelType")


# =============================================================================
# DEPENDENCY INJECTION FUNCTIONS
# =============================================================================

def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency para obtener sesión de base de datos
    
    Usage:
        @app.get("/patients")
        def get_patients(db: Session = Depends(get_db)):
            return db.query(Patient).all()
    """
    return _get_db()


# =============================================================================
# BASE REPOSITORY PATTERN
# =============================================================================

class BaseRepository(Generic[ModelType]):
    """
    Repositorio base con operaciones CRUD comunes
    Reduce código repetitivo en operaciones de base de datos
    """
    
    def __init__(self, model: Type[ModelType], db: Session):
        self.model = model
        self.db = db
    
    def get(self, id: int) -> Optional[ModelType]:
        """Obtener un registro por ID"""
        try:
            return self.db.query(self.model).filter(self.model.id == id).first()
        except SQLAlchemyError as e:
            logger.error(f"Error getting {self.model.__name__} with id {id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error: {str(e)}"
            )
    
    def get_or_404(self, id: int) -> ModelType:
        """Obtener un registro por ID o lanzar 404"""
        obj = self.get(id)
        if not obj:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"{self.model.__name__} with id {id} not found"
            )
        return obj
    
    def get_all(self, skip: int = 0, limit: int = 100) -> List[ModelType]:
        """Obtener todos los registros con paginación"""
        try:
            return self.db.query(self.model).offset(skip).limit(limit).all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting all {self.model.__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error: {str(e)}"
            )
    
    def count(self) -> int:
        """Contar total de registros"""
        try:
            return self.db.query(self.model).count()
        except SQLAlchemyError as e:
            logger.error(f"Error counting {self.model.__name__}: {e}")
            return 0
    
    def create(self, obj: ModelType) -> ModelType:
        """Crear un nuevo registro"""
        try:
            self.db.add(obj)
            self.db.commit()
            self.db.refresh(obj)
            logger.info(f"Created {self.model.__name__} with id {obj.id}")
            return obj
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error creating {self.model.__name__}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error creating record: {str(e)}"
            )
    
    def update(self, id: int, obj_data: dict) -> Optional[ModelType]:
        """Actualizar un registro existente"""
        try:
            obj = self.get_or_404(id)
            for key, value in obj_data.items():
                if hasattr(obj, key):
                    setattr(obj, key, value)
            
            self.db.commit()
            self.db.refresh(obj)
            logger.info(f"Updated {self.model.__name__} with id {id}")
            return obj
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error updating {self.model.__name__} with id {id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error updating record: {str(e)}"
            )
    
    def delete(self, id: int) -> bool:
        """Eliminar un registro"""
        try:
            obj = self.get_or_404(id)
            self.db.delete(obj)
            self.db.commit()
            logger.info(f"Deleted {self.model.__name__} with id {id}")
            return True
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error deleting {self.model.__name__} with id {id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error deleting record: {str(e)}"
            )
    
    def soft_delete(self, id: int) -> Optional[ModelType]:
        """Soft delete - marcar como eliminado sin borrar"""
        if not hasattr(self.model, 'is_deleted'):
            raise NotImplementedError(
                f"{self.model.__name__} does not support soft delete"
            )
        
        return self.update(id, {"is_deleted": True})


# =============================================================================
# SPECIFIC REPOSITORIES FOR ENDOCENTER
# =============================================================================

class UserRepository(BaseRepository):
    """Repository específico para User con métodos adicionales"""
    
    def get_by_email(self, email: str) -> Optional[ModelType]:
        """Obtener usuario por email"""
        try:
            return self.db.query(self.model).filter(
                self.model.email == email
            ).first()
        except SQLAlchemyError as e:
            logger.error(f"Error getting user by email {email}: {e}")
            return None
    
    def get_by_username(self, username: str) -> Optional[ModelType]:
        """Obtener usuario por username"""
        try:
            return self.db.query(self.model).filter(
                self.model.username == username
            ).first()
        except SQLAlchemyError as e:
            logger.error(f"Error getting user by username {username}: {e}")
            return None
    
    def get_active_users(self, skip: int = 0, limit: int = 100) -> List[ModelType]:
        """Obtener solo usuarios activos"""
        try:
            return self.db.query(self.model).filter(
                self.model.is_active == True,
                self.model.is_deleted == False
            ).offset(skip).limit(limit).all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting active users: {e}")
            return []


class PatientRepository(BaseRepository):
    """Repository específico para Patient"""
    
    def get_by_user_id(self, user_id: int) -> Optional[ModelType]:
        """Obtener paciente por user_id"""
        try:
            return self.db.query(self.model).filter(
                self.model.user_id == user_id
            ).first()
        except SQLAlchemyError as e:
            logger.error(f"Error getting patient by user_id {user_id}: {e}")
            return None
    
    def get_with_appointments(self, patient_id: int) -> Optional[ModelType]:
        """Obtener paciente con sus appointments cargados"""
        from sqlalchemy.orm import joinedload
        try:
            return self.db.query(self.model).options(
                joinedload(self.model.appointments)
            ).filter(self.model.id == patient_id).first()
        except SQLAlchemyError as e:
            logger.error(f"Error getting patient with appointments: {e}")
            return None


class RAGConsultationRepository(BaseRepository):
    """Repository específico para RAGConsultation"""
    
    def get_by_session_id(self, session_id: str) -> Optional[ModelType]:
        """Obtener consulta por session_id"""
        try:
            return self.db.query(self.model).filter(
                self.model.session_id == session_id
            ).first()
        except SQLAlchemyError as e:
            logger.error(f"Error getting consultation by session_id: {e}")
            return None
    
    def get_by_user(self, user_id: int, skip: int = 0, limit: int = 50) -> List[ModelType]:
        """Obtener consultas de un usuario"""
        try:
            return self.db.query(self.model).filter(
                self.model.user_id == user_id
            ).order_by(self.model.started_at.desc()).offset(skip).limit(limit).all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting consultations for user {user_id}: {e}")
            return []
    
    def get_completed(self, skip: int = 0, limit: int = 100) -> List[ModelType]:
        """Obtener consultas completadas"""
        from endocenter.db.models import RAGConsultationStatus
        try:
            return self.db.query(self.model).filter(
                self.model.status == RAGConsultationStatus.COMPLETED
            ).order_by(self.model.completed_at.desc()).offset(skip).limit(limit).all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting completed consultations: {e}")
            return []


# =============================================================================
# REPOSITORY FACTORY
# =============================================================================

def get_repository(model: Type[ModelType], db: Session = Depends(get_db)) -> BaseRepository:
    """
    Factory para crear repositorios específicos según el modelo
    
    Usage:
        @app.get("/patients/{id}")
        def get_patient(
            id: int,
            repo: PatientRepository = Depends(lambda db: get_repository(Patient, db))
        ):
            return repo.get_or_404(id)
    """
    # Map de modelos a repositorios específicos
    from endocenter.db.models import User, Patient, RAGConsultation
    
    repository_map = {
        User: UserRepository,
        Patient: PatientRepository,
        RAGConsultation: RAGConsultationRepository,
    }
    
    # Obtener repository específico o usar el base
    repo_class = repository_map.get(model, BaseRepository)
    return repo_class(model, db)


# =============================================================================
# DEPENDENCY INJECTION HELPERS
# =============================================================================

def get_user_repository(db: Session = Depends(get_db)) -> UserRepository:
    """Dependency para UserRepository"""
    from endocenter.db.models import User
    return UserRepository(User, db)


def get_patient_repository(db: Session = Depends(get_db)) -> PatientRepository:
    """Dependency para PatientRepository"""
    from endocenter.db.models import Patient
    return PatientRepository(Patient, db)


def get_rag_consultation_repository(db: Session = Depends(get_db)) -> RAGConsultationRepository:
    """Dependency para RAGConsultationRepository"""
    from endocenter.db.models import RAGConsultation
    return RAGConsultationRepository(RAGConsultation, db)


# =============================================================================
# TRANSACTION HELPERS
# =============================================================================

class TransactionManager:
    """
    Context manager para manejar transacciones complejas
    
    Usage:
        with TransactionManager(db) as tm:
            user = User(...)
            tm.add(user)
            patient = Patient(user_id=user.id)
            tm.add(patient)
            # Commit automático al salir del context
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.objects = []
    
    def add(self, obj):
        """Agregar objeto a la transacción"""
        self.objects.append(obj)
        self.db.add(obj)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            try:
                self.db.commit()
                for obj in self.objects:
                    self.db.refresh(obj)
                logger.info(f"Transaction committed: {len(self.objects)} objects")
            except SQLAlchemyError as e:
                self.db.rollback()
                logger.error(f"Transaction failed: {e}")
                raise
        else:
            self.db.rollback()
            logger.error(f"Transaction rolled back: {exc_val}")
        
        return False


# =============================================================================
# DEMO/TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing EndoCenter Session Management...")
    print("=" * 60)
    
    from endocenter.db.database import get_db_context
    from endocenter.db.models import User
    
    # Test repository pattern
    print("\n1. Testing User Repository...")
    try:
        with get_db_context() as db:
            user_repo = UserRepository(User, db)
            
            # Count users
            total = user_repo.count()
            print(f"   Total users in DB: {total}")
            
            # Get all users
            users = user_repo.get_all(limit=5)
            print(f"   First {len(users)} users retrieved")
            
            if users:
                # Get specific user
                user = user_repo.get(users[0].id)
                print(f"   Retrieved user: {user.email if user else 'None'}")
        
        print("   ✅ Repository pattern working")
    except Exception as e:
        print(f"   ⚠️ Repository test failed: {e}")
    
    print("\n" + "=" * 60)
    print("Session management ready!")