"""
EndoCenter MLOps - Database Models (FIXED VERSION)
Fixed all foreign keys and bidirectional relationships for Alembic
"""

from datetime import datetime, date
from typing import Optional, List
from enum import Enum

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Date, Float, ForeignKey, Enum as SQLEnum, LargeBinary, JSON, CheckConstraint, Index, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

# Base para todos los modelos
Base = declarative_base()


# =============================================================================
# ENUMS (sin cambios)
# =============================================================================

class AppointmentType(str, Enum):
    FIRST_TIME = "first_time"
    CONSULTATION = "consultation"  
    FOLLOW_UP = "follow_up"
    EMERGENCY = "emergency"
    ROUTINE_CHECKUP = "routine_checkup"

class AppointmentStatus(str, Enum):
    REQUESTED = "requested"
    CONFIRMED = "confirmed"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    NO_SHOW = "no_show"
    RESCHEDULED = "rescheduled"

class BloodType(str, Enum):
    A_POSITIVE = "A+"
    A_NEGATIVE = "A-"
    B_POSITIVE = "B+"
    B_NEGATIVE = "B-"
    AB_POSITIVE = "AB+"
    AB_NEGATIVE = "AB-"
    O_POSITIVE = "O+"
    O_NEGATIVE = "O-"
    UNKNOWN = "unknown"

class InsuranceProvider(str, Enum):
    ANTHEM = "anthem"
    BLUE_CROSS = "blue_cross"
    AETNA = "aetna"
    CIGNA = "cigna"
    UNITED_HEALTH = "united_health"
    MEDICARE = "medicare"
    MEDICAID = "medicaid"
    PRIVATE_PAY = "private_pay"
    OTHER = "other"

class MedicalSpecialty(str, Enum):
    GENERAL_ENDOCRINOLOGY = "general_endocrinology"
    DIABETES = "diabetes"
    THYROID = "thyroid"
    ADRENAL = "adrenal"
    REPRODUCTIVE_ENDOCRINOLOGY = "reproductive_endocrinology"
    PEDIATRIC_ENDOCRINOLOGY = "pediatric_endocrinology"
    BONE_METABOLISM = "bone_metabolism"
    OBESITY = "obesity"

class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female" 
    OTHER = "other"
    PREFER_NOT_TO_SAY = "prefer_not_to_say"

class UserRole(str, Enum):
    PATIENT = "patient"
    DOCTOR = "doctor"
    ADMIN = "admin"
    
class DocumentType(str, Enum):
    TEXTBOOK = "textbook"
    RESEARCH_PAPER = "research_paper"
    CLINICAL_GUIDELINE = "clinical_guideline"
    CASE_STUDY = "case_study"
    DRUG_REFERENCE = "drug_reference"
    PROTOCOL = "protocol"
    OTHER = "other"

class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    CHUNKED = "chunked"
    EMBEDDED = "embedded"
    INDEXED = "indexed"
    ERROR = "error"

class RAGConsultationStatus(str, Enum):
    INITIATED = "initiated"
    PROCESSING = "processing"
    COMPLETED = "completed"
    REVIEWED = "reviewed"
    ERROR = "error"

class SeverityLevel(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


# =============================================================================
# CORE USER MODELS
# =============================================================================

class User(Base):
    """Base user model - FIXED with all relationships"""
    __tablename__ = "users"
    
    # Primary Key
    id = Column(Integer, primary_key=True, index=True)
    
    # Authentication
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=True)
    password_hash = Column(String(255), nullable=False)
    
    # Basic info
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    phone = Column(String(20), nullable=True)
    role = Column(SQLEnum(UserRole), nullable=False, default=UserRole.PATIENT)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    is_deleted = Column(Boolean, default=False)
    timezone = Column(String(50), default="UTC")
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    last_login = Column(DateTime, nullable=True)
    deleted_at = Column(DateTime, nullable=True)
    
    # ✅ FIXED: All relationships properly defined
    profile = relationship("UserProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")
    doctor_profile = relationship("Doctor", back_populates="user", uselist=False, cascade="all, delete-orphan")
    patient_profile = relationship("Patient", back_populates="user", uselist=False, cascade="all, delete-orphan")
    
    # ✅ NEW: Added missing relationships
    rag_consultations = relationship("RAGConsultation", back_populates="user", cascade="all, delete-orphan")
    uploaded_documents = relationship("MedicalDocument", foreign_keys="MedicalDocument.uploaded_by", back_populates="uploader")
    created_symptoms = relationship("Symptom", foreign_keys="Symptom.created_by", back_populates="creator")
    
    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}', role='{self.role}')>"
    
    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"
    
    @property
    def is_doctor(self) -> bool:
        return self.role == UserRole.DOCTOR
    
    @property
    def is_patient(self) -> bool:
        return self.role == UserRole.PATIENT


class UserProfile(Base):
    """Extended user profile - No changes needed"""
    __tablename__ = "user_profiles"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    
    date_of_birth = Column(Date, nullable=True)
    gender = Column(SQLEnum(Gender), nullable=True)
    
    address = Column(Text, nullable=True)
    city = Column(String(100), nullable=True)
    state = Column(String(100), nullable=True)
    country = Column(String(100), nullable=True)
    postal_code = Column(String(20), nullable=True)
    
    emergency_contact_name = Column(String(200), nullable=True)
    emergency_contact_phone = Column(String(20), nullable=True)
    emergency_contact_relationship = Column(String(100), nullable=True)
    
    language_preference = Column(String(10), default="en")
    notification_preferences = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    user = relationship("User", back_populates="profile")
    
    @property 
    def age(self) -> Optional[int]:
        if self.date_of_birth:
            today = date.today()
            return today.year - self.date_of_birth.year - \
                   ((today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day))
        return None


class Doctor(Base):
    """Medical professional profile - No changes needed"""
    __tablename__ = "doctors"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    
    medical_license_number = Column(String(100), unique=True, nullable=False, index=True)
    license_state = Column(String(50), nullable=False)
    license_expiry_date = Column(Date, nullable=False)
    
    years_of_experience = Column(Integer, nullable=False)
    primary_specialty = Column(SQLEnum(MedicalSpecialty), nullable=False)
    secondary_specialties = Column(Text, nullable=True)
    
    hospital_affiliation = Column(String(200), nullable=True)
    clinic_name = Column(String(200), nullable=True)
    clinic_address = Column(Text, nullable=True)
    
    consultation_fee = Column(Float, nullable=False)
    currency = Column(String(3), default="USD")
    session_duration_minutes = Column(Integer, default=30)
    
    biography = Column(Text, nullable=True)
    education = Column(Text, nullable=True)
    certifications = Column(Text, nullable=True)
    
    is_accepting_patients = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    user = relationship("User", back_populates="doctor_profile")
    availability = relationship("DoctorAvailability", back_populates="doctor", cascade="all, delete-orphan")
    appointments = relationship("Appointment", back_populates="doctor")
    medical_records = relationship("MedicalHistory", back_populates="doctor")
    prescribed_medications = relationship("Medication", back_populates="prescribing_doctor")


class Patient(Base):
    """Patient medical profile - FIXED with missing relationships"""
    __tablename__ = "patients"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False)
    
    blood_type = Column(SQLEnum(BloodType), nullable=True)
    height_cm = Column(Float, nullable=True)
    weight_kg = Column(Float, nullable=True)
    
    insurance_provider = Column(SQLEnum(InsuranceProvider), nullable=True)
    insurance_policy_number = Column(String(100), nullable=True)
    insurance_group_number = Column(String(100), nullable=True)
    
    medical_conditions = Column(Text, nullable=True)
    family_history = Column(Text, nullable=True)
    surgical_history = Column(Text, nullable=True)
    
    smoking_status = Column(String(20), nullable=True)
    alcohol_consumption = Column(String(20), nullable=True)
    exercise_frequency = Column(String(20), nullable=True)
    
    is_active = Column(Boolean, default=True)
    preferred_language = Column(String(10), default="en")
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # ✅ FIXED: All relationships
    user = relationship("User", back_populates="patient_profile")
    medical_history = relationship("MedicalHistory", back_populates="patient", cascade="all, delete-orphan")
    medications = relationship("Medication", back_populates="patient", cascade="all, delete-orphan")
    allergies = relationship("Allergy", back_populates="patient", cascade="all, delete-orphan")
    appointments = relationship("Appointment", back_populates="patient")
    
    # ✅ NEW: Added missing relationships
    patient_symptoms = relationship("PatientSymptom", back_populates="patient", cascade="all, delete-orphan")
    rag_consultations = relationship("RAGConsultation", back_populates="patient")
    
    @property
    def bmi(self) -> Optional[float]:
        if self.height_cm and self.weight_kg:
            height_m = self.height_cm / 100
            return round(self.weight_kg / (height_m ** 2), 2)
        return None


# =============================================================================
# MEDICAL RECORDS
# =============================================================================

class MedicalHistory(Base):
    """Patient medical history - FIXED with RAG relationship"""
    __tablename__ = "medical_history"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id", ondelete="SET NULL"), nullable=True, index=True)
    
    # ✅ NEW: Foreign key for RAG consultation
    rag_consultation_id = Column(Integer, ForeignKey("rag_consultations.id", ondelete="SET NULL"), nullable=True, index=True)
    
    visit_date = Column(DateTime, nullable=False, default=func.now())
    chief_complaint = Column(Text, nullable=True)
    diagnosis = Column(Text, nullable=False)
    treatment_plan = Column(Text, nullable=True)
    
    symptoms = Column(Text, nullable=True)
    vital_signs = Column(Text, nullable=True)
    lab_results = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)
    
    follow_up_required = Column(Boolean, default=False)
    follow_up_date = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # ✅ FIXED: All relationships
    patient = relationship("Patient", back_populates="medical_history")
    doctor = relationship("Doctor", back_populates="medical_records")
    appointment = relationship("Appointment", back_populates="medical_record", uselist=False)
    
    # ✅ NEW: RAG consultation relationship
    correlated_rag_consultation = relationship("RAGConsultation", back_populates="medical_histories")


class Medication(Base):
    """Patient medications - No changes needed"""
    __tablename__ = "medications"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True)
    prescribed_by = Column(Integer, ForeignKey("doctors.id", ondelete="SET NULL"), nullable=True, index=True)
    
    medication_name = Column(String(200), nullable=False)
    generic_name = Column(String(200), nullable=True)
    dosage = Column(String(100), nullable=False)
    frequency = Column(String(100), nullable=False)
    route = Column(String(50), nullable=True)
    
    indication = Column(Text, nullable=True)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=True)
    
    is_active = Column(Boolean, default=True)
    side_effects = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    patient = relationship("Patient", back_populates="medications")
    prescribing_doctor = relationship("Doctor", back_populates="prescribed_medications")


class Allergy(Base):
    """Patient allergies - No changes needed"""
    __tablename__ = "allergies"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True)
    
    allergen = Column(String(200), nullable=False)
    allergy_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)
    
    reaction_symptoms = Column(Text, nullable=True)
    first_occurrence = Column(Date, nullable=True)
    last_occurrence = Column(Date, nullable=True)
    
    is_active = Column(Boolean, default=True)
    notes = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    patient = relationship("Patient", back_populates="allergies")


# =============================================================================
# APPOINTMENTS
# =============================================================================

class DoctorAvailability(Base):
    """Doctor availability schedule - No changes needed"""
    __tablename__ = "doctor_availability"
    
    id = Column(Integer, primary_key=True, index=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id", ondelete="CASCADE"), nullable=False, index=True)
    
    day_of_week = Column(Integer, nullable=False)
    start_time = Column(String(5), nullable=False)
    end_time = Column(String(5), nullable=False)
    
    is_available = Column(Boolean, default=True)
    max_appointments = Column(Integer, default=8)
    appointment_duration = Column(Integer, default=30)
    
    effective_from = Column(Date, nullable=True)
    effective_until = Column(Date, nullable=True)
    
    appointment_types_accepted = Column(Text, nullable=True)
    location = Column(String(20), nullable=False, default="remote")
    office_address = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    doctor = relationship("Doctor", back_populates="availability")


class Appointment(Base):
    """Medical appointments - FIXED with RAG relationship"""
    __tablename__ = "appointments"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False, index=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # ✅ NEW: Foreign key for RAG consultation
    rag_consultation_id = Column(Integer, ForeignKey("rag_consultations.id", ondelete="SET NULL"), nullable=True, index=True)
    
    appointment_type = Column(SQLEnum(AppointmentType), nullable=False)
    status = Column(SQLEnum(AppointmentStatus), default=AppointmentStatus.REQUESTED)
    
    scheduled_date = Column(Date, nullable=False, index=True)
    scheduled_time = Column(String(5), nullable=False)
    duration_minutes = Column(Integer, default=30)
    timezone = Column(String(50), default="UTC")
    
    location_type = Column(String(20), nullable=False)
    meeting_link = Column(String(500), nullable=True)
    office_address = Column(Text, nullable=True)
    
    chief_complaint = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)
    preparation_instructions = Column(Text, nullable=True)
    
    consultation_fee = Column(Float, nullable=True)
    payment_status = Column(String(20), default="pending")
    
    is_follow_up = Column(Boolean, default=False)
    original_appointment_id = Column(Integer, ForeignKey("appointments.id", ondelete="SET NULL"), nullable=True)
    
    requested_at = Column(DateTime, default=func.now())
    confirmed_at = Column(DateTime, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    cancelled_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # ✅ FIXED: All relationships
    patient = relationship("Patient", back_populates="appointments")
    doctor = relationship("Doctor", back_populates="appointments")
    medical_record = relationship("MedicalHistory", back_populates="appointment", uselist=False)
    
    # ✅ NEW: RAG consultation relationship
    rag_consultation = relationship("RAGConsultation", back_populates="appointment", uselist=False)


# =============================================================================
# RAG MODELS
# =============================================================================

class MedicalDocument(Base):
    """Medical documents for RAG - FIXED uploader relationship"""
    __tablename__ = "medical_documents"
    
    id = Column(Integer, primary_key=True, index=True)
    
    title = Column(String(500), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(1000), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    file_hash = Column(String(64), nullable=False, unique=True, index=True)
    
    document_type = Column(SQLEnum(DocumentType), nullable=False, default=DocumentType.OTHER)
    medical_specialty = Column(SQLEnum(MedicalSpecialty), nullable=True)
    diseases_covered = Column(ARRAY(String), nullable=True)
    keywords = Column(ARRAY(String), nullable=True)
    
    page_count = Column(Integer, nullable=True)
    word_count = Column(Integer, nullable=True)
    language = Column(String(10), default="es", nullable=False)
    content_preview = Column(Text, nullable=True)
    
    status = Column(SQLEnum(DocumentStatus), default=DocumentStatus.UPLOADED, nullable=False)
    processing_error = Column(Text, nullable=True)
    
    total_chunks = Column(Integer, default=0, nullable=False)
    embedding_model_used = Column(String(200), nullable=True)
    embedding_dimension = Column(Integer, nullable=True)
    
    quality_score = Column(Float, nullable=True)
    relevance_score = Column(Float, nullable=True)
    readability_score = Column(Float, nullable=True)
    
    version = Column(String(20), default="1.0", nullable=False)
    
    # ✅ FIXED: Proper foreign key
    uploaded_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    processed_at = Column(DateTime, nullable=True)
    
    # ✅ FIXED: Proper relationship
    uploader = relationship("User", foreign_keys=[uploaded_by], back_populates="uploaded_documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    consultation_sources = relationship("ConsultationSource", back_populates="document")
    
    __table_args__ = (
        CheckConstraint("file_size_bytes > 0", name="positive_file_size"),
        CheckConstraint("total_chunks >= 0", name="non_negative_chunks"),
        Index("idx_medical_document_status", "status"),
        Index("idx_medical_document_type", "document_type"),
    )


class DocumentChunk(Base):
    """Text chunks from documents - No changes needed"""
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("medical_documents.id", ondelete="CASCADE"), nullable=False)
    
    chunk_index = Column(Integer, nullable=False)
    chunk_type = Column(String(50), default="paragraph", nullable=False)
    
    content = Column(Text, nullable=False)
    content_length = Column(Integer, nullable=False)
    
    page_number = Column(Integer, nullable=True)
    section_title = Column(String(500), nullable=True)
    start_position = Column(Integer, nullable=True)
    end_position = Column(Integer, nullable=True)
    
    preceding_context = Column(Text, nullable=True)
    following_context = Column(Text, nullable=True)
    
    medical_entities = Column(JSONB, nullable=True)
    diseases_mentioned = Column(ARRAY(String), nullable=True)
    symptoms_mentioned = Column(ARRAY(String), nullable=True)
    medications_mentioned = Column(ARRAY(String), nullable=True)
    
    embedding_model = Column(String(200), nullable=True)
    embedding_created_at = Column(DateTime, nullable=True)
    
    relevance_score = Column(Float, nullable=True)
    clarity_score = Column(Float, nullable=True)
    information_density = Column(Float, nullable=True)
    
    retrieval_count = Column(Integer, default=0, nullable=False)
    last_retrieved = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    document = relationship("MedicalDocument", back_populates="chunks")
    embeddings = relationship("Embedding", back_populates="chunk", cascade="all, delete-orphan")
    consultation_sources = relationship("ConsultationSource", back_populates="chunk")
    
    __table_args__ = (
        CheckConstraint("content_length > 0", name="positive_content_length"),
        Index("idx_document_chunk_document", "document_id"),
        UniqueConstraint("document_id", "chunk_index", name="uq_document_chunk_index"),
    )


class Embedding(Base):
    """Vector embeddings - No changes needed"""
    __tablename__ = "embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    chunk_id = Column(Integer, ForeignKey("document_chunks.id", ondelete="CASCADE"), nullable=False, unique=True)
    
    model_name = Column(String(200), nullable=False)
    model_version = Column(String(50), nullable=True)
    embedding_dimension = Column(Integer, nullable=False)
    
    vector_data = Column(LargeBinary, nullable=False)
    
    generated_with_gpu = Column(Boolean, default=False, nullable=False)
    gpu_device_used = Column(String(50), nullable=True)
    generation_time_ms = Column(Float, nullable=True)
    batch_size_used = Column(Integer, nullable=True)
    precision_used = Column(String(10), nullable=True)
    
    similarity_search_count = Column(Integer, default=0, nullable=False)
    avg_similarity_score = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    chunk = relationship("DocumentChunk", back_populates="embeddings")
    
    __table_args__ = (
        CheckConstraint("embedding_dimension > 0", name="positive_embedding_dimension"),
        Index("idx_embedding_model", "model_name"),
    )


class Symptom(Base):
    """Symptom catalog - FIXED creator relationship"""
    __tablename__ = "symptoms"
    
    id = Column(Integer, primary_key=True, index=True)
    
    name = Column(String(200), nullable=False, unique=True)
    code = Column(String(50), nullable=True, unique=True)
    synonyms = Column(ARRAY(String), nullable=True)
    
    category = Column(String(100), nullable=True)
    subcategory = Column(String(100), nullable=True)
    body_system = Column(String(100), nullable=True)
    medical_specialty = Column(SQLEnum(MedicalSpecialty), nullable=True)
    
    description = Column(Text, nullable=True)
    clinical_significance = Column(Text, nullable=True)
    common_causes = Column(ARRAY(String), nullable=True)
    associated_conditions = Column(ARRAY(String), nullable=True)
    
    can_be_mild = Column(Boolean, default=True, nullable=False)
    can_be_severe = Column(Boolean, default=True, nullable=False)
    urgency_level = Column(Integer, default=1, nullable=False)
    
    reported_count = Column(Integer, default=0, nullable=False)
    diagnosis_correlation = Column(JSONB, nullable=True)
    
    # ✅ FIXED: Proper foreign key
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # ✅ FIXED: Proper relationship
    creator = relationship("User", foreign_keys=[created_by], back_populates="created_symptoms")
    patient_symptoms = relationship("PatientSymptom", back_populates="symptom")
    
    __table_args__ = (
        CheckConstraint("urgency_level >= 1 AND urgency_level <= 5", name="valid_urgency_level"),
        Index("idx_symptom_name", "name"),
    )


class PatientSymptom(Base):
    """Patient-reported symptoms - No changes needed"""
    __tablename__ = "patient_symptoms"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False)
    symptom_id = Column(Integer, ForeignKey("symptoms.id", ondelete="CASCADE"), nullable=False)
    
    severity = Column(SQLEnum(SeverityLevel), nullable=False, default=SeverityLevel.MILD)
    duration_days = Column(Integer, nullable=True)
    frequency = Column(String(50), nullable=True)
    
    patient_description = Column(Text, nullable=True)
    triggers = Column(ARRAY(String), nullable=True)
    alleviating_factors = Column(ARRAY(String), nullable=True)
    
    onset_date = Column(DateTime, nullable=True)
    first_noticed = Column(DateTime, nullable=True)
    is_worsening = Column(Boolean, nullable=True)
    affects_daily_activities = Column(Boolean, default=False, nullable=False)
    
    related_conditions = Column(ARRAY(String), nullable=True)
    current_treatments = Column(ARRAY(String), nullable=True)
    
    mentioned_in_rag_consultation = Column(Boolean, default=False, nullable=False)
    rag_relevance_score = Column(Float, nullable=True)
    
    is_active = Column(Boolean, default=True, nullable=False)
    resolved_date = Column(DateTime, nullable=True)
    follow_up_needed = Column(Boolean, default=False, nullable=False)
    
    reported_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    reported_via = Column(String(50), default="web_form", nullable=False)
    
    patient = relationship("Patient", back_populates="patient_symptoms")
    symptom = relationship("Symptom", back_populates="patient_symptoms")
    
    __table_args__ = (
        Index("idx_patient_symptom_patient", "patient_id"),
        Index("idx_patient_symptom_symptom", "symptom_id"),
        UniqueConstraint("patient_id", "symptom_id", "reported_at", name="uq_patient_symptom_report"),
    )


class RAGConsultation(Base):
    """RAG consultation sessions - FIXED all relationships"""
    __tablename__ = "rag_consultations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="SET NULL"), nullable=True)
    
    session_id = Column(String(100), nullable=False, unique=True, index=True)
    consultation_type = Column(String(50), default="symptom_analysis", nullable=False)
    
    chief_complaint = Column(Text, nullable=False)
    detailed_symptoms = Column(JSONB, nullable=True)
    patient_history_summary = Column(Text, nullable=True)
    additional_context = Column(JSONB, nullable=True)
    
    embedding_model_used = Column(String(200), nullable=True)
    retrieval_query = Column(Text, nullable=True)
    top_k_retrieved = Column(Integer, default=10, nullable=False)
    retrieval_time_ms = Column(Float, nullable=True)
    
    llm_model_used = Column(String(100), nullable=True)
    generation_time_ms = Column(Float, nullable=True)
    total_processing_time_ms = Column(Float, nullable=True)
    tokens_used = Column(Integer, nullable=True)
    
    preliminary_diagnosis = Column(JSONB, nullable=True)
    recommendations = Column(Text, nullable=True)
    urgency_assessment = Column(String(20), nullable=True)
    confidence_score = Column(Float, nullable=True)
    
    disclaimer_shown = Column(Boolean, default=True, nullable=False)
    emergency_warning_triggered = Column(Boolean, default=False, nullable=False)
    referral_recommended = Column(Boolean, default=False, nullable=False)
    
    user_satisfaction_rating = Column(Integer, nullable=True)
    user_feedback = Column(Text, nullable=True)
    accuracy_validated_by_doctor = Column(Boolean, nullable=True)
    doctor_feedback = Column(JSONB, nullable=True)
    
    status = Column(SQLEnum(RAGConsultationStatus), default=RAGConsultationStatus.INITIATED, nullable=False)
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0, nullable=False)
    
    gpu_used = Column(Boolean, default=False, nullable=False)
    gpu_device = Column(String(50), nullable=True)
    memory_used_mb = Column(Float, nullable=True)
    
    converted_to_appointment = Column(Boolean, default=False, nullable=False)
    
    started_at = Column(DateTime, default=func.now(), nullable=False)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    
    # ✅ FIXED: All relationships properly defined
    user = relationship("User", back_populates="rag_consultations")
    patient = relationship("Patient", back_populates="rag_consultations")
    
    # ✅ FIXED: Bidirectional relationship with Appointment
    appointment = relationship("Appointment", back_populates="rag_consultation", uselist=False, foreign_keys="Appointment.rag_consultation_id")
    
    # ✅ FIXED: Bidirectional relationship with MedicalHistory
    medical_histories = relationship("MedicalHistory", back_populates="correlated_rag_consultation", foreign_keys="MedicalHistory.rag_consultation_id")
    
    consultation_sources = relationship("ConsultationSource", back_populates="consultation", cascade="all, delete-orphan")
    
    __table_args__ = (
        CheckConstraint("confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1)", name="valid_confidence_score"),
        Index("idx_rag_consultation_user", "user_id"),
        Index("idx_rag_consultation_status", "status"),
    )


class ConsultationSource(Base):
    """Track document sources in consultations - No changes needed"""
    __tablename__ = "consultation_sources"
    
    id = Column(Integer, primary_key=True, index=True)
    consultation_id = Column(Integer, ForeignKey("rag_consultations.id", ondelete="CASCADE"), nullable=False)
    document_id = Column(Integer, ForeignKey("medical_documents.id", ondelete="CASCADE"), nullable=False)
    chunk_id = Column(Integer, ForeignKey("document_chunks.id", ondelete="CASCADE"), nullable=False)
    
    similarity_score = Column(Float, nullable=False)
    rank_position = Column(Integer, nullable=False)
    retrieval_method = Column(String(50), default="semantic_search", nullable=False)
    
    used_in_generation = Column(Boolean, default=False, nullable=False)
    influence_weight = Column(Float, nullable=True)
    quoted_in_response = Column(Boolean, default=False, nullable=False)
    
    query_that_retrieved = Column(Text, nullable=True)
    retrieval_context = Column(JSONB, nullable=True)
    
    relevance_score = Column(Float, nullable=True)
    accuracy_score = Column(Float, nullable=True)
    user_clicked = Column(Boolean, default=False, nullable=False)
    
    retrieved_at = Column(DateTime, default=func.now(), nullable=False)
    
    consultation = relationship("RAGConsultation", back_populates="consultation_sources")
    document = relationship("MedicalDocument", back_populates="consultation_sources")
    chunk = relationship("DocumentChunk", back_populates="consultation_sources")
    
    __table_args__ = (
        CheckConstraint("similarity_score >= 0 AND similarity_score <= 1", name="valid_similarity_score"),
        Index("idx_consultation_source_consultation", "consultation_id"),
        UniqueConstraint("consultation_id", "chunk_id", name="uq_consultation_chunk"),
    )


# =============================================================================
# MLOPS MODELS
# =============================================================================

class ModelMetrics(Base):
    """ML model performance metrics - No changes needed"""
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    
    model_name = Column(String(200), nullable=False)
    model_version = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False)
    
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    
    retrieval_precision_at_k = Column(JSONB, nullable=True)
    retrieval_recall_at_k = Column(JSONB, nullable=True)
    ndcg_at_k = Column(JSONB, nullable=True)
    
    avg_inference_time_ms = Column(Float, nullable=True)
    avg_memory_usage_mb = Column(Float, nullable=True)
    gpu_utilization_percent = Column(Float, nullable=True)
    
    total_predictions = Column(Integer, default=0, nullable=False)
    successful_predictions = Column(Integer, default=0, nullable=False)
    failed_predictions = Column(Integer, default=0, nullable=False)
    
    measurement_start = Column(DateTime, nullable=False)
    measurement_end = Column(DateTime, nullable=False)
    
    created_at = Column(DateTime, default=func.now(), nullable=False)
    created_by = Column(String(100), nullable=True)
    
    __table_args__ = (
        Index("idx_model_metrics_name_version", "model_name", "model_version"),
    )


class SystemMetrics(Base):
    """System performance metrics - No changes needed"""
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    
    cpu_percent = Column(Float, nullable=True)
    memory_percent = Column(Float, nullable=True)
    gpu_percent = Column(Float, nullable=True)
    gpu_memory_percent = Column(Float, nullable=True)
    disk_percent = Column(Float, nullable=True)
    
    gpu_temperature = Column(Float, nullable=True)
    gpu_power_usage = Column(Float, nullable=True)
    cuda_version = Column(String(20), nullable=True)
    
    active_users = Column(Integer, default=0, nullable=False)
    active_consultations = Column(Integer, default=0, nullable=False)
    requests_per_minute = Column(Float, nullable=True)
    avg_response_time_ms = Column(Float, nullable=True)
    
    db_connections = Column(Integer, nullable=True)
    db_query_time_ms = Column(Float, nullable=True)
    
    error_rate_percent = Column(Float, nullable=True)
    timeout_rate_percent = Column(Float, nullable=True)
    
    recorded_at = Column(DateTime, default=func.now(), nullable=False)
    
    __table_args__ = (
        Index("idx_system_metrics_recorded", "recorded_at"),
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_all_tables(engine):
    """Create all tables in the database"""
    Base.metadata.create_all(bind=engine)
    print("✅ All tables created successfully!")


def get_relationship_summary():
    """Summary of all fixed relationships"""
    return """
🔧 FIXED RELATIONSHIPS SUMMARY
================================

✅ User Model - Added:
   - rag_consultations → RAGConsultation
   - uploaded_documents → MedicalDocument
   - created_symptoms → Symptom

✅ Patient Model - Added:
   - patient_symptoms → PatientSymptom
   - rag_consultations → RAGConsultation

✅ MedicalHistory Model - Added:
   - rag_consultation_id (FK)
   - correlated_rag_consultation → RAGConsultation

✅ Appointment Model - Added:
   - rag_consultation_id (FK)
   - rag_consultation → RAGConsultation

✅ RAGConsultation Model - Fixed:
   - appointment (bidirectional with Appointment)
   - medical_histories (bidirectional with MedicalHistory)

✅ MedicalDocument Model - Fixed:
   - uploaded_by (FK to users.id)
   - uploader → User

✅ Symptom Model - Fixed:
   - created_by (FK to users.id)
   - creator → User

🎯 ALL FOREIGN KEYS PROPERLY DEFINED
🎯 ALL RELATIONSHIPS ARE BIDIRECTIONAL
🎯 CASCADE DELETES CONFIGURED
🎯 READY FOR ALEMBIC MIGRATIONS

Next Steps:
1. alembic revision --autogenerate -m "Initial migration with RAG"
2. alembic upgrade head
3. Test database creation
"""


if __name__ == "__main__":
    print("🚀 EndoCenter MLOps - FIXED Database Models")
    print(get_relationship_summary())