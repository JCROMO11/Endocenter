"""
EndoCenter MLOps - Database Models
Optimized model system for medical endocrinology applications
"""

from datetime import datetime, date
from typing import Optional, List
from enum import Enum
import uuid

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Date, Float, ForeignKey, Enum as SQLEnum, LargeBinary, JSON, CheckConstraint, Index, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

# Base para todos los modelos
Base = declarative_base()


class AppointmentType(str, Enum):
    """Types of medical appointments"""
    FIRST_TIME = "first_time"
    CONSULTATION = "consultation"  
    FOLLOW_UP = "follow_up"
    EMERGENCY = "emergency"
    ROUTINE_CHECKUP = "routine_checkup"


class AppointmentStatus(str, Enum):
    """Appointment status workflow"""
    REQUESTED = "requested"      # Patient requested
    CONFIRMED = "confirmed"      # Doctor confirmed
    SCHEDULED = "scheduled"      # Time slot reserved
    IN_PROGRESS = "in_progress"  # Currently happening
    COMPLETED = "completed"      # Finished successfully
    CANCELLED = "cancelled"      # Cancelled by patient/doctor
    NO_SHOW = "no_show"         # Patient didn't show up
    RESCHEDULED = "rescheduled"  # Moved to different time


class BloodType(str, Enum):
    """Blood type options"""
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
    """Common insurance providers"""
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
    """Medical specialties in Endocrinology"""
    GENERAL_ENDOCRINOLOGY = "general_endocrinology"
    DIABETES = "diabetes"
    THYROID = "thyroid"
    ADRENAL = "adrenal"
    REPRODUCTIVE_ENDOCRINOLOGY = "reproductive_endocrinology"
    PEDIATRIC_ENDOCRINOLOGY = "pediatric_endocrinology"
    BONE_METABOLISM = "bone_metabolism"
    OBESITY = "obesity"


class Gender(str, Enum):
    """Gender options"""
    MALE = "male"
    FEMALE = "female" 
    OTHER = "other"
    PREFER_NOT_TO_SAY = "prefer_not_to_say"


class UserRole(str, Enum):
    """Roles de usuario en el sistema"""
    PATIENT = "patient"
    DOCTOR = "doctor"
    ADMIN = "admin"
    
class DocumentType(str, Enum):
    """Medical document types for RAG"""
    TEXTBOOK = "textbook"
    RESEARCH_PAPER = "research_paper"
    CLINICAL_GUIDELINE = "clinical_guideline"
    CASE_STUDY = "case_study"
    DRUG_REFERENCE = "drug_reference"
    PROTOCOL = "protocol"
    OTHER = "other"


class DocumentStatus(str, Enum):
    """Document processing status"""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    CHUNKED = "chunked"
    EMBEDDED = "embedded"
    INDEXED = "indexed"
    ERROR = "error"


class RAGConsultationStatus(str, Enum):
    """RAG consultation status"""
    INITIATED = "initiated"
    PROCESSING = "processing"
    COMPLETED = "completed"
    REVIEWED = "reviewed"
    ERROR = "error"


class SeverityLevel(str, Enum):
    """Symptom severity levels"""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class User(Base):
    """
    Base user model - Main table for authentication
    Both patients and doctors have a record here
    """
    __tablename__ = "users"
    
    # Primary Key
    id = Column(Integer, primary_key=True, index=True)
    
    # Authentication fields
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=True)
    password_hash = Column(String(255), nullable=False)  # Better naming
    
    # Basic information
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    phone = Column(String(20), nullable=True)  # International phone support
    role = Column(SQLEnum(UserRole), nullable=False, default=UserRole.PATIENT)
    
    # User status
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    is_deleted = Column(Boolean, default=False)  # Soft delete support
    
    # Timezone support for international patients
    timezone = Column(String(50), default="UTC")
    
    # Add appointment_id for linking
    #appointment_id = Column(Integer, ForeignKey("appointments.id"), nullable=True, index=True)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    last_login = Column(DateTime, nullable=True)
    deleted_at = Column(DateTime, nullable=True)  # Soft delete timestamp
    
    # Relationships
    profile = relationship("UserProfile", back_populates="user", uselist=False)
    doctor_profile = relationship("Doctor", back_populates="user", uselist=False)
    patient_profile = relationship("Patient", back_populates="user", uselist=False)
    
    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}', role='{self.role}')>"
    
    @property
    def full_name(self) -> str:
        """Nombre completo del usuario"""
        return f"{self.first_name} {self.last_name}"
    
    @property
    def is_doctor(self) -> bool:
        """Verificar si el usuario es médico"""
        return self.role == UserRole.DOCTOR
    
    @property
    def is_patient(self) -> bool:
        """Verify if user is a patient"""
        return self.role == UserRole.PATIENT


class UserProfile(Base):
    """
    Extended user profile information
    Separated from User for better data organization
    """
    __tablename__ = "user_profiles"
    
    # Primary Key & Foreign Key
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    
    # Personal information
    date_of_birth = Column(Date, nullable=True)
    gender = Column(SQLEnum(Gender), nullable=True)
    
    # Contact information
    address = Column(Text, nullable=True)
    city = Column(String(100), nullable=True)
    state = Column(String(100), nullable=True)
    country = Column(String(100), nullable=True)
    postal_code = Column(String(20), nullable=True)
    
    # Emergency contact
    emergency_contact_name = Column(String(200), nullable=True)
    emergency_contact_phone = Column(String(20), nullable=True)
    emergency_contact_relationship = Column(String(100), nullable=True)
    
    # Profile settings
    language_preference = Column(String(10), default="en")  # en, es, etc.
    notification_preferences = Column(Text, nullable=True)  # JSON string
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationship back to User
    user = relationship("User", back_populates="profile")
    
    def __repr__(self):
        return f"<UserProfile(user_id={self.user_id}, city='{self.city}')>"
    
    @property 
    def age(self) -> Optional[int]:
        """Calculate age from date of birth"""
        if self.date_of_birth:
            today = date.today()
            return today.year - self.date_of_birth.year - \
                   ((today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day))
        return None


class Doctor(Base):
    """
    Medical professional profile
    Extends User for doctors with medical credentials
    """
    __tablename__ = "doctors"
    
    # Primary Key & Foreign Key
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    
    # Medical credentials
    medical_license_number = Column(String(100), unique=True, nullable=False, index=True)
    license_state = Column(String(50), nullable=False)  # State where licensed
    license_expiry_date = Column(Date, nullable=False)
    
    # Professional information
    years_of_experience = Column(Integer, nullable=False)
    primary_specialty = Column(SQLEnum(MedicalSpecialty), nullable=False)
    secondary_specialties = Column(Text, nullable=True)  # JSON array of additional specialties
    
    # Practice information
    hospital_affiliation = Column(String(200), nullable=True)
    clinic_name = Column(String(200), nullable=True)
    clinic_address = Column(Text, nullable=True)
    
    # Consultation details
    consultation_fee = Column(Float, nullable=False)  # In USD
    currency = Column(String(3), default="USD")
    session_duration_minutes = Column(Integer, default=30)  # Default 30-minute sessions
    
    # Professional bio
    biography = Column(Text, nullable=True)
    education = Column(Text, nullable=True)  # JSON string with education details
    certifications = Column(Text, nullable=True)  # JSON string with certifications
    
    # Status
    is_accepting_patients = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)  # Medical verification status
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="doctor_profile")
    availability = relationship("DoctorAvailability", back_populates="doctor")
    appointments = relationship("Appointment", back_populates="doctor")
    medical_records = relationship("MedicalHistory", back_populates="doctor")
    prescribed_medications = relationship("Medication", back_populates="prescribing_doctor")
    
    def __repr__(self):
        return f"<Doctor(id={self.id}, license='{self.medical_license_number}', specialty='{self.primary_specialty}')>"
    
    @property
    def full_name(self) -> str:
        """Get doctor's full name from user profile"""
        return self.user.full_name if self.user else "Unknown Doctor"
    
    @property
    def display_name(self) -> str:
        """Professional display name with Dr. prefix"""
        return f"Dr. {self.full_name}"
    
    @property
    def is_license_valid(self) -> bool:
        """Check if medical license is still valid"""
        from datetime import date
        return self.license_expiry_date > date.today()
    
    @property
    def experience_level(self) -> str:
        """Categorize doctor by experience level"""
        if self.years_of_experience < 3:
            return "Junior"
        elif self.years_of_experience < 10:
            return "Mid-level" 
        else:
            return "Senior"


class Patient(Base):
    """
    Patient medical profile
    Extends User for patients with medical information
    """
    __tablename__ = "patients"
    
    # Primary Key & Foreign Key
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    
    # Medical Information
    blood_type = Column(SQLEnum(BloodType), nullable=True)
    height_cm = Column(Float, nullable=True)  # Height in centimeters
    weight_kg = Column(Float, nullable=True)  # Weight in kilograms
    
    # Insurance Information
    insurance_provider = Column(SQLEnum(InsuranceProvider), nullable=True)
    insurance_policy_number = Column(String(100), nullable=True)
    insurance_group_number = Column(String(100), nullable=True)
    
    # Medical History Summary
    medical_conditions = Column(Text, nullable=True)  # JSON array of conditions
    family_history = Column(Text, nullable=True)     # JSON object with family medical history
    surgical_history = Column(Text, nullable=True)   # JSON array of past surgeries
    
    # Lifestyle Information
    smoking_status = Column(String(20), nullable=True)  # never, former, current
    alcohol_consumption = Column(String(20), nullable=True)  # none, light, moderate, heavy
    exercise_frequency = Column(String(20), nullable=True)  # daily, weekly, rarely, never
    
    # Patient Status
    is_active = Column(Boolean, default=True)
    preferred_language = Column(String(10), default="en")
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="patient_profile")
    medical_history = relationship("MedicalHistory", back_populates="patient")
    medications = relationship("Medication", back_populates="patient")
    allergies = relationship("Allergy", back_populates="patient") 
    appointments = relationship("Appointment", back_populates="patient")
    
    def __repr__(self):
        return f"<Patient(id={self.id}, user_id={self.user_id}, blood_type='{self.blood_type}')>"
    
    @property
    def full_name(self) -> str:
        """Get patient's full name from user profile"""
        return self.user.full_name if self.user else "Unknown Patient"
    
    @property
    def age(self) -> Optional[int]:
        """Get age from user profile"""
        if self.user and self.user.profile:
            return self.user.profile.age
        return None
    
    @property
    def bmi(self) -> Optional[float]:
        """Calculate Body Mass Index"""
        if self.height_cm and self.weight_kg:
            height_m = self.height_cm / 100  # Convert to meters
            return round(self.weight_kg / (height_m ** 2), 2)
        return None
    
    @property
    def bmi_category(self) -> Optional[str]:
        """BMI category classification"""
        bmi = self.bmi
        if bmi is None:
            return None
        
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Normal weight"
        elif bmi < 30:
            return "Overweight"
        else:
            return "Obese"


class MedicalHistory(Base):
    """
    Patient medical history records
    Stores clinical records, diagnoses, and visit notes
    """
    __tablename__ = "medical_history"
    
    # Primary Key & Foreign Key
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=True, index=True)
    
    # Medical Record Information
    visit_date = Column(DateTime, nullable=False, default=func.now())
    chief_complaint = Column(Text, nullable=True)  # Why patient came
    diagnosis = Column(Text, nullable=False)       # Doctor's diagnosis
    treatment_plan = Column(Text, nullable=True)   # Treatment recommendations
    
    # Clinical Data
    symptoms = Column(Text, nullable=True)          # JSON array of symptoms
    vital_signs = Column(Text, nullable=True)       # JSON object with BP, HR, etc.
    lab_results = Column(Text, nullable=True)       # JSON object with lab values
    notes = Column(Text, nullable=True)             # Doctor's additional notes
    
    # Follow-up Information
    follow_up_required = Column(Boolean, default=False)
    follow_up_date = Column(DateTime, nullable=True)
    
    # Record Status
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    patient = relationship("Patient", back_populates="medical_history")
    doctor = relationship("Doctor", back_populates="medical_records")
    appointment = relationship("Appointment", back_populates="medical_record")
    
    def __repr__(self):
        return f"<MedicalHistory(id={self.id}, patient_id={self.patient_id}, date='{self.visit_date.date()}')>"


class Medication(Base):
    """
    Patient current medications
    Tracks current and past medications
    """
    __tablename__ = "medications"
    
    # Primary Key & Foreign Key
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    prescribed_by = Column(Integer, ForeignKey("doctors.id"), nullable=True, index=True)
    
    # Medication Information
    medication_name = Column(String(200), nullable=False)
    generic_name = Column(String(200), nullable=True)
    dosage = Column(String(100), nullable=False)  # e.g., "10mg"
    frequency = Column(String(100), nullable=False)  # e.g., "twice daily"
    route = Column(String(50), nullable=True)  # oral, injection, etc.
    
    # Treatment Information
    indication = Column(Text, nullable=True)  # What it's treating
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=True)  # null if ongoing
    
    # Status
    is_active = Column(Boolean, default=True)
    side_effects = Column(Text, nullable=True)  # Any reported side effects
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    patient = relationship("Patient", back_populates="medications")
    prescribing_doctor = relationship("Doctor", back_populates="prescribed_medications")
    
    def __repr__(self):
        return f"<Medication(id={self.id}, name='{self.medication_name}', dosage='{self.dosage}')>"
    
    @property
    def is_current(self) -> bool:
        """Check if medication is currently active"""
        from datetime import date
        return self.is_active and (self.end_date is None or self.end_date > date.today())


class Allergy(Base):
    """
    Patient allergies and adverse reactions
    Important for safe prescribing
    """
    __tablename__ = "allergies"
    
    # Primary Key & Foreign Key
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    
    # Allergy Information
    allergen = Column(String(200), nullable=False)  # What they're allergic to
    allergy_type = Column(String(50), nullable=False)  # drug, food, environmental
    severity = Column(String(20), nullable=False)  # mild, moderate, severe
    
    # Reaction Details
    reaction_symptoms = Column(Text, nullable=True)  # JSON array of symptoms
    first_occurrence = Column(Date, nullable=True)
    last_occurrence = Column(Date, nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True)
    notes = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    patient = relationship("Patient", back_populates="allergies")
    
    def __repr__(self):
        return f"<Allergy(id={self.id}, allergen='{self.allergen}', severity='{self.severity}')>"


class DoctorAvailability(Base):
    """
    Doctor availability schedule
    Defines when doctors are available for appointments
    """
    __tablename__ = "doctor_availability"
    
    # Primary Key & Foreign Key
    id = Column(Integer, primary_key=True, index=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=False, index=True)
    
    # Time Slot Information
    day_of_week = Column(Integer, nullable=False)  # 0=Monday, 6=Sunday
    start_time = Column(String(5), nullable=False)  # "09:00" format
    end_time = Column(String(5), nullable=False)    # "17:00" format
    
    # Availability Details
    is_available = Column(Boolean, default=True)
    max_appointments = Column(Integer, default=8)  # Max appointments in this slot
    appointment_duration = Column(Integer, default=30)  # Minutes per appointment
    
    # Date Range (for temporary availability changes)
    effective_from = Column(Date, nullable=True)  # When this schedule starts
    effective_until = Column(Date, nullable=True) # When this schedule ends
    
    # Appointment Types Accepted
    appointment_types_accepted = Column(Text, nullable=True)  # JSON array
    
    # Location
    location = Column(String(20), nullable=False, default="remote")  # remote, in_person, both
    office_address = Column(Text, nullable=True)  # If in_person
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    doctor = relationship("Doctor", back_populates="availability")
    
    def __repr__(self):
        return f"<DoctorAvailability(doctor_id={self.doctor_id}, day={self.day_of_week}, {self.start_time}-{self.end_time})>"
    
    @property
    def day_name(self) -> str:
        """Convert day number to day name"""
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        return days[self.day_of_week]
    
    @property
    def time_slot(self) -> str:
        """Formatted time slot"""
        return f"{self.start_time} - {self.end_time}"


class Appointment(Base):
    """
    Medical appointments between doctors and patients
    Core business logic of the scheduling system
    """
    __tablename__ = "appointments"
    
    # Primary Key & Foreign Keys
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=False, index=True)
    
    # Appointment Details
    appointment_type = Column(SQLEnum(AppointmentType), nullable=False)
    status = Column(SQLEnum(AppointmentStatus), default=AppointmentStatus.REQUESTED)
    
    # Scheduling Information
    scheduled_date = Column(Date, nullable=False, index=True)
    scheduled_time = Column(String(5), nullable=False)  # "14:30" format
    duration_minutes = Column(Integer, default=30)
    timezone = Column(String(50), default="UTC")
    
    # Location
    location_type = Column(String(20), nullable=False)  # remote, in_person
    meeting_link = Column(String(500), nullable=True)   # For remote appointments
    office_address = Column(Text, nullable=True)        # For in-person
    
    # Appointment Content
    chief_complaint = Column(Text, nullable=True)       # Why patient is coming
    notes = Column(Text, nullable=True)                 # Additional notes
    preparation_instructions = Column(Text, nullable=True) # Pre-appointment instructions
    
    # Payment Information
    consultation_fee = Column(Float, nullable=True)
    payment_status = Column(String(20), default="pending")  # pending, paid, refunded
    
    # Follow-up
    is_follow_up = Column(Boolean, default=False)
    original_appointment_id = Column(Integer, ForeignKey("appointments.id"), nullable=True)
    
    # Status Timestamps
    requested_at = Column(DateTime, default=func.now())
    confirmed_at = Column(DateTime, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    cancelled_at = Column(DateTime, nullable=True)
    
    # General Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    patient = relationship("Patient", back_populates="appointments")
    doctor = relationship("Doctor", back_populates="appointments")
    medical_record = relationship("MedicalHistory", back_populates="appointment", uselist=False)
    
    def __repr__(self):
        return f"<Appointment(id={self.id}, patient_id={self.patient_id}, doctor_id={self.doctor_id}, date={self.scheduled_date})>"
    
    @property
    def appointment_datetime(self) -> datetime:
        """Combine date and time into datetime object"""
        from datetime import datetime, time
        time_obj = time.fromisoformat(self.scheduled_time)
        return datetime.combine(self.scheduled_date, time_obj)
    
    @property
    def is_upcoming(self) -> bool:
        """Check if appointment is in the future"""
        return self.appointment_datetime > datetime.now() and self.status in [
            AppointmentStatus.CONFIRMED, 
            AppointmentStatus.SCHEDULED
        ]
    
    @property
    def is_today(self) -> bool:
        """Check if appointment is today"""
        from datetime import date
        return self.scheduled_date == date.today()
    
    @property
    def can_be_cancelled(self) -> bool:
        """Check if appointment can still be cancelled"""
        return self.status in [
            AppointmentStatus.REQUESTED,
            AppointmentStatus.CONFIRMED, 
            AppointmentStatus.SCHEDULED
        ] and self.is_upcoming
        
class MedicalDocument(Base):
    """Medical documents for RAG knowledge base"""
    __tablename__ = "medical_documents"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Document metadata
    title = Column(String(500), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(1000), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    file_hash = Column(String(64), nullable=False, unique=True, index=True)  # SHA-256 hash
    
    # Document classification
    document_type = Column(SQLEnum(DocumentType), nullable=False, default=DocumentType.OTHER)
    medical_specialty = Column(SQLEnum(MedicalSpecialty), nullable=True)
    diseases_covered = Column(ARRAY(String), nullable=True)  # PostgreSQL array
    keywords = Column(ARRAY(String), nullable=True)
    
    # Content metadata
    page_count = Column(Integer, nullable=True)
    word_count = Column(Integer, nullable=True)
    language = Column(String(10), default="es", nullable=False)
    content_preview = Column(Text, nullable=True)  # First 500 characters
    
    # Processing status
    status = Column(SQLEnum(DocumentStatus), default=DocumentStatus.UPLOADED, nullable=False)
    processing_error = Column(Text, nullable=True)
    
    # RAG metrics
    total_chunks = Column(Integer, default=0, nullable=False)
    embedding_model_used = Column(String(200), nullable=True)
    embedding_dimension = Column(Integer, nullable=True)
    
    # Quality metrics
    quality_score = Column(Float, nullable=True)  # Document quality assessment (0-1)
    relevance_score = Column(Float, nullable=True)  # Medical relevance (0-1)
    readability_score = Column(Float, nullable=True)  # Text readability (0-1)
    
    # Audit and versioning
    version = Column(String(20), default="1.0", nullable=False)
    uploaded_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    processed_at = Column(DateTime, nullable=True)
    
    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    consultation_sources = relationship("ConsultationSource", back_populates="document")
    
    # Constraints and Indexes
    __table_args__ = (
        CheckConstraint("file_size_bytes > 0", name="positive_file_size"),
        CheckConstraint("total_chunks >= 0", name="non_negative_chunks"),
        CheckConstraint("quality_score IS NULL OR (quality_score >= 0 AND quality_score <= 1)", name="valid_quality_score"),
        Index("idx_medical_document_status", "status"),
        Index("idx_medical_document_type", "document_type"),
        Index("idx_medical_document_specialty", "medical_specialty"),
        Index("idx_medical_document_diseases", "diseases_covered"),
    )
    
    def __repr__(self):
        return f"<MedicalDocument(id={self.id}, title='{self.title[:50]}...', status={self.status})>"


class DocumentChunk(Base):
    """Text chunks from medical documents with embeddings"""
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("medical_documents.id"), nullable=False)
    
    # Chunk metadata
    chunk_index = Column(Integer, nullable=False)  # Order within document
    chunk_type = Column(String(50), default="paragraph", nullable=False)  # paragraph, table, list, etc.
    
    # Content
    content = Column(Text, nullable=False)
    content_length = Column(Integer, nullable=False)
    
    # Position within document
    page_number = Column(Integer, nullable=True)
    section_title = Column(String(500), nullable=True)
    start_position = Column(Integer, nullable=True)
    end_position = Column(Integer, nullable=True)
    
    # Context
    preceding_context = Column(Text, nullable=True)  # Previous chunk for context
    following_context = Column(Text, nullable=True)  # Next chunk for context
    
    # Medical annotations
    medical_entities = Column(JSONB, nullable=True)  # NER extracted entities
    diseases_mentioned = Column(ARRAY(String), nullable=True)
    symptoms_mentioned = Column(ARRAY(String), nullable=True)
    medications_mentioned = Column(ARRAY(String), nullable=True)
    
    # Embedding information
    embedding_model = Column(String(200), nullable=True)
    embedding_created_at = Column(DateTime, nullable=True)
    
    # Quality metrics
    relevance_score = Column(Float, nullable=True)
    clarity_score = Column(Float, nullable=True)
    information_density = Column(Float, nullable=True)
    
    # Usage statistics
    retrieval_count = Column(Integer, default=0, nullable=False)
    last_retrieved = Column(DateTime, nullable=True)
    
    # Audit
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    document = relationship("MedicalDocument", back_populates="chunks")
    embeddings = relationship("Embedding", back_populates="chunk", cascade="all, delete-orphan")
    consultation_sources = relationship("ConsultationSource", back_populates="chunk")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("content_length > 0", name="positive_content_length"),
        CheckConstraint("chunk_index >= 0", name="non_negative_chunk_index"),
        CheckConstraint("retrieval_count >= 0", name="non_negative_retrieval_count"),
        Index("idx_document_chunk_document", "document_id"),
        Index("idx_document_chunk_index", "chunk_index"),
        Index("idx_document_chunk_diseases", "diseases_mentioned"),
        Index("idx_document_chunk_symptoms", "symptoms_mentioned"),
        UniqueConstraint("document_id", "chunk_index", name="uq_document_chunk_index"),
    )
    
    def __repr__(self):
        return f"<DocumentChunk(id={self.id}, document_id={self.document_id}, chunk_index={self.chunk_index})>"


class Embedding(Base):
    """Vector embeddings for document chunks - GPU optimized"""
    __tablename__ = "embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    chunk_id = Column(Integer, ForeignKey("document_chunks.id"), nullable=False, unique=True)
    
    # Embedding metadata
    model_name = Column(String(200), nullable=False)
    model_version = Column(String(50), nullable=True)
    embedding_dimension = Column(Integer, nullable=False)
    
    # Vector data - Using binary storage for compatibility
    # Note: In production, you can use pgvector extension for native vector operations
    vector_data = Column(LargeBinary, nullable=False)
    
    # Generation metadata - GPU optimization tracking
    generated_with_gpu = Column(Boolean, default=False, nullable=False)
    gpu_device_used = Column(String(50), nullable=True)
    generation_time_ms = Column(Float, nullable=True)
    batch_size_used = Column(Integer, nullable=True)
    precision_used = Column(String(10), nullable=True)  # fp16, fp32
    
    # Performance metrics
    similarity_search_count = Column(Integer, default=0, nullable=False)
    avg_similarity_score = Column(Float, nullable=True)
    
    # Audit
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    chunk = relationship("DocumentChunk", back_populates="embeddings")
    
    # Constraints and Indexes
    __table_args__ = (
        CheckConstraint("embedding_dimension > 0", name="positive_embedding_dimension"),
        CheckConstraint("similarity_search_count >= 0", name="non_negative_search_count"),
        Index("idx_embedding_model", "model_name"),
        Index("idx_embedding_dimension", "embedding_dimension"),
        Index("idx_embedding_gpu", "generated_with_gpu"),
    )
    
    def __repr__(self):
        return f"<Embedding(id={self.id}, chunk_id={self.chunk_id}, model={self.model_name}, dim={self.embedding_dimension})>"


class Symptom(Base):
    """Medical symptom catalog for structured data"""
    __tablename__ = "symptoms"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Symptom identification
    name = Column(String(200), nullable=False, unique=True)
    code = Column(String(50), nullable=True, unique=True)  # ICD-10 or internal code
    synonyms = Column(ARRAY(String), nullable=True)
    
    # Classification
    category = Column(String(100), nullable=True)  # endocrine, metabolic, etc.
    subcategory = Column(String(100), nullable=True)
    body_system = Column(String(100), nullable=True)
    medical_specialty = Column(SQLEnum(MedicalSpecialty), nullable=True)
    
    # Medical details
    description = Column(Text, nullable=True)
    clinical_significance = Column(Text, nullable=True)
    common_causes = Column(ARRAY(String), nullable=True)
    associated_conditions = Column(ARRAY(String), nullable=True)
    
    # Severity and urgency
    can_be_mild = Column(Boolean, default=True, nullable=False)
    can_be_severe = Column(Boolean, default=True, nullable=False)
    urgency_level = Column(Integer, default=1, nullable=False)  # 1-5 scale
    
    # Usage statistics
    reported_count = Column(Integer, default=0, nullable=False)
    diagnosis_correlation = Column(JSONB, nullable=True)  # Statistical correlations
    
    # Audit
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Relationships
    patient_symptoms = relationship("PatientSymptom", back_populates="symptom")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("urgency_level >= 1 AND urgency_level <= 5", name="valid_urgency_level"),
        CheckConstraint("reported_count >= 0", name="non_negative_reported_count"),
        Index("idx_symptom_category", "category"),
        Index("idx_symptom_specialty", "medical_specialty"),
        Index("idx_symptom_urgency", "urgency_level"),
        Index("idx_symptom_name", "name"),
    )
    
    def __repr__(self):
        return f"<Symptom(id={self.id}, name='{self.name}', category='{self.category}')>"


class PatientSymptom(Base):
    """Patient-reported symptoms with context"""
    __tablename__ = "patient_symptoms"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    symptom_id = Column(Integer, ForeignKey("symptoms.id"), nullable=False)
    
    # Symptom details
    severity = Column(SQLEnum(SeverityLevel), nullable=False, default=SeverityLevel.MILD)
    duration_days = Column(Integer, nullable=True)
    frequency = Column(String(50), nullable=True)  # daily, weekly, occasional, etc.
    
    # Patient description
    patient_description = Column(Text, nullable=True)
    triggers = Column(ARRAY(String), nullable=True)
    alleviating_factors = Column(ARRAY(String), nullable=True)
    
    # Timing and context
    onset_date = Column(DateTime, nullable=True)
    first_noticed = Column(DateTime, nullable=True)
    is_worsening = Column(Boolean, nullable=True)
    affects_daily_activities = Column(Boolean, default=False, nullable=False)
    
    # Medical context
    related_conditions = Column(ARRAY(String), nullable=True)
    current_treatments = Column(ARRAY(String), nullable=True)
    
    # RAG integration
    mentioned_in_rag_consultation = Column(Boolean, default=False, nullable=False)
    rag_relevance_score = Column(Float, nullable=True)
    
    # Status tracking
    is_active = Column(Boolean, default=True, nullable=False)
    resolved_date = Column(DateTime, nullable=True)
    follow_up_needed = Column(Boolean, default=False, nullable=False)
    
    # Audit
    reported_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    reported_via = Column(String(50), default="web_form", nullable=False)  # web_form, rag_chat, appointment
    
    # Relationships
    patient = relationship("Patient", back_populates="patient_symptoms")
    symptom = relationship("Symptom", back_populates="patient_symptoms")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("duration_days IS NULL OR duration_days >= 0", name="non_negative_duration"),
        CheckConstraint("reported_via IN ('web_form', 'rag_chat', 'appointment', 'phone', 'mobile_app')", name="valid_reported_via"),
        CheckConstraint("rag_relevance_score IS NULL OR (rag_relevance_score >= 0 AND rag_relevance_score <= 1)", name="valid_rag_relevance_score"),
        Index("idx_patient_symptom_patient", "patient_id"),
        Index("idx_patient_symptom_symptom", "symptom_id"),
        Index("idx_patient_symptom_severity", "severity"),
        Index("idx_patient_symptom_active", "is_active"),
        Index("idx_patient_symptom_onset", "onset_date"),
        UniqueConstraint("patient_id", "symptom_id", "reported_at", name="uq_patient_symptom_report"),
    )
    
    def __repr__(self):
        return f"<PatientSymptom(id={self.id}, patient_id={self.patient_id}, severity={self.severity})>"


class RAGConsultation(Base):
    """RAG-powered medical consultation sessions"""
    __tablename__ = "rag_consultations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=True)  # Nullable for anonymous consultations
    
    # Session metadata
    session_id = Column(String(100), nullable=False, unique=True, index=True)  # For tracking chat sessions
    consultation_type = Column(String(50), default="symptom_analysis", nullable=False)
    
    # User input
    chief_complaint = Column(Text, nullable=False)
    detailed_symptoms = Column(JSONB, nullable=True)  # Structured symptom data
    patient_history_summary = Column(Text, nullable=True)
    additional_context = Column(JSONB, nullable=True)
    
    # RAG processing
    embedding_model_used = Column(String(200), nullable=True)
    retrieval_query = Column(Text, nullable=True)  # Processed query for retrieval
    top_k_retrieved = Column(Integer, default=10, nullable=False)
    retrieval_time_ms = Column(Float, nullable=True)
    
    # AI processing
    llm_model_used = Column(String(100), nullable=True)
    generation_time_ms = Column(Float, nullable=True)
    total_processing_time_ms = Column(Float, nullable=True)
    tokens_used = Column(Integer, nullable=True)
    
    # Results
    preliminary_diagnosis = Column(JSONB, nullable=True)  # Structured diagnosis with probabilities
    recommendations = Column(Text, nullable=True)
    urgency_assessment = Column(String(20), nullable=True)  # low, medium, high, emergency
    confidence_score = Column(Float, nullable=True)  # 0-1 confidence in diagnosis
    
    # Medical disclaimers and warnings
    disclaimer_shown = Column(Boolean, default=True, nullable=False)
    emergency_warning_triggered = Column(Boolean, default=False, nullable=False)
    referral_recommended = Column(Boolean, default=False, nullable=False)
    
    # Quality and feedback
    user_satisfaction_rating = Column(Integer, nullable=True)  # 1-5 stars
    user_feedback = Column(Text, nullable=True)
    accuracy_validated_by_doctor = Column(Boolean, nullable=True)
    doctor_feedback = Column(JSONB, nullable=True)
    
    # Status and lifecycle
    status = Column(SQLEnum(RAGConsultationStatus), default=RAGConsultationStatus.INITIATED, nullable=False)
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0, nullable=False)
    
    # Performance metrics - GPU optimization tracking
    gpu_used = Column(Boolean, default=False, nullable=False)
    gpu_device = Column(String(50), nullable=True)
    memory_used_mb = Column(Float, nullable=True)
    
    # Integration with appointment system
    converted_to_appointment = Column(Boolean, default=False, nullable=False)
    appointment_id = Column(Integer, nullable=True)  # Will be linked after appointment creation
    
    # Audit
    started_at = Column(DateTime, default=func.now(), nullable=False)
    completed_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    ip_address = Column(String(45), nullable=True)  # For analytics and security
    user_agent = Column(Text, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="rag_consultations")
    patient = relationship("Patient", back_populates="rag_consultations")
    appointment = relationship("Appointment", back_populates="rag_consultation", uselist=False)
    medical_histories = relationship("MedicalHistory", back_populates="correlated_rag_consultation")
    consultation_sources = relationship("ConsultationSource", back_populates="consultation", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("consultation_type IN ('symptom_analysis', 'medication_inquiry', 'general_question', 'follow_up')", name="valid_consultation_type"),
        CheckConstraint("urgency_assessment IS NULL OR urgency_assessment IN ('low', 'medium', 'high', 'emergency')", name="valid_urgency"),
        CheckConstraint("confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1)", name="valid_confidence_score"),
        CheckConstraint("user_satisfaction_rating IS NULL OR (user_satisfaction_rating >= 1 AND user_satisfaction_rating <= 5)", name="valid_satisfaction_rating"),
        CheckConstraint("retry_count >= 0", name="non_negative_retry_count"),
        CheckConstraint("top_k_retrieved > 0", name="positive_top_k"),
        Index("idx_rag_consultation_user", "user_id"),
        Index("idx_rag_consultation_patient", "patient_id"),
        Index("idx_rag_consultation_status", "status"),
        Index("idx_rag_consultation_started", "started_at"),
        Index("idx_rag_consultation_urgency", "urgency_assessment"),
        Index("idx_rag_consultation_converted", "converted_to_appointment"),
    )
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate consultation duration in seconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def __repr__(self):
        return f"<RAGConsultation(id={self.id}, session_id='{self.session_id}', status={self.status}, confidence={self.confidence_score})>"


class ConsultationSource(Base):
    """Track which documents/chunks were used in each consultation"""
    __tablename__ = "consultation_sources"
    
    id = Column(Integer, primary_key=True, index=True)
    consultation_id = Column(Integer, ForeignKey("rag_consultations.id"), nullable=False)
    document_id = Column(Integer, ForeignKey("medical_documents.id"), nullable=False)
    chunk_id = Column(Integer, ForeignKey("document_chunks.id"), nullable=False)
    
    # Retrieval metrics
    similarity_score = Column(Float, nullable=False)
    rank_position = Column(Integer, nullable=False)  # 1 = most relevant
    retrieval_method = Column(String(50), default="semantic_search", nullable=False)
    
    # Usage in generation
    used_in_generation = Column(Boolean, default=False, nullable=False)
    influence_weight = Column(Float, nullable=True)  # How much this source influenced the response
    quoted_in_response = Column(Boolean, default=False, nullable=False)
    
    # Context
    query_that_retrieved = Column(Text, nullable=True)
    retrieval_context = Column(JSONB, nullable=True)
    
    # Quality metrics
    relevance_score = Column(Float, nullable=True)  # User/doctor assessment
    accuracy_score = Column(Float, nullable=True)
    user_clicked = Column(Boolean, default=False, nullable=False)  # Did user click to read full source?
    
    # Audit
    retrieved_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Relationships
    consultation = relationship("RAGConsultation", back_populates="consultation_sources")
    document = relationship("MedicalDocument", back_populates="consultation_sources")
    chunk = relationship("DocumentChunk", back_populates="consultation_sources")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("similarity_score >= 0 AND similarity_score <= 1", name="valid_similarity_score"),
        CheckConstraint("rank_position > 0", name="positive_rank_position"),
        CheckConstraint("influence_weight IS NULL OR (influence_weight >= 0 AND influence_weight <= 1)", name="valid_influence_weight"),
        CheckConstraint("relevance_score IS NULL OR (relevance_score >= 0 AND relevance_score <= 1)", name="valid_relevance_score"),
        CheckConstraint("accuracy_score IS NULL OR (accuracy_score >= 0 AND accuracy_score <= 1)", name="valid_accuracy_score"),
        CheckConstraint("retrieval_method IN ('semantic_search', 'keyword_search', 'hybrid', 'reranking')", name="valid_retrieval_method"),
        Index("idx_consultation_source_consultation", "consultation_id"),
        Index("idx_consultation_source_document", "document_id"),
        Index("idx_consultation_source_chunk", "chunk_id"),
        Index("idx_consultation_source_similarity", "similarity_score"),
        Index("idx_consultation_source_rank", "rank_position"),
        Index("idx_consultation_source_used", "used_in_generation"),
        UniqueConstraint("consultation_id", "chunk_id", name="uq_consultation_chunk"),
    )
    
    def __repr__(self):
        return f"<ConsultationSource(consultation_id={self.consultation_id}, chunk_id={self.chunk_id}, rank={self.rank_position}, similarity={self.similarity_score:.3f})>"

# =============================================================================
# MLOPS AND MONITORING MODELS
# =============================================================================

class ModelMetrics(Base):
    """Track ML model performance metrics for MLOps"""
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Model identification
    model_name = Column(String(200), nullable=False)
    model_version = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False)  # embedding, llm, classification, etc.
    
    # Performance metrics
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    
    # RAG-specific metrics
    retrieval_precision_at_k = Column(JSONB, nullable=True)  # {k: precision} for different k values
    retrieval_recall_at_k = Column(JSONB, nullable=True)
    ndcg_at_k = Column(JSONB, nullable=True)  # Normalized Discounted Cumulative Gain
    
    # Performance metrics
    avg_inference_time_ms = Column(Float, nullable=True)
    avg_memory_usage_mb = Column(Float, nullable=True)
    gpu_utilization_percent = Column(Float, nullable=True)
    
    # Usage statistics
    total_predictions = Column(Integer, default=0, nullable=False)
    successful_predictions = Column(Integer, default=0, nullable=False)
    failed_predictions = Column(Integer, default=0, nullable=False)
    
    # Time period for metrics
    measurement_start = Column(DateTime, nullable=False)
    measurement_end = Column(DateTime, nullable=False)
    
    # Audit
    created_at = Column(DateTime, default=func.now(), nullable=False)
    created_by = Column(String(100), nullable=True)  # System or user
    
    # Constraints
    __table_args__ = (
        CheckConstraint("total_predictions >= 0", name="non_negative_total_predictions"),
        CheckConstraint("successful_predictions >= 0", name="non_negative_successful_predictions"),
        CheckConstraint("failed_predictions >= 0", name="non_negative_failed_predictions"),
        CheckConstraint("successful_predictions + failed_predictions <= total_predictions", name="valid_prediction_counts"),
        Index("idx_model_metrics_name_version", "model_name", "model_version"),
        Index("idx_model_metrics_type", "model_type"),
        Index("idx_model_metrics_period", "measurement_start", "measurement_end"),
    )
    
    def __repr__(self):
        return f"<ModelMetrics(model={self.model_name}:{self.model_version}, accuracy={self.accuracy})>"


class SystemMetrics(Base):
    """System performance and resource usage metrics"""
    __tablename__ = "system_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Resource utilization
    cpu_percent = Column(Float, nullable=True)
    memory_percent = Column(Float, nullable=True)
    gpu_percent = Column(Float, nullable=True)
    gpu_memory_percent = Column(Float, nullable=True)
    disk_percent = Column(Float, nullable=True)
    
    # GPU-specific metrics (RTX 5070 optimized)
    gpu_temperature = Column(Float, nullable=True)
    gpu_power_usage = Column(Float, nullable=True)
    cuda_version = Column(String(20), nullable=True)
    
    # Application metrics
    active_users = Column(Integer, default=0, nullable=False)
    active_consultations = Column(Integer, default=0, nullable=False)
    requests_per_minute = Column(Float, nullable=True)
    avg_response_time_ms = Column(Float, nullable=True)
    
    # Database metrics
    db_connections = Column(Integer, nullable=True)
    db_query_time_ms = Column(Float, nullable=True)
    
    # Error rates
    error_rate_percent = Column(Float, nullable=True)
    timeout_rate_percent = Column(Float, nullable=True)
    
    # Timestamp
    recorded_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Constraints
    __table_args__ = (
        CheckConstraint("cpu_percent IS NULL OR (cpu_percent >= 0 AND cpu_percent <= 100)", name="valid_cpu_percent"),
        CheckConstraint("memory_percent IS NULL OR (memory_percent >= 0 AND memory_percent <= 100)", name="valid_memory_percent"),
        CheckConstraint("gpu_percent IS NULL OR (gpu_percent >= 0 AND gpu_percent <= 100)", name="valid_gpu_percent"),
        CheckConstraint("active_users >= 0", name="non_negative_active_users"),
        CheckConstraint("active_consultations >= 0", name="non_negative_active_consultations"),
        Index("idx_system_metrics_recorded", "recorded_at"),
    )
    
    def __repr__(self):
        return f"<SystemMetrics(recorded_at={self.recorded_at}, cpu={self.cpu_percent}%, gpu={self.gpu_percent}%)>"

# =============================================================================
# HELPER FUNCTIONS AND UTILITIES
# =============================================================================

def create_all_tables(engine):
    """Create all tables in the database"""
    Base.metadata.create_all(bind=engine)
    print("✅ All EndoCenter tables (including RAG models) created successfully!")

def get_model_summary():
    """Get a summary of all models and their relationships"""
    existing_models = ["User", "UserProfile", "Doctor", "Patient", "MedicalHistory", "Medication", "Allergy", "DoctorAvailability", "Appointment"]
    rag_models = ["MedicalDocument", "DocumentChunk", "Embedding", "Symptom", "PatientSymptom", "RAGConsultation", "ConsultationSource"]
    mlops_models = ["ModelMetrics", "SystemMetrics"]
    
    total_models = len(existing_models) + len(rag_models) + len(mlops_models)
    
    summary = f"""
🏥 EndoCenter MLOps - Enhanced Database Schema Summary
{'='*70}
Total Models: {total_models}

📊 Model Categories:

🔹 Existing Models ({len(existing_models)}):
   {', '.join(existing_models)}

🔹 RAG Models ({len(rag_models)}):
   {', '.join(rag_models)}

🔹 MLOps Models ({len(mlops_models)}):
   {', '.join(mlops_models)}

🔗 Key RAG Integrations:
   • User → RAGConsultation (1:N)
   • Patient → PatientSymptom + RAGConsultation (1:N)
   • Appointment ↔ RAGConsultation (bidirectional)
   • MedicalHistory ↔ RAGConsultation (correlation tracking)
   • MedicalDocument → DocumentChunk → Embedding (1:N:1)
   • RAGConsultation → ConsultationSource (provenance tracking)

🚀 GPU Optimization Features:
   • FAISS GPU-accelerated embeddings (RTX 5070 optimized)
   • Batch processing tracking in Embedding model
   • GPU usage metrics in RAGConsultation
   • Performance monitoring in SystemMetrics
   • Model inference tracking in ModelMetrics

📈 MLOps Integration:
   • Real-time model performance tracking
   • System resource monitoring
   • RAG accuracy validation by doctors
   • User feedback loop for continuous improvement
   • Statistical correlation tracking for symptoms
"""
    
    return summary

# Print summary when imported
if __name__ == "__main__":
    print("🚀 EndoCenter MLOps - RAG-Enhanced Database Models")
    print(get_model_summary())
    
    print("\n✅ RAG Integration Complete!")
    print("\n🔥 Next Steps:")
    print("1. Create Alembic migration: alembic revision --autogenerate -m 'Add RAG models'")
    print("2. Apply migration: alembic upgrade head")
    print("3. Implement Phase 2: Medical ETL Pipeline")
    print("4. Build Phase 3: RAG Inference System")
    print("5. Setup Phase 4: MLFlow Integration")
    print("6. Deploy Phase 5: Monitoring & Alerts")