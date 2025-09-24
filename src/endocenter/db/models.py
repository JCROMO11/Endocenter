"""
EndoCenter MLOps - Database Models
Sistema de modelos optimizado para aplicación médica endocrinológica
"""

from datetime import datetime, date
from typing import Optional, List
from enum import Enum

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Date, Float, ForeignKey, Enum as SQLEnum
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
    appointment_id = Column(Integer, ForeignKey("appointments.id"), nullable=True, index=True)
    
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