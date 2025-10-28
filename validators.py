import re
from typing import Dict, Any, List, Tuple

class ValidationError(Exception):
    """Structured validation error with field-specific messages."""
    
    def __init__(self, errors: Dict[str, str]):
        super().__init__("Validation failed")
        self.errors = errors
    
    def to_dict(self):
        """Convert to dictionary for JSON responses."""
        return {
            'success': False,
            'message': 'Validation failed',
            'errors': self.errors
        }

class LoanFormValidator:
    """Comprehensive loan application form validator with business rules."""
    
    # Field bounds
    MIN_INCOME = 1000
    MAX_INCOME = 50000000  # 50M reasonable upper bound
    MIN_CIBIL = 300
    MAX_CIBIL = 900
    MIN_TENURE = 12  # months
    MAX_TENURE = 480  # 40 years
    MIN_LOAN_AMOUNT = 10000  # 10K minimum
    MAX_LOAN_AMOUNT = 50000000  # 50M reasonable upper bound
    
    # Business rules
    MAX_LOAN_TO_INCOME_RATIO = 10.0  # 10x annual income
    MAX_EMI_TO_INCOME_RATIO = 0.6  # 60% of monthly income
    
    @staticmethod
    def _parse_number(value: Any, field_name: str, number_type=float) -> float:
        """Parse and validate numeric input."""
        if value is None or str(value).strip() == '':
            raise ValidationError({field_name: 'This field is required'})
        
        try:
            return number_type(value)
        except (ValueError, TypeError):
            type_name = 'integer' if number_type == int else 'number'
            raise ValidationError({field_name: f'Must be a valid {type_name}'})
    
    @classmethod
    def validate_loan_application(cls, form_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate complete loan application with business rules."""
        errors = {}
        validated_data = {}
        
        try:
            # Required fields validation
            required_fields = [
                'first_name', 'last_name', 'cibil_score', 'applicant_income',
                'coapplicant_income', 'loan_amount', 'loan_amount_term'
            ]
            
            for field in required_fields:
                value = form_data.get(field)
                if not value or str(value).strip() == '':
                    errors[field] = 'This field is required'
            
            if errors:
                raise ValidationError(errors)
            
            # Personal information validation
            validated_data['first_name'] = cls._validate_name(form_data.get('first_name'), 'first_name')
            validated_data['last_name'] = cls._validate_name(form_data.get('last_name'), 'last_name')
            
            # Categorical fields with safe defaults
            validated_data['gender'] = int(form_data.get('gender', 0))
            validated_data['marital_status'] = int(form_data.get('married', 0))
            validated_data['dependents_count'] = int(form_data.get('dependents', 0))
            validated_data['education_level'] = int(form_data.get('education', 0))
            validated_data['employment_type'] = int(form_data.get('self_employed', 0))
            validated_data['property_location'] = int(form_data.get('property_area', 0))
            validated_data['credit_history'] = float(form_data.get('credit_history', 1.0))
            
            # Numeric field validation
            cibil_score = cls._parse_number(form_data.get('cibil_score'), 'cibil_score', int)
            applicant_income = cls._parse_number(form_data.get('applicant_income'), 'applicant_income')
            coapplicant_income = cls._parse_number(form_data.get('coapplicant_income'), 'coapplicant_income')
            loan_amount = cls._parse_number(form_data.get('loan_amount'), 'loan_amount')
            loan_tenure = cls._parse_number(form_data.get('loan_amount_term'), 'loan_amount_term')
            
            # Range validations
            if not (cls.MIN_CIBIL <= cibil_score <= cls.MAX_CIBIL):
                errors['cibil_score'] = f'CIBIL score must be between {cls.MIN_CIBIL} and {cls.MAX_CIBIL}'
            
            if applicant_income < cls.MIN_INCOME:
                errors['applicant_income'] = f'Applicant income must be at least {cls.MIN_INCOME:,}'
            elif applicant_income > cls.MAX_INCOME:
                errors['applicant_income'] = f'Applicant income cannot exceed {cls.MAX_INCOME:,}'
            
            if coapplicant_income < 0:
                errors['coapplicant_income'] = 'Co-applicant income cannot be negative'
            elif coapplicant_income > cls.MAX_INCOME:
                errors['coapplicant_income'] = f'Co-applicant income cannot exceed {cls.MAX_INCOME:,}'
            
            if not (cls.MIN_LOAN_AMOUNT <= loan_amount <= cls.MAX_LOAN_AMOUNT):
                errors['loan_amount'] = f'Loan amount must be between {cls.MIN_LOAN_AMOUNT:,} and {cls.MAX_LOAN_AMOUNT:,}'
            
            if not (cls.MIN_TENURE <= loan_tenure <= cls.MAX_TENURE):
                errors['loan_amount_term'] = f'Loan tenure must be between {cls.MIN_TENURE} and {cls.MAX_TENURE} months'
            
            # Business rule validations
            if not errors:  # Only check business rules if basic validation passes
                total_income = applicant_income + coapplicant_income
                
                if total_income <= 0:
                    errors['applicant_income'] = 'Total household income must be positive'
                else:
                    # Loan-to-income ratio check (annual basis)
                    annual_income = total_income * 12
                    loan_to_income_ratio = (loan_amount * 1000) / annual_income  # loan_amount is in thousands
                    
                    if loan_to_income_ratio > cls.MAX_LOAN_TO_INCOME_RATIO:
                        errors['loan_amount'] = f'Loan amount is too high relative to income (max {cls.MAX_LOAN_TO_INCOME_RATIO}x annual income)'
                    
                    # EMI affordability check
                    if loan_tenure > 0:
                        # Rough EMI calculation (simple interest approximation)
                        monthly_emi = (loan_amount * 1000) / loan_tenure
                        emi_to_income_ratio = monthly_emi / total_income
                        
                        if emi_to_income_ratio > cls.MAX_EMI_TO_INCOME_RATIO:
                            errors['loan_amount_term'] = f'EMI would be too high relative to income (max {cls.MAX_EMI_TO_INCOME_RATIO * 100:.0f}% of monthly income)'
            
            if errors:
                raise ValidationError(errors)
            
            # Store validated numeric values
            validated_data.update({
                'cibil_score': cibil_score,
                'applicant_income': applicant_income,
                'coapplicant_income': coapplicant_income,
                'loan_amount': loan_amount,
                'loan_tenure': loan_tenure
            })
            
            return validated_data
            
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError({'general': f'Validation error: {str(e)}'})
    
    @staticmethod
    def _validate_name(name: str, field_name: str) -> str:
        """Validate name fields."""
        if not name or not name.strip():
            raise ValidationError({field_name: 'This field is required'})
        
        name = name.strip()
        
        if len(name) < 1:
            raise ValidationError({field_name: 'Name is too short'})
        
        if len(name) > 100:
            raise ValidationError({field_name: 'Name is too long (max 100 characters)'})
        
        # Allow letters, spaces, apostrophes, hyphens
        if not re.match(r"^[a-zA-Z\s'\-]+$", name):
            raise ValidationError({field_name: 'Name can only contain letters, spaces, apostrophes, and hyphens'})
        
        return name
    
    @staticmethod
    def validate_user_registration(form_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate user registration form."""
        errors = {}
        validated_data = {}
        
        # Required fields
        required_fields = ['user_id', 'email', 'full_name', 'password', 'confirm_password']
        for field in required_fields:
            if not form_data.get(field) or not str(form_data.get(field)).strip():
                errors[field] = 'This field is required'
        
        if errors:
            raise ValidationError(errors)
        
        # User ID validation
        user_id = form_data['user_id'].strip()
        if len(user_id) < 3:
            errors['user_id'] = 'Username must be at least 3 characters long'
        elif len(user_id) > 50:
            errors['user_id'] = 'Username cannot exceed 50 characters'
        elif not re.match(r'^[a-zA-Z0-9_]+$', user_id):
            errors['user_id'] = 'Username can only contain letters, numbers, and underscores'
        
        # Email validation
        email = form_data['email'].strip().lower()
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            errors['email'] = 'Please enter a valid email address'
        
        # Password validation
        password = form_data['password']
        confirm_password = form_data['confirm_password']
        
        if password != confirm_password:
            errors['confirm_password'] = 'Passwords do not match'
        
        password_errors = LoanFormValidator._validate_password_strength(password)
        if password_errors:
            errors['password'] = password_errors
        
        # Full name validation
        full_name = LoanFormValidator._validate_name(form_data['full_name'], 'full_name')
        
        if errors:
            raise ValidationError(errors)
        
        validated_data = {
            'user_id': user_id,
            'email': email,
            'full_name': full_name,
            'password': password,
            'mobile_number': form_data.get('mobile_number', '').strip()
        }
        
        return validated_data
    
    @staticmethod
    def _validate_password_strength(password: str) -> str:
        """Validate password meets security requirements."""
        if len(password) < 8:
            return 'Password must be at least 8 characters long'
        
        if not re.search(r'[A-Z]', password):
            return 'Password must contain at least one uppercase letter'
        
        if not re.search(r'[a-z]', password):
            return 'Password must contain at least one lowercase letter'
        
        if not re.search(r'\d', password):
            return 'Password must contain at least one number'
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return 'Password must contain at least one special character'
        
        return ''
