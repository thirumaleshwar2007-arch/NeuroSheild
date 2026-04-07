"""
Input Validation Module
"""
import re
from typing import List, Tuple, Union

class InputValidator:
    def __init__(self):
        self.ranges = {
            'age': (1, 120),
            'glucose': (50, 300),
            'bmi': (10, 60),
            'blood_pressure': (50, 200),
            'cholesterol': (100, 600),
            'heart_rate': (40, 220)
        }
    
    def validate_numeric_input(self, value: str, field_name: str = '') -> Tuple[bool, Union[float, str]]:
        """
        Validate numeric input
        
        Returns:
            Tuple of (is_valid, converted_value_or_error_message)
        """
        # Check if empty
        if not value or value.strip() == '':
            return False, f"{field_name} cannot be empty"
        
        # Remove any whitespace
        value = value.strip()
        
        # Check if it's a valid number
        try:
            num = float(value)
        except ValueError:
            return False, f"{field_name} must be a valid number"
        
        # Check if it's a reasonable value
        if num < 0:
            return False, f"{field_name} cannot be negative"
        
        if num > 10000:  # Arbitrary upper bound
            return False, f"{field_name} value seems too high"
        
        # Check specific ranges for known fields
        field_lower = field_name.lower()
        for key, (min_val, max_val) in self.ranges.items():
            if key in field_lower:
                if num < min_val or num > max_val:
                    return False, f"{field_name} should be between {min_val} and {max_val}"
        
        return True, num
    
    def validate_all_inputs(self, inputs: List[str], field_names: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate multiple inputs at once
        
        Returns:
            Tuple of (all_valid, list_of_errors)
        """
        errors = []
        valid_values = []
        
        for value, name in zip(inputs, field_names):
            is_valid, result = self.validate_numeric_input(value, name)
            if is_valid:
                valid_values.append(result)
            else:
                errors.append(result)
        
        return len(errors) == 0, errors if errors else valid_values