import os

from dataclasses import dataclass, fields
from typing import Any, get_type_hints
from dotenv import load_dotenv

# Call once
load_dotenv()

def _cast_value(value: str, target_type: Any) -> Any:
    """Helper to convert environment strings into explicit Python types.
    
    Accounts for string cleaning and ensures string type assertions.
    """
    # Clean up accidental leading/trailing whitespace from the environment
    cleaned_value = value.strip()

    # Handle string targets explicitly
    if target_type is str:
        if cleaned_value == "":
            raise ValueError("String fields cannot be empty")
        return cleaned_value

    # Handle empty string values for non-string types safely
    if cleaned_value == "":
        raise ValueError("Empty string cannot be cast to a non-string type")

    # Handle booleans safely
    if target_type is bool:
        return cleaned_value.lower() in ("true", "1", "t", "yes", "on")
        
    # Handle numbers
    if target_type is int:
        return int(cleaned_value)
    if target_type is float:
        return float(cleaned_value)
        
    return cleaned_value  # Fallback

@dataclass
class CVStreamAppEnv:
    RTSP_STREAM_SOURCE: str
    LOG_LEVEL: str

    @classmethod
    def load(cls) -> 'CVStreamAppEnv':
        init_kwargs = {}
        type_hints = get_type_hints(cls)

        for field in fields(cls):
            name = field.name
            type = type_hints[name]

            raw_value = os.getenv(name)
            
            if raw_value is not None:
                try:
                    init_kwargs[name] = _cast_value(raw_value, type)
                except ValueError as e:
                    raise TypeError(f"Env var '{name}' could not be cast to {type.__name__}: {e}")
            elif field.default is not field.default_factory:
                # Use dataclass default if missing from environment
                init_kwargs[name] = field.default
            else:
                raise KeyError(f"Missing required environment variable: '{name}'")

        return cls(**init_kwargs)