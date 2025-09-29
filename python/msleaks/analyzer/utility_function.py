from typing import Any, Optional


def safe_convert_int(input_value: Any) -> Optional[int]:
    try:
        return int(input_value)
    except ValueError:
        print(f"ERROR: Invalid integer format")
    except TypeError:
        print(f"ERROR: Unconvertible type")
    return None