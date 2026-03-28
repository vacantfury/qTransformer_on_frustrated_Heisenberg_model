import re

# Pattern to match a number possibly followed by units/symbols (e.g., "60°", "3.14m", "100%")
_NUMBER_WITH_SUFFIX_PATTERN = re.compile(r'^([+-]?\d+(?:\.\d+)?)\s*[^\d]*$')


def is_str_a_number(s: str) -> bool:
    """Check if string represents a number, including cases like '60°', '100%'."""
    if not isinstance(s, str):
        return False
    s = s.strip()
    # Try direct float conversion first
    try:
        float(s)
        return True
    except ValueError:
        pass
    # Try matching number with suffix (e.g., "60°", "3.14m")
    match = _NUMBER_WITH_SUFFIX_PATTERN.match(s)
    return match is not None


def parse_str_to_number(s: str):
    """Parse string to number, handling cases like '60°', '100%'. Returns int or float."""
    if not isinstance(s, str):
        return None
    s = s.strip()
    # Try direct float conversion first
    try:
        n = float(s)
        if n.is_integer():
            return int(n)
        return n
    except ValueError:
        pass
    # Try matching number with suffix
    match = _NUMBER_WITH_SUFFIX_PATTERN.match(s)
    if match:
        num_str = match.group(1)
        n = float(num_str)
        return int(n) if n.is_integer() else n
    return None


def extract_first_number(s: str):
    """Extract the first number found in a string."""
    if not isinstance(s, str):
        return None
    match = re.search(r'[+-]?\d+(?:\.\d+)?', s)
    if not match:
        return None

    num_str = match.group()
    num = float(num_str)
    return int(num) if num.is_integer() else num
