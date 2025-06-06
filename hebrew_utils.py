from bidi.algorithm import get_display
from rich.console import Console


def hebrew_print(t=""):
    """Print Hebrew (RTL) text in correct display order."""
    print(get_hebrew_display(t))


def get_hebrew_display(t):
    """Return Hebrew (RTL) text in correct display order as a string."""
    return get_display(t)


class HebrewConsole(Console):
    def print(self, *objects, **kwargs):
        new_objects = [
            get_hebrew_display(obj) if isinstance(obj, str) else obj for obj in objects
        ]
        super().print(*new_objects, **kwargs)


def parse_gematria(s: str) -> int:
    """
    Parse a Hebrew gematria string and return its integer value.
    Handles standard, final forms, and punctuation (geresh/gershayim).
    """
    try:
        ret = float(s)
        return ret
    except ValueError:
        pass
    gematria_map = {
        "א": 1,
        "ב": 2,
        "ג": 3,
        "ד": 4,
        "ה": 5,
        "ו": 6,
        "ז": 7,
        "ח": 8,
        "ט": 9,
        "י": 10,
        "כ": 20,
        "ך": 20,
        "ל": 30,
        "מ": 40,
        "ם": 40,
        "נ": 50,
        "ן": 50,
        "ס": 60,
        "ע": 70,
        "פ": 80,
        "ף": 80,
        "צ": 90,
        "ץ": 90,
        "ק": 100,
        "ר": 200,
        "ש": 300,
        "ת": 400,
    }
    s = s.replace("״", "").replace("׳", "").replace('"', "").replace("'", "")
    total = 0
    for c in s:
        if c in gematria_map:
            total += gematria_map[c]
    return total
