"""
Base grading result class.
Header author: Devin
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GradeResult:
    scores: dict[str, float]
    details: dict[str, Any] = field(default_factory=dict)
