"""
Structured Response Schema for LLM Responses
Defines the JSON schema and validation for structured LLM responses
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json


class ConfidenceLevel(Enum):
    """Confidence level enumeration"""
    VERY_HIGH = "very_high"  # 90-100%
    HIGH = "high"           # 75-89%
    MEDIUM = "medium"       # 60-74%
    LOW = "low"            # 40-59%
    VERY_LOW = "very_low"  # 0-39%


class ResponseType(Enum):
    """Type of response"""
    DOCUMENT_BASED = "document_based"      # Based on retrieved documents
    CONVERSATIONAL = "conversational"     # General conversation/greeting
    CLARIFICATION = "clarification"       # Asking for clarification
    ERROR = "error"                       # Error response


@dataclass
class ConfidenceAssessment:
    """Detailed confidence assessment"""
    score: float  # 0.0 to 1.0
    level: ConfidenceLevel
    rationale: str
    factors: Dict[str, Dict[str, Union[int, str]]]  # Factor scores and explanations


@dataclass
class SourceReference:
    """Source document reference"""
    id: str
    title: str
    relevance_score: float
    excerpt: str
    metadata: Dict[str, Any]


@dataclass
class ResponseMetadata:
    """Response generation metadata"""
    model: str
    tokens_used: Dict[str, int]
    processing_time: float
    search_strategy: str
    documents_retrieved: int
    cache_metrics: Optional[Dict[str, Any]] = None


@dataclass
class StructuredResponse:
    """Complete structured response"""
    # Core response
    response_type: ResponseType
    answer: str
    
    # Confidence assessment
    confidence: ConfidenceAssessment
    
    # Supporting information
    sources: List[SourceReference]
    key_points: List[str]
    
    # Metadata
    metadata: ResponseMetadata
    
    # Optional fields
    follow_up_questions: Optional[List[str]] = None
    related_topics: Optional[List[str]] = None
    warnings: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StructuredResponse':
        """Create from dictionary"""
        # Convert enums
        data['response_type'] = ResponseType(data['response_type'])
        data['confidence']['level'] = ConfidenceLevel(data['confidence']['level'])
        
        # Convert nested dataclasses
        data['confidence'] = ConfidenceAssessment(**data['confidence'])
        data['sources'] = [SourceReference(**src) for src in data['sources']]
        data['metadata'] = ResponseMetadata(**data['metadata'])
        
        return cls(**data)


def get_confidence_level(score: float) -> ConfidenceLevel:
    """Get confidence level from score"""
    if score >= 0.9:
        return ConfidenceLevel.VERY_HIGH
    elif score >= 0.75:
        return ConfidenceLevel.HIGH
    elif score >= 0.6:
        return ConfidenceLevel.MEDIUM
    elif score >= 0.4:
        return ConfidenceLevel.LOW
    else:
        return ConfidenceLevel.VERY_LOW


def get_confidence_emoji(level: ConfidenceLevel) -> str:
    """Get emoji for confidence level"""
    emoji_map = {
        ConfidenceLevel.VERY_HIGH: "🟢",
        ConfidenceLevel.HIGH: "🟢", 
        ConfidenceLevel.MEDIUM: "🟡",
        ConfidenceLevel.LOW: "🟠",
        ConfidenceLevel.VERY_LOW: "🔴"
    }
    return emoji_map.get(level, "🟡")


# JSON Schema for validation
STRUCTURED_RESPONSE_SCHEMA = {
    "type": "object",
    "required": ["response_type", "answer", "confidence", "sources", "key_points", "metadata"],
    "properties": {
        "response_type": {
            "type": "string",
            "enum": ["document_based", "conversational", "clarification", "error"]
        },
        "answer": {
            "type": "string",
            "minLength": 1
        },
        "confidence": {
            "type": "object",
            "required": ["score", "level", "rationale", "factors"],
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 1},
                "level": {
                    "type": "string", 
                    "enum": ["very_high", "high", "medium", "low", "very_low"]
                },
                "rationale": {"type": "string"},
                "factors": {
                    "type": "object",
                    "properties": {
                        "information_quality": {
                            "type": "object",
                            "properties": {
                                "score": {"type": "integer", "minimum": 0, "maximum": 30},
                                "explanation": {"type": "string"}
                            }
                        },
                        "query_coverage": {
                            "type": "object", 
                            "properties": {
                                "score": {"type": "integer", "minimum": 0, "maximum": 30},
                                "explanation": {"type": "string"}
                            }
                        },
                        "source_consistency": {
                            "type": "object",
                            "properties": {
                                "score": {"type": "integer", "minimum": 0, "maximum": 20},
                                "explanation": {"type": "string"}
                            }
                        },
                        "specificity": {
                            "type": "object",
                            "properties": {
                                "score": {"type": "integer", "minimum": 0, "maximum": 20},
                                "explanation": {"type": "string"}
                            }
                        }
                    }
                }
            }
        },
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "title", "relevance_score", "excerpt"],
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "relevance_score": {"type": "number", "minimum": 0, "maximum": 1},
                    "excerpt": {"type": "string"},
                    "metadata": {"type": "object"}
                }
            }
        },
        "key_points": {
            "type": "array",
            "items": {"type": "string"}
        },
        "metadata": {
            "type": "object",
            "required": ["model", "tokens_used", "processing_time", "search_strategy", "documents_retrieved"],
            "properties": {
                "model": {"type": "string"},
                "tokens_used": {"type": "object"},
                "processing_time": {"type": "number"},
                "search_strategy": {"type": "string"},
                "documents_retrieved": {"type": "integer"},
                "cache_metrics": {"type": "object"}
            }
        },
        "follow_up_questions": {
            "type": "array",
            "items": {"type": "string"}
        },
        "related_topics": {
            "type": "array", 
            "items": {"type": "string"}
        },
        "warnings": {
            "type": "array",
            "items": {"type": "string"}
        }
    }
}
