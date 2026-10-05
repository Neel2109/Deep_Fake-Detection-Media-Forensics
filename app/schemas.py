from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class UserRegister(BaseModel):
    email: str = Field(..., description="User email")
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=2)


class UserLogin(BaseModel):
    email: str
    password: str


class CaseCreate(BaseModel):
    title: str = Field(..., min_length=2)
    category: str = "Forensic review"
    investigator: str = "Analyst"
    description: str = ""
    priority: str = "Medium"


class AnalysisRequest(BaseModel):
    case_id: Optional[str] = None
    case_title: Optional[str] = None
    investigator: Optional[str] = None


class AnalystReview(BaseModel):
    outcome: Literal["likely_original", "potentially_manipulated", "inconclusive"]
    rationale: str = Field(..., min_length=12, max_length=2000)
    reviewer: str = Field(default="Analyst", min_length=2, max_length=120)


class AuditEntry(BaseModel):
    timestamp: str
    message: str


class InvestigationResult(BaseModel):
    case_id: str
    media_type: str
    filename: str
    verdict: str
    probability: float
    confidence: float
    forensic_score: float
    summary: str
    manipulation_type: str
    metadata_summary: Dict[str, Any]
    suspicious_regions: List[Dict[str, Any]]
    suspicious_frames: List[Dict[str, Any]]
    evidence_scores: Dict[str, float]
    explanation: List[str]
    file_hash: str
