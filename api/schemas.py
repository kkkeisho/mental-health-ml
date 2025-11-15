from pydantic import BaseModel
from typing import Optional


class SurveyInput(BaseModel):
    Age: Optional[int]
    Gender: str
    Country: str
    self_employed: Optional[str]
    family_history: str
    work_interfere: Optional[str]
    no_employees: str
    remote_work: str
    tech_company: str
    benefits: str
    care_options: str
    wellness_program: str
    seek_help: str
    anonymity: str
    leave: str
    mental_health_consequence: str
    phys_health_consequence: str
    coworkers: str
    supervisor: str
    mental_health_interview: str
    phys_health_interview: str
    mental_vs_physical: str
    obs_consequence: str
