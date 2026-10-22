import pytest
from pydantic import ValidationError
from src.schemas import DimensionAssessment, EvidenceItem, LLMAssessment
from src.assessment_service import AssessmentError, parse_llm_json

GOOD = '{"dimensions":[{"dimension":"clarity","score":4,"sufficient_evidence":true,"explanation":"x","evidence":[]}]}'


def test_valid_schema():
    assert parse_llm_json(GOOD).dimensions[0].score == 4


def test_fenced_json_ok():
    assert parse_llm_json("```json\n" + GOOD + "\n```").dimensions[0].dimension == "clarity"


@pytest.mark.parametrize("bad", ["not json", "{", '{"dimensions":[{"dimension":"clarity","score":9}]}',
                                 '{"nothing":1}', ""])
def test_malformed_llm_output(bad):
    with pytest.raises(AssessmentError):
        parse_llm_json(bad)


def test_invalid_score_and_kind():
    with pytest.raises(ValidationError):
        DimensionAssessment(dimension="clarity", score=0)
    with pytest.raises(ValidationError):
        EvidenceItem(kind="guess", source="audio", description="abc")
