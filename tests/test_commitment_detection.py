"""
Tests for commitment detection functionality.
"""
import pytest
import pandas as pd
import re
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import DEFAULT_COMMITMENT_PROMPT, normalize_phone


class TestCommitmentKeywordDetection:
    """Tests for keyword-based commitment detection."""

    def get_commitment_keywords(self):
        """Return the commitment keywords used in the app."""
        return [
            "yes", "i will", "i'll", "count me in", "sign me up",
            "i'm in", "absolutely", "definitely", "for sure",
            "commit", "committed", "attend", "participate", "join",
            "coming", "be there"
        ]

    def check_keyword_match(self, text, keywords):
        """Check if text contains any keyword."""
        if pd.isna(text):
            return False
        text_lower = str(text).lower()
        return any(kw in text_lower for kw in keywords)

    def test_yes_detected(self):
        """'Yes' should be detected."""
        keywords = self.get_commitment_keywords()
        assert self.check_keyword_match("Yes", keywords) == True
        assert self.check_keyword_match("yes!", keywords) == True
        assert self.check_keyword_match("YES", keywords) == True

    def test_commitment_phrases_detected(self):
        """Common commitment phrases should be detected."""
        keywords = self.get_commitment_keywords()
        assert self.check_keyword_match("I will be there", keywords) == True
        assert self.check_keyword_match("I'll do it", keywords) == True
        assert self.check_keyword_match("Count me in!", keywords) == True
        assert self.check_keyword_match("Sign me up", keywords) == True
        assert self.check_keyword_match("I'm in", keywords) == True

    def test_strong_affirmatives_detected(self):
        """Strong affirmative words should be detected."""
        keywords = self.get_commitment_keywords()
        assert self.check_keyword_match("Absolutely!", keywords) == True
        assert self.check_keyword_match("Definitely", keywords) == True
        assert self.check_keyword_match("For sure", keywords) == True

    def test_action_words_detected(self):
        """Action commitment words should be detected."""
        keywords = self.get_commitment_keywords()
        assert self.check_keyword_match("I'm committed to this", keywords) == True
        assert self.check_keyword_match("I will attend", keywords) == True
        assert self.check_keyword_match("I'll participate", keywords) == True
        assert self.check_keyword_match("I'll join", keywords) == True
        assert self.check_keyword_match("I'm coming", keywords) == True
        assert self.check_keyword_match("I'll be there", keywords) == True

    def test_non_commitment_not_detected(self):
        """Non-commitment messages should not be detected."""
        keywords = self.get_commitment_keywords()
        assert self.check_keyword_match("No thanks", keywords) == False
        assert self.check_keyword_match("Maybe later", keywords) == False
        assert self.check_keyword_match("I don't know", keywords) == False
        assert self.check_keyword_match("What time?", keywords) == False

    def test_false_positive_scenarios(self):
        """Messages that might be false positives should still be detected by keywords.
        LLM verification filters these out."""
        keywords = self.get_commitment_keywords()
        # These ARE detected by keywords (LLM filters)
        assert self.check_keyword_match("Yes I got your message", keywords) == True
        assert self.check_keyword_match("Yes, what do you want?", keywords) == True

    def test_empty_and_null_handling(self):
        """Empty and null values should not be detected."""
        keywords = self.get_commitment_keywords()
        assert self.check_keyword_match("", keywords) == False
        assert self.check_keyword_match(None, keywords) == False
        assert self.check_keyword_match(pd.NA, keywords) == False


class TestCommitmentPromptTemplate:
    """Tests for the commitment prompt template using actual prompt from app.py."""

    def test_prompt_has_conversation_placeholder(self):
        """Prompt should have {conversation} placeholder."""
        assert "{conversation}" in DEFAULT_COMMITMENT_PROMPT

    def test_prompt_format_works(self):
        """Prompt formatting should work correctly."""
        conversation = "YOU: Will you attend?\nTHEM: Yes, I'll be there!"
        formatted = DEFAULT_COMMITMENT_PROMPT.format(conversation=conversation)
        assert "Will you attend?" in formatted
        assert "Yes, I'll be there!" in formatted
        assert "{conversation}" not in formatted

    def test_prompt_instructs_yes_no_response(self):
        """Prompt should ask for YES or NO response."""
        assert "YES" in DEFAULT_COMMITMENT_PROMPT
        assert "NO" in DEFAULT_COMMITMENT_PROMPT

    def test_prompt_has_sequence_check(self):
        """Prompt should mention checking the sequence of messages."""
        assert "SEQUENCE" in DEFAULT_COMMITMENT_PROMPT.upper()

    def test_prompt_has_opinion_warning(self):
        """Prompt should warn about opinion vs commitment distinction."""
        assert "OPINION" in DEFAULT_COMMITMENT_PROMPT.upper()

    def test_prompt_has_action_request_check(self):
        """Prompt should mention checking for action requests."""
        assert "action" in DEFAULT_COMMITMENT_PROMPT.lower()

    def test_prompt_has_confidence_score_instruction(self):
        """Prompt should ask for confidence score."""
        # Should mention score or 0-100
        assert "0-100" in DEFAULT_COMMITMENT_PROMPT or "score" in DEFAULT_COMMITMENT_PROMPT.lower()


class TestConfidenceScoreParsing:
    """Tests for parsing YES/NO responses with confidence scores."""

    def parse_response(self, response_text):
        """Parse LLM response for YES/NO and confidence score.
        Returns (answer, confidence, is_committed)
        """
        text = response_text.strip().upper()

        # Try to match YES/NO with confidence score
        match = re.search(r'(YES|NO)[\s:\-\(]*(\d+)', text)
        if match:
            answer = match.group(1)
            confidence = int(match.group(2))
            is_committed = (answer == "YES" and confidence >= 85)
            return answer, confidence, is_committed

        # Fallback: look for YES/NO anywhere
        yes_match = re.search(r'\bYES\b', text)
        no_match = re.search(r'\bNO\b', text)

        if yes_match and not no_match:
            return "YES", 50, False  # Below threshold
        elif no_match:
            return "NO", 50, False
        else:
            return "NO", 0, False  # Unparseable defaults to NO

    def test_standard_yes_format(self):
        """Standard YES with confidence should parse correctly."""
        answer, conf, committed = self.parse_response("YES 95")
        assert answer == "YES"
        assert conf == 95
        assert committed == True

    def test_standard_no_format(self):
        """Standard NO with confidence should parse correctly."""
        answer, conf, committed = self.parse_response("NO 85")
        assert answer == "NO"
        assert conf == 85
        assert committed == False

    def test_yes_with_various_separators(self):
        """YES with different separators should parse correctly."""
        # Space
        answer, conf, _ = self.parse_response("YES 95")
        assert answer == "YES" and conf == 95

        # No space
        answer, conf, _ = self.parse_response("YES95")
        assert answer == "YES" and conf == 95

        # Dash
        answer, conf, _ = self.parse_response("YES - 95")
        assert answer == "YES" and conf == 95

        # Colon
        answer, conf, _ = self.parse_response("YES: 95")
        assert answer == "YES" and conf == 95

        # With percent
        answer, conf, _ = self.parse_response("YES 95%")
        assert answer == "YES" and conf == 95

        # Parentheses
        answer, conf, _ = self.parse_response("YES (95)")
        assert answer == "YES" and conf == 95

    def test_no_with_various_separators(self):
        """NO with different separators should parse correctly."""
        answer, conf, _ = self.parse_response("NO 75")
        assert answer == "NO" and conf == 75

        answer, conf, _ = self.parse_response("NO75")
        assert answer == "NO" and conf == 75

        answer, conf, _ = self.parse_response("NO - 75")
        assert answer == "NO" and conf == 75

    def test_lowercase_response(self):
        """Lowercase responses should parse correctly."""
        answer, conf, committed = self.parse_response("yes 95")
        assert answer == "YES"
        assert conf == 95
        assert committed == True

    def test_response_with_extra_text(self):
        """Response with extra text should still parse."""
        answer, conf, _ = self.parse_response("Based on the conversation: YES 90")
        assert answer == "YES"
        assert conf == 90

    def test_missing_confidence_yes(self):
        """YES without confidence should default to 50 (not committed)."""
        answer, conf, committed = self.parse_response("YES")
        assert answer == "YES"
        assert conf == 50
        assert committed == False  # Below 85 threshold

    def test_missing_confidence_no(self):
        """NO without confidence should default to 50."""
        answer, conf, committed = self.parse_response("NO")
        assert answer == "NO"
        assert conf == 50
        assert committed == False

    def test_unparseable_response(self):
        """Completely unparseable response should default to NO 0."""
        answer, conf, committed = self.parse_response("BASED ON THE GUIDELINES...")
        assert answer == "NO"
        assert conf == 0
        assert committed == False

    def test_empty_response(self):
        """Empty response should default to NO 0."""
        answer, conf, committed = self.parse_response("")
        assert answer == "NO"
        assert conf == 0
        assert committed == False


class TestCommitmentThreshold:
    """Tests for the >=85% commitment threshold."""

    def is_committed(self, answer, confidence):
        """Check if response meets commitment threshold."""
        return answer == "YES" and confidence >= 85

    def test_yes_85_is_committed(self):
        """YES with exactly 85% should be committed."""
        assert self.is_committed("YES", 85) == True

    def test_yes_86_is_committed(self):
        """YES with 86% should be committed."""
        assert self.is_committed("YES", 86) == True

    def test_yes_100_is_committed(self):
        """YES with 100% should be committed."""
        assert self.is_committed("YES", 100) == True

    def test_yes_84_is_not_committed(self):
        """YES with 84% should NOT be committed."""
        assert self.is_committed("YES", 84) == False

    def test_yes_50_is_not_committed(self):
        """YES with 50% should NOT be committed."""
        assert self.is_committed("YES", 50) == False

    def test_yes_0_is_not_committed(self):
        """YES with 0% should NOT be committed."""
        assert self.is_committed("YES", 0) == False

    def test_no_95_is_not_committed(self):
        """NO with 95% should NOT be committed (regardless of confidence)."""
        assert self.is_committed("NO", 95) == False

    def test_no_100_is_not_committed(self):
        """NO with 100% should NOT be committed."""
        assert self.is_committed("NO", 100) == False

    def test_no_85_is_not_committed(self):
        """NO with 85% should NOT be committed."""
        assert self.is_committed("NO", 85) == False


class TestPhoneNormalization:
    """Tests for phone number normalization using actual function from app.py."""

    def test_removes_plus_one(self):
        """Should remove +1 prefix."""
        assert normalize_phone("+15551234567") == "5551234567"

    def test_removes_one_prefix(self):
        """Should remove leading 1 from 11-digit numbers."""
        assert normalize_phone("15551234567") == "5551234567"

    def test_removes_formatting(self):
        """Should remove dashes, spaces, parentheses."""
        assert normalize_phone("(555) 123-4567") == "5551234567"
        assert normalize_phone("555-123-4567") == "5551234567"
        assert normalize_phone("555 123 4567") == "5551234567"

    def test_handles_float_format(self):
        """Should handle phone numbers stored as floats (with .0)."""
        assert normalize_phone("5551234567.0") == "5551234567"
        assert normalize_phone(5551234567.0) == "5551234567"

    def test_handles_integer(self):
        """Should handle phone numbers as integers."""
        assert normalize_phone(5551234567) == "5551234567"

    def test_10_digit_unchanged(self):
        """10-digit numbers should remain unchanged."""
        assert normalize_phone("5551234567") == "5551234567"

    def test_handles_none(self):
        """Should return None for None input."""
        assert normalize_phone(None) is None

    def test_handles_nan(self):
        """Should return None for NaN input."""
        assert normalize_phone(pd.NA) is None
        assert normalize_phone(float('nan')) is None

    def test_full_international_format(self):
        """Should handle full international format."""
        assert normalize_phone("+1-555-123-4567") == "5551234567"
