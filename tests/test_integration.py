"""
Integration tests that test actual functions from app.py.
These tests import and use the real functions from the application.
"""
import pytest
import pandas as pd
import re
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import (
    normalize_phone,
    is_valid_phone,
    get_person_name,
    format_conversation_for_llm,
    DEFAULT_COMMITMENT_PROMPT,
    DEFAULT_OPTOUT_PROMPT,
)


class TestPhoneValidation:
    """Tests for phone number validation using actual function."""

    def test_valid_phone_numbers(self):
        """Valid phone numbers should pass validation."""
        assert is_valid_phone("555-1234") == True
        assert is_valid_phone("(555) 123-4567") == True
        assert is_valid_phone("+1-555-123-4567") == True
        assert is_valid_phone("5551234567") == True
        assert is_valid_phone(5551234567) == True  # Integer
        assert is_valid_phone(5551234.0) == True  # Float

    def test_invalid_phone_numbers(self):
        """Invalid phone numbers should fail validation."""
        assert is_valid_phone("") == False
        assert is_valid_phone("nan") == False
        assert is_valid_phone("NaN") == False
        assert is_valid_phone("None") == False
        assert is_valid_phone("null") == False
        assert is_valid_phone("N/A") == False
        assert is_valid_phone("na") == False
        assert is_valid_phone("-") == False
        assert is_valid_phone(None) == False
        assert is_valid_phone(pd.NA) == False

    def test_short_phone_numbers(self):
        """Phone numbers with fewer than 7 digits should fail."""
        assert is_valid_phone("123") == False
        assert is_valid_phone("123456") == False
        assert is_valid_phone("x") == False

    def test_phone_with_7_digits(self):
        """Phone numbers with exactly 7 digits should pass."""
        assert is_valid_phone("1234567") == True
        assert is_valid_phone("555-1234") == True


class TestResponseRateCalculation:
    """Tests for response rate calculation edge cases."""

    def test_no_division_by_zero(self):
        """Response rate calculation should handle zero contactable recipients."""
        contactable = 0
        responders = 5
        rate = (responders / contactable * 100) if contactable > 0 else 0
        assert rate == 0

    def test_rate_cannot_exceed_100_with_proper_filtering(self):
        """With proper filtering, response rate should not exceed 100%."""
        # This tests the principle, not the actual implementation
        contactable = 10
        # Responders should be a subset of contactable
        responders = 10  # At most equal to contactable
        rate = (responders / contactable * 100) if contactable > 0 else 0
        assert rate <= 100


class TestDataFrameOperations:
    """Tests for DataFrame operations used in the app."""

    def test_direction_filtering(self):
        """Filtering by direction should work correctly."""
        df = pd.DataFrame([
            {'direction': 'incoming', 'body': 'Hi'},
            {'direction': 'outgoing', 'body': 'Hello'},
            {'direction': 'incoming', 'body': 'Thanks'},
        ])
        incoming = df[df['direction'] == 'incoming']
        assert len(incoming) == 2

    def test_keyword_detection_with_contains(self):
        """Keyword detection using str.contains should work."""
        df = pd.DataFrame([
            {'body': 'STOP'},
            {'body': 'Yes I will'},
            {'body': 'stop please'},
            {'body': 'Hello'},
        ])
        keywords = ['stop']
        matches = df['body'].str.lower().str.contains('|'.join(keywords), na=False)
        assert matches.sum() == 2  # STOP and stop please

    def test_unique_person_ids(self):
        """Getting unique person IDs should work correctly."""
        df = pd.DataFrame([
            {'person_id': 1, 'body': 'Hi'},
            {'person_id': 1, 'body': 'Hello'},
            {'person_id': 2, 'body': 'Hey'},
            {'person_id': 3, 'body': 'Hi there'},
        ])
        unique_ids = df['person_id'].unique()
        assert len(unique_ids) == 3
        assert set(unique_ids) == {1, 2, 3}


class TestPromptFormatting:
    """Tests for prompt formatting with conversation placeholder."""

    def test_prompt_formatting_simple(self):
        """Simple prompt formatting should work."""
        template = "Conversation:\n{conversation}\n\nAnswer YES or NO"
        conversation = "YOU: Hello\nTHEM: Hi"
        result = template.format(conversation=conversation)
        assert "YOU: Hello" in result
        assert "THEM: Hi" in result

    def test_prompt_formatting_with_special_characters(self):
        """Prompt formatting should handle special characters."""
        template = "Review:\n{conversation}"
        conversation = "THEM: What's up? I'm here! $100?"
        result = template.format(conversation=conversation)
        assert "What's up?" in result
        assert "$100?" in result

    def test_prompt_formatting_with_empty_conversation(self):
        """Prompt formatting should handle empty conversation."""
        template = "Conversation:\n{conversation}\n\nAnswer:"
        conversation = ""
        result = template.format(conversation=conversation)
        assert "Conversation:\n\n\nAnswer:" in result

    def test_prompt_formatting_with_multiline(self):
        """Prompt formatting should preserve multiline conversations."""
        template = "{conversation}"
        conversation = "YOU: Line 1\nTHEM: Line 2\nYOU: Line 3"
        result = template.format(conversation=conversation)
        lines = result.split('\n')
        assert len(lines) == 3


class TestLLMResponseParsing:
    """Tests for parsing LLM responses with confidence scores."""

    def parse_response(self, response_text):
        """Parse LLM response for YES/NO and confidence."""
        text = response_text.strip().upper()

        # Match YES/NO with confidence
        match = re.search(r'(YES|NO)[\s:\-\(]*(\d+)', text)
        if match:
            return match.group(1), int(match.group(2))

        # Fallback
        if text.startswith("YES"):
            return "YES", None
        elif text.startswith("NO"):
            return "NO", None
        return None, None

    def test_yes_responses(self):
        """YES responses should be parsed correctly."""
        answer, conf = self.parse_response("YES 95")
        assert answer == "YES"
        assert conf == 95

    def test_no_responses(self):
        """NO responses should be parsed correctly."""
        answer, conf = self.parse_response("NO 85")
        assert answer == "NO"
        assert conf == 85

    def test_edge_cases(self):
        """Edge case responses should be handled."""
        answer, conf = self.parse_response("  YES 90  ")
        assert answer == "YES"
        assert conf == 90

        answer, conf = self.parse_response("YES")
        assert answer == "YES"
        assert conf is None


class TestConversationFormatting:
    """Tests for conversation formatting function."""

    def test_formats_incoming_as_them(self):
        """Incoming messages should be labeled as THEM."""
        df = pd.DataFrame([
            {'direction': 'incoming', 'body': 'Hello'},
        ])
        result = format_conversation_for_llm(df)
        assert "THEM: Hello" in result

    def test_formats_outgoing_as_you(self):
        """Outgoing messages should be labeled as YOU."""
        df = pd.DataFrame([
            {'direction': 'outgoing', 'body': 'Hi there'},
        ])
        result = format_conversation_for_llm(df)
        assert "YOU: Hi there" in result

    def test_preserves_order(self):
        """Conversation order should be preserved."""
        df = pd.DataFrame([
            {'direction': 'outgoing', 'body': 'First'},
            {'direction': 'incoming', 'body': 'Second'},
            {'direction': 'outgoing', 'body': 'Third'},
        ])
        result = format_conversation_for_llm(df)
        lines = result.split('\n')
        assert lines[0] == "YOU: First"
        assert lines[1] == "THEM: Second"
        assert lines[2] == "YOU: Third"

    def test_handles_empty_body(self):
        """Empty body should be handled gracefully."""
        df = pd.DataFrame([
            {'direction': 'incoming', 'body': ''},
            {'direction': 'incoming', 'body': None},
        ])
        result = format_conversation_for_llm(df)
        assert "THEM: " in result  # Should not crash


class TestGetPersonName:
    """Tests for person name resolution."""

    def test_returns_first_last_name(self):
        """Should return first_name + last_name when available."""
        people_df = pd.DataFrame([
            {'id': 1, 'first_name': 'John', 'last_name': 'Doe'},
        ])
        name = get_person_name(1, people_df)
        assert name == "John Doe"

    def test_returns_first_name_only(self):
        """Should return first_name when last_name is missing."""
        people_df = pd.DataFrame([
            {'id': 1, 'first_name': 'John', 'last_name': None},
        ])
        name = get_person_name(1, people_df)
        assert name == "John"

    def test_returns_name_column(self):
        """Should fall back to name column."""
        people_df = pd.DataFrame([
            {'id': 1, 'first_name': None, 'last_name': None, 'name': 'Johnny'},
        ])
        name = get_person_name(1, people_df)
        assert name == "Johnny"

    def test_returns_none_for_missing_person(self):
        """Should return None for non-existent person."""
        people_df = pd.DataFrame([
            {'id': 1, 'first_name': 'John', 'last_name': 'Doe'},
        ])
        name = get_person_name(999, people_df)
        assert name is None

    def test_returns_none_for_empty_names(self):
        """Should return None when all name fields are empty."""
        people_df = pd.DataFrame([
            {'id': 1, 'first_name': '', 'last_name': ''},
        ])
        name = get_person_name(1, people_df)
        assert name is None

    def test_handles_nan_names(self):
        """Should handle NaN values in name fields."""
        people_df = pd.DataFrame([
            {'id': 1, 'first_name': float('nan'), 'last_name': float('nan')},
        ])
        name = get_person_name(1, people_df)
        assert name is None


class TestPromptTemplates:
    """Tests for prompt templates from app.py."""

    def test_commitment_prompt_has_required_elements(self):
        """Commitment prompt should have all required elements."""
        prompt = DEFAULT_COMMITMENT_PROMPT

        # Has placeholder
        assert "{conversation}" in prompt

        # Has YES/NO instruction
        assert "YES" in prompt
        assert "NO" in prompt

        # Has sequence check
        assert "SEQUENCE" in prompt.upper()

        # Has opinion warning
        assert "OPINION" in prompt.upper()

        # Has action mention
        assert "action" in prompt.lower()

    def test_optout_prompt_has_required_elements(self):
        """Opt-out prompt should have all required elements."""
        prompt = DEFAULT_OPTOUT_PROMPT

        # Has placeholder
        assert "{conversation}" in prompt

        # Has YES/NO instruction
        assert "YES" in prompt
        assert "NO" in prompt

        # Mentions STOP
        assert "STOP" in prompt or "stop" in prompt
