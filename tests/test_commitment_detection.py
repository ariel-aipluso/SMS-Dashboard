"""
Tests for commitment detection functionality.
"""
import pytest
import pandas as pd


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
    """Tests for the commitment prompt template."""

    def get_default_prompt(self):
        """Return the default commitment prompt."""
        return """Analyze this SMS conversation. Did the person (THEM) make a genuine commitment to take action?

Answer YES only if THEM clearly committed to:
- Attending an event ("I'll be there", "count me in", "I'm coming")
- Taking a specific action ("I will do it", "I'll help", "sign me up")
- Participating in something ("I'm in", "absolutely", "definitely joining")

Answer NO if:
- They just said "yes" to acknowledge a message without committing to action
- They're asking questions or being conversational
- The commitment is vague or unclear
- They said words like "yes" but in context of something else (e.g., "yes I got your message")

CONVERSATION:
{conversation}

Reply with ONLY one word: YES or NO"""

    def test_prompt_has_conversation_placeholder(self):
        """Prompt should have {conversation} placeholder."""
        prompt = self.get_default_prompt()
        assert "{conversation}" in prompt

    def test_prompt_format_works(self):
        """Prompt formatting should work correctly."""
        prompt = self.get_default_prompt()
        conversation = "YOU: Will you attend?\nTHEM: Yes, I'll be there!"
        formatted = prompt.format(conversation=conversation)
        assert "Will you attend?" in formatted
        assert "Yes, I'll be there!" in formatted
        assert "{conversation}" not in formatted

    def test_prompt_instructs_yes_no_response(self):
        """Prompt should ask for YES or NO response."""
        prompt = self.get_default_prompt()
        assert "YES" in prompt
        assert "NO" in prompt

    def test_prompt_covers_false_positive_cases(self):
        """Prompt should mention false positive scenarios to avoid."""
        prompt = self.get_default_prompt()
        assert "acknowledge" in prompt.lower() or "got your message" in prompt.lower()
