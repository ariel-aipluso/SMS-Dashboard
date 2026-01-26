"""
Tests for opt-out detection functionality.
"""
import pytest
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestOptoutKeywordDetection:
    """Tests for keyword-based opt-out detection."""

    def get_stop_keywords(self):
        """Return the stop keywords used in the app."""
        return ['stop', 'unsubscribe', 'opt out', 'opt-out', 'remove', 'quit', 'cancel']

    def check_keyword_match(self, text, keywords):
        """Check if text contains any keyword."""
        if pd.isna(text):
            return False
        text_lower = str(text).lower()
        return any(kw in text_lower for kw in keywords)

    def test_standalone_stop_detected(self):
        """STOP by itself should be detected."""
        keywords = self.get_stop_keywords()
        assert self.check_keyword_match("STOP", keywords) == True
        assert self.check_keyword_match("stop", keywords) == True
        assert self.check_keyword_match("Stop", keywords) == True

    def test_stop_with_punctuation_detected(self):
        """STOP with punctuation should be detected."""
        keywords = self.get_stop_keywords()
        assert self.check_keyword_match("STOP!", keywords) == True
        assert self.check_keyword_match("Stop.", keywords) == True
        assert self.check_keyword_match("STOP!!!", keywords) == True

    def test_stop_in_sentence_detected(self):
        """Stop within a sentence should be detected by keywords (LLM filters false positives)."""
        keywords = self.get_stop_keywords()
        # These ARE detected by keyword matching (LLM verification filters them)
        assert self.check_keyword_match("Just stop spreading BULL", keywords) == True
        assert self.check_keyword_match("Can you stop by later?", keywords) == True
        assert self.check_keyword_match("Please stop texting", keywords) == True

    def test_unsubscribe_detected(self):
        """Unsubscribe variations should be detected."""
        keywords = self.get_stop_keywords()
        assert self.check_keyword_match("unsubscribe", keywords) == True
        assert self.check_keyword_match("UNSUBSCRIBE", keywords) == True
        assert self.check_keyword_match("Please unsubscribe me", keywords) == True

    def test_opt_out_variations_detected(self):
        """Opt out variations should be detected."""
        keywords = self.get_stop_keywords()
        assert self.check_keyword_match("opt out", keywords) == True
        assert self.check_keyword_match("opt-out", keywords) == True
        assert self.check_keyword_match("I want to opt out", keywords) == True

    def test_other_keywords_detected(self):
        """Other opt-out keywords should be detected."""
        keywords = self.get_stop_keywords()
        assert self.check_keyword_match("remove me", keywords) == True
        assert self.check_keyword_match("quit", keywords) == True
        assert self.check_keyword_match("cancel", keywords) == True

    def test_non_optout_not_detected(self):
        """Non opt-out messages should not be detected."""
        keywords = self.get_stop_keywords()
        assert self.check_keyword_match("Yes I'll be there", keywords) == False
        assert self.check_keyword_match("Count me in!", keywords) == False
        assert self.check_keyword_match("Thanks for the info", keywords) == False
        assert self.check_keyword_match("Hello", keywords) == False

    def test_empty_and_null_handling(self):
        """Empty and null values should not be detected."""
        keywords = self.get_stop_keywords()
        assert self.check_keyword_match("", keywords) == False
        assert self.check_keyword_match(None, keywords) == False
        assert self.check_keyword_match(pd.NA, keywords) == False


class TestOptoutPromptTemplate:
    """Tests for the opt-out prompt template."""

    def get_default_prompt(self):
        """Return the default opt-out prompt."""
        return """Look at this SMS conversation. Did the person (THEM) send "STOP" to unsubscribe?

Answer YES only if THEM sent a standalone message like:
- "STOP", "Stop", "stop" (by itself)
- "STOP!" or "Stop."

Answer NO if:
- "stop" appears inside a longer sentence (e.g., "stop spreading bull", "stop by later")
- They're having a conversation, even if frustrated
- They said "Over and out" or similar (not STOP)

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
        conversation = "YOU: Hello\nTHEM: STOP"
        formatted = prompt.format(conversation=conversation)
        assert "YOU: Hello" in formatted
        assert "THEM: STOP" in formatted
        assert "{conversation}" not in formatted

    def test_prompt_instructs_yes_no_response(self):
        """Prompt should ask for YES or NO response."""
        prompt = self.get_default_prompt()
        assert "YES" in prompt
        assert "NO" in prompt


class TestConversationFormatting:
    """Tests for conversation formatting for LLM."""

    def format_conversation(self, messages_df):
        """Format a conversation for LLM analysis."""
        lines = []
        for _, msg in messages_df.iterrows():
            direction = "THEM" if msg['direction'] == 'incoming' else "YOU"
            body = msg.get('body', '') or ''
            lines.append(f"{direction}: {body}")
        return "\n".join(lines)

    def test_incoming_labeled_as_them(self):
        """Incoming messages should be labeled as THEM."""
        df = pd.DataFrame([
            {'direction': 'incoming', 'body': 'Hello'}
        ])
        result = self.format_conversation(df)
        assert "THEM: Hello" in result

    def test_outgoing_labeled_as_you(self):
        """Outgoing messages should be labeled as YOU."""
        df = pd.DataFrame([
            {'direction': 'outgoing', 'body': 'Hi there'}
        ])
        result = self.format_conversation(df)
        assert "YOU: Hi there" in result

    def test_conversation_order_preserved(self):
        """Conversation order should be preserved."""
        df = pd.DataFrame([
            {'direction': 'outgoing', 'body': 'Hello'},
            {'direction': 'incoming', 'body': 'Hi'},
            {'direction': 'outgoing', 'body': 'How are you?'},
            {'direction': 'incoming', 'body': 'Good'},
        ])
        result = self.format_conversation(df)
        lines = result.split('\n')
        assert lines[0] == "YOU: Hello"
        assert lines[1] == "THEM: Hi"
        assert lines[2] == "YOU: How are you?"
        assert lines[3] == "THEM: Good"

    def test_empty_body_handled(self):
        """Empty message body should be handled gracefully."""
        df = pd.DataFrame([
            {'direction': 'incoming', 'body': None},
            {'direction': 'incoming', 'body': ''},
        ])
        result = self.format_conversation(df)
        assert "THEM: " in result  # Should not crash
