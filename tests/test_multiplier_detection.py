"""
Tests for network multiplier detection functionality.
"""
import pytest
import pandas as pd


class TestMultiplierKeywordDetection:
    """Tests for keyword-based network multiplier detection."""

    def get_multiplier_keywords(self):
        """Return the multiplier keywords used in the app."""
        return [
            "invite", "invited", "inviting",
            "bring my", "bringing my", "bring a", "bringing a",
            "bring some", "bringing some",
            "+1", "plus one", "guest", "guests",
            "friends to", "family to",
            "neighbor", "neighbors",
            "spread the word", "pass it on", "let them know",
            "share this", "forward", "forwarding",
            "crew", "team", "group",
            "coworkers", "colleagues",
            "recruit", "get others", "reach out"
        ]

    def check_keyword_match(self, text, keywords):
        """Check if text contains any keyword."""
        if pd.isna(text):
            return False
        text_lower = str(text).lower()
        return any(kw in text_lower for kw in keywords)

    def test_invite_variations_detected(self):
        """Invite variations should be detected."""
        keywords = self.get_multiplier_keywords()
        assert self.check_keyword_match("I'll invite my friends", keywords) == True
        assert self.check_keyword_match("I invited some people", keywords) == True
        assert self.check_keyword_match("I'm inviting the team", keywords) == True

    def test_bring_with_context_detected(self):
        """'Bring' with proper context should be detected."""
        keywords = self.get_multiplier_keywords()
        assert self.check_keyword_match("I'll bring my wife", keywords) == True
        assert self.check_keyword_match("bringing my kids", keywords) == True
        assert self.check_keyword_match("bring a friend", keywords) == True
        assert self.check_keyword_match("bringing a coworker", keywords) == True

    def test_standalone_bring_not_detected(self):
        """Standalone 'bring' should NOT be detected (false positive prevention)."""
        keywords = self.get_multiplier_keywords()
        # These should NOT match because we use "bring my", "bring a", etc.
        assert self.check_keyword_match("What should I bring?", keywords) == False
        assert self.check_keyword_match("Bring it up later", keywords) == False

    def test_plus_one_detected(self):
        """Plus one variations should be detected."""
        keywords = self.get_multiplier_keywords()
        assert self.check_keyword_match("I'll bring a +1", keywords) == True
        assert self.check_keyword_match("plus one for me", keywords) == True
        assert self.check_keyword_match("Can I bring a guest?", keywords) == True
        assert self.check_keyword_match("bringing some guests", keywords) == True

    def test_specific_groups_detected(self):
        """Specific group references should be detected."""
        keywords = self.get_multiplier_keywords()
        # "friends to" is in the keywords, so this SHOULD match
        assert self.check_keyword_match("I'll tell my friends to come", keywords) == True  # "friends to" matched
        assert self.check_keyword_match("bringing friends to the event", keywords) == True  # "friends to" matched
        assert self.check_keyword_match("my family to join", keywords) == True
        assert self.check_keyword_match("I'll ask my neighbor", keywords) == True
        assert self.check_keyword_match("telling the neighbors", keywords) == True

    def test_spreading_word_detected(self):
        """Spreading the word phrases should be detected."""
        keywords = self.get_multiplier_keywords()
        assert self.check_keyword_match("I'll spread the word", keywords) == True
        assert self.check_keyword_match("I'll pass it on", keywords) == True
        assert self.check_keyword_match("Let them know I said hi", keywords) == True
        assert self.check_keyword_match("I'll share this with others", keywords) == True
        assert self.check_keyword_match("forwarding to my list", keywords) == True

    def test_group_references_detected(self):
        """Group/team references should be detected."""
        keywords = self.get_multiplier_keywords()
        assert self.check_keyword_match("I'll bring my crew", keywords) == True
        assert self.check_keyword_match("telling my team", keywords) == True
        assert self.check_keyword_match("sharing with my group", keywords) == True
        assert self.check_keyword_match("I'll tell my coworkers", keywords) == True
        assert self.check_keyword_match("asking my colleagues", keywords) == True

    def test_recruiting_language_detected(self):
        """Recruiting language should be detected."""
        keywords = self.get_multiplier_keywords()
        assert self.check_keyword_match("I'll recruit some volunteers", keywords) == True
        assert self.check_keyword_match("I'll get others to come", keywords) == True
        assert self.check_keyword_match("I'll reach out to people", keywords) == True

    def test_non_multiplier_not_detected(self):
        """Non-multiplier messages should not be detected."""
        keywords = self.get_multiplier_keywords()
        assert self.check_keyword_match("Yes I'll be there", keywords) == False
        assert self.check_keyword_match("Count me in", keywords) == False
        assert self.check_keyword_match("What time does it start?", keywords) == False
        assert self.check_keyword_match("Thanks for the info", keywords) == False

    def test_generic_tell_not_detected(self):
        """Generic 'tell' should NOT be detected (removed from keywords)."""
        keywords = self.get_multiplier_keywords()
        # "tell" alone was removed to prevent false positives
        assert self.check_keyword_match("Can you tell me more?", keywords) == False
        assert self.check_keyword_match("Tell me when", keywords) == False

    def test_generic_friend_not_detected(self):
        """Generic 'friend' should NOT be detected (only 'friends to' matches)."""
        keywords = self.get_multiplier_keywords()
        assert self.check_keyword_match("My friend said hi", keywords) == False
        assert self.check_keyword_match("A friendly reminder", keywords) == False

    def test_empty_and_null_handling(self):
        """Empty and null values should not be detected."""
        keywords = self.get_multiplier_keywords()
        assert self.check_keyword_match("", keywords) == False
        assert self.check_keyword_match(None, keywords) == False
        assert self.check_keyword_match(pd.NA, keywords) == False
