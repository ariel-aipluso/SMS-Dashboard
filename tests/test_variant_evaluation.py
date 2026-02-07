"""
Tests for the Message Variant Evaluation feature.
"""
import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import (
    parse_variant_eval_response,
    get_person_variant,
    aggregate_variant_metrics,
    DEFAULT_VARIANT_EVAL_PROMPT,
    VARIANT_EVAL_DIMENSIONS,
)


class TestVariantEvalResponseParsing:
    """Tests for parse_variant_eval_response() — now 6 dimensions (no REPLY/OPTOUT)."""

    def test_standard_six_line_response(self):
        response = (
            "CALL_COMMITMENT: YES 90 - THEM agreed to call their representative.\n"
            "CALL_FOLLOWTHROUGH: YES 85 - THEM said they called and left a voicemail.\n"
            "LETTER_COMMITMENT: NO 95 - No letter was requested.\n"
            "LETTER_FOLLOWTHROUGH: NO 99 - No letter commitment was made.\n"
            "MEETING_COMMITMENT: YES 88 - THEM agreed to attend the rally.\n"
            "MEETING_FOLLOWTHROUGH: NO 90 - THEM committed but never reported attending."
        )
        result = parse_variant_eval_response(response)
        assert result['call_commitment']['answer'] == 'YES'
        assert result['call_commitment']['confidence'] == 90
        assert result['call_followthrough']['answer'] == 'YES'
        assert result['call_followthrough']['confidence'] == 85
        assert result['letter_commitment']['answer'] == 'NO'
        assert result['letter_followthrough']['answer'] == 'NO'
        assert result['meeting_commitment']['answer'] == 'YES'
        assert result['meeting_commitment']['confidence'] == 88
        assert result['meeting_followthrough']['answer'] == 'NO'

    def test_missing_dimensions_default_to_no(self):
        response = "CALL_COMMITMENT: YES 90 - Agreed to call.\n"
        result = parse_variant_eval_response(response)
        assert result['call_commitment']['answer'] == 'YES'
        # All other action types default to NO / 0
        assert result['call_followthrough']['answer'] == 'NO'
        assert result['call_followthrough']['confidence'] == 0
        for action in ['letter', 'meeting']:
            assert result[f'{action}_commitment']['answer'] == 'NO'
            assert result[f'{action}_commitment']['confidence'] == 0
            assert result[f'{action}_followthrough']['answer'] == 'NO'
            assert result[f'{action}_followthrough']['confidence'] == 0

    def test_case_insensitive(self):
        response = (
            "call_commitment: yes 90 - They agreed to call.\n"
            "call_followthrough: no 85 - No follow-through.\n"
            "letter_commitment: no 95 - Not requested.\n"
            "letter_followthrough: no 99 - N/A.\n"
            "meeting_commitment: yes 88 - They agreed to attend.\n"
            "meeting_followthrough: no 80 - No evidence."
        )
        result = parse_variant_eval_response(response)
        assert result['call_commitment']['answer'] == 'YES'
        assert result['meeting_commitment']['answer'] == 'YES'

    def test_empty_response(self):
        result = parse_variant_eval_response("")
        for dim in VARIANT_EVAL_DIMENSIONS:
            assert result[dim.lower()]['answer'] == 'NO'
            assert result[dim.lower()]['confidence'] == 0
            assert result[dim.lower()]['summary'] is None

    def test_partial_response_with_no_summary(self):
        response = "CALL_COMMITMENT: YES 85\nMEETING_COMMITMENT: NO 90"
        result = parse_variant_eval_response(response)
        assert result['call_commitment']['answer'] == 'YES'
        assert result['call_commitment']['confidence'] == 85
        assert result['call_commitment']['summary'] is None
        assert result['meeting_commitment']['answer'] == 'NO'

    def test_extra_text_before_structured_lines(self):
        response = (
            "Let me analyze this conversation:\n\n"
            "CALL_COMMITMENT: NO 80 - No call was discussed.\n"
            "CALL_FOLLOWTHROUGH: NO 99 - N/A.\n"
            "LETTER_COMMITMENT: NO 80 - No letter discussed.\n"
            "LETTER_FOLLOWTHROUGH: NO 99 - N/A.\n"
            "MEETING_COMMITMENT: YES 90 - THEM committed to attend.\n"
            "MEETING_FOLLOWTHROUGH: NO 85 - No evidence of attending."
        )
        result = parse_variant_eval_response(response)
        assert result['meeting_commitment']['answer'] == 'YES'
        assert result['call_commitment']['answer'] == 'NO'

    def test_varied_separators(self):
        """Handle different dash types between confidence and summary."""
        response = (
            "CALL_COMMITMENT: YES 90 - THEM agreed to call.\n"
            "CALL_FOLLOWTHROUGH: NO 80 – No evidence.\n"
            "LETTER_COMMITMENT: NO 95 — Not requested.\n"
            "LETTER_FOLLOWTHROUGH: NO 99 - N/A.\n"
            "MEETING_COMMITMENT: NO 80 - Not discussed.\n"
            "MEETING_FOLLOWTHROUGH: NO 99 - N/A."
        )
        result = parse_variant_eval_response(response)
        assert result['call_commitment']['summary'] is not None
        assert result['call_followthrough']['summary'] is not None
        assert result['call_commitment']['answer'] == 'YES'

    def test_multiple_action_types_yes(self):
        """A conversation can have multiple action types committed."""
        response = (
            "CALL_COMMITMENT: YES 92 - Agreed to call representative.\n"
            "CALL_FOLLOWTHROUGH: YES 88 - Said they called.\n"
            "LETTER_COMMITMENT: YES 85 - Agreed to write a letter.\n"
            "LETTER_FOLLOWTHROUGH: NO 90 - Never reported sending.\n"
            "MEETING_COMMITMENT: YES 90 - Agreed to attend rally.\n"
            "MEETING_FOLLOWTHROUGH: YES 86 - Said they went."
        )
        result = parse_variant_eval_response(response)
        assert result['call_commitment']['answer'] == 'YES'
        assert result['letter_commitment']['answer'] == 'YES'
        assert result['meeting_commitment']['answer'] == 'YES'
        assert result['call_followthrough']['answer'] == 'YES'
        assert result['letter_followthrough']['answer'] == 'NO'
        assert result['meeting_followthrough']['answer'] == 'YES'

    def test_no_reply_or_optout_in_dimensions(self):
        """REPLY and OPTOUT should not be in VARIANT_EVAL_DIMENSIONS."""
        assert 'REPLY' not in VARIANT_EVAL_DIMENSIONS
        assert 'OPTOUT' not in VARIANT_EVAL_DIMENSIONS
        assert len(VARIANT_EVAL_DIMENSIONS) == 6


class TestGetPersonVariant:
    """Tests for get_person_variant()."""

    def test_basic_variant_mapping(self, variant_messages_df):
        variant_messages_df['created_at'] = pd.to_datetime(variant_messages_df['created_at'])
        result = get_person_variant(variant_messages_df)
        assert result[1] == 'Variant A'
        assert result[2] == 'Variant A'
        assert result[3] == 'Variant B'
        assert result[4] == 'Variant B'
        # Person 5 has empty variant, should not be in map
        assert 5 not in result

    def test_empty_variants_excluded(self):
        df = pd.DataFrame([
            {'person_id': 1, 'direction': 'outgoing', 'body': 'Hi', 'created_at': '2026-01-01', 'message_variant_name': ''},
            {'person_id': 2, 'direction': 'outgoing', 'body': 'Hi', 'created_at': '2026-01-01', 'message_variant_name': None},
        ])
        df['created_at'] = pd.to_datetime(df['created_at'])
        result = get_person_variant(df)
        assert len(result) == 0

    def test_first_outgoing_message_used(self):
        df = pd.DataFrame([
            {'person_id': 1, 'direction': 'outgoing', 'body': 'First', 'created_at': '2026-01-01 10:00:00', 'message_variant_name': 'V1'},
            {'person_id': 1, 'direction': 'outgoing', 'body': 'Second', 'created_at': '2026-01-01 11:00:00', 'message_variant_name': 'V2'},
        ])
        df['created_at'] = pd.to_datetime(df['created_at'])
        result = get_person_variant(df)
        assert result[1] == 'V1'

    def test_incoming_messages_ignored(self):
        df = pd.DataFrame([
            {'person_id': 1, 'direction': 'incoming', 'body': 'Reply', 'created_at': '2026-01-01', 'message_variant_name': 'V1'},
            {'person_id': 1, 'direction': 'outgoing', 'body': 'Outgoing', 'created_at': '2026-01-02', 'message_variant_name': 'V2'},
        ])
        df['created_at'] = pd.to_datetime(df['created_at'])
        result = get_person_variant(df)
        assert result[1] == 'V2'


def _make_eval_result(call_c='NO', call_f='NO',
                      letter_c='NO', letter_f='NO',
                      meeting_c='NO', meeting_f='NO',
                      confidence=90):
    """Helper to build an eval result dict (6 dimensions only)."""
    return {
        'call_commitment': {'answer': call_c, 'confidence': confidence},
        'call_followthrough': {'answer': call_f, 'confidence': confidence},
        'letter_commitment': {'answer': letter_c, 'confidence': confidence},
        'letter_followthrough': {'answer': letter_f, 'confidence': confidence},
        'meeting_commitment': {'answer': meeting_c, 'confidence': confidence},
        'meeting_followthrough': {'answer': meeting_f, 'confidence': confidence},
    }


class TestAggregateVariantMetrics:
    """Tests for aggregate_variant_metrics() — reply/optout from sets, not LLM."""

    def test_single_variant(self):
        eval_results = {
            1: _make_eval_result(call_c='YES'),
            2: _make_eval_result(),
        }
        person_variant_map = {1: 'V1', 2: 'V1'}
        df = aggregate_variant_metrics(eval_results, person_variant_map, {1, 2}, {2})
        assert len(df) == 1
        assert df.iloc[0]['Variant'] == 'V1'
        assert df.iloc[0]['Total Recipients'] == 2
        assert df.iloc[0]['Total Responders'] == 2
        # Person 2 opted out, so active = 1, reply = 1
        assert df.iloc[0]['_optout_rate'] == pytest.approx(50.0)

    def test_two_variants(self):
        eval_results = {
            1: _make_eval_result(meeting_c='YES'),
            2: _make_eval_result(),
        }
        person_variant_map = {1: 'V1', 2: 'V2'}
        df = aggregate_variant_metrics(eval_results, person_variant_map, {1}, set())
        assert len(df) == 2
        assert set(df['Variant'].tolist()) == {'V1', 'V2'}

    def test_action_type_rates_use_active_responders_denominator(self):
        # 1 responder, no opt-outs -> active = 1
        eval_results = {
            1: _make_eval_result(call_c='YES', call_f='YES',
                                 meeting_c='YES', meeting_f='NO'),
        }
        person_variant_map = {1: 'V1'}
        df = aggregate_variant_metrics(eval_results, person_variant_map, {1}, set())
        # Commitment rates: denom = active responders (1)
        assert df.iloc[0]['_call_commitment_rate'] == 100.0
        assert df.iloc[0]['_meeting_commitment_rate'] == 100.0
        assert df.iloc[0]['_letter_commitment_rate'] == 0.0
        # Follow-through rates: denom = corresponding commitment count
        assert df.iloc[0]['_call_followthrough_rate'] == 100.0  # 1/1 call commits
        assert df.iloc[0]['_meeting_followthrough_rate'] == 0.0  # 0/1 meeting commits

    def test_active_responders_calculated(self):
        # 3 responders, 1 opted out -> active = 2
        eval_results = {
            1: _make_eval_result(call_c='YES'),
            2: _make_eval_result(),  # opted out
            3: _make_eval_result(call_c='YES'),
        }
        person_variant_map = {1: 'V1', 2: 'V1', 3: 'V1'}
        df = aggregate_variant_metrics(eval_results, person_variant_map, {1, 2, 3}, {2})
        assert df.iloc[0]['_active_responders_rate'] == pytest.approx(66.7, abs=0.1)  # 2/3
        # Call commit: 2 commits / 2 active = 100%
        assert df.iloc[0]['_call_commitment_rate'] == 100.0

    def test_followthrough_uses_commitment_denominator(self):
        # 2 active responders, 1 committed to call, 1 followed through
        eval_results = {
            1: _make_eval_result(call_c='YES', call_f='YES'),
            2: _make_eval_result(),
        }
        person_variant_map = {1: 'V1', 2: 'V1'}
        df = aggregate_variant_metrics(eval_results, person_variant_map, {1, 2}, set())
        # Call commit: 1/2 active = 50%
        assert df.iloc[0]['_call_commitment_rate'] == 50.0
        # Call follow-through: 1/1 commit = 100%
        assert df.iloc[0]['_call_followthrough_rate'] == 100.0

    def test_followthrough_zero_when_no_commitments(self):
        eval_results = {
            1: _make_eval_result(),
        }
        person_variant_map = {1: 'V1'}
        df = aggregate_variant_metrics(eval_results, person_variant_map, {1}, set())
        # No commitments -> follow-through = 0
        assert df.iloc[0]['_call_followthrough_rate'] == 0.0
        assert df.iloc[0]['_letter_followthrough_rate'] == 0.0
        assert df.iloc[0]['_meeting_followthrough_rate'] == 0.0

    def test_responders_counted_correctly(self):
        eval_results = {
            1: _make_eval_result(),
        }
        person_variant_map = {1: 'V1', 2: 'V1'}
        # Only person 1 is a responder
        df = aggregate_variant_metrics(eval_results, person_variant_map, {1}, set())
        assert df.iloc[0]['Total Recipients'] == 2
        assert df.iloc[0]['Total Responders'] == 1

    def test_reply_and_optout_use_recipients_denominator(self):
        # 4 recipients, 2 responders, 1 replied, 1 opted out
        eval_results = {
            1: _make_eval_result(),
            2: _make_eval_result(),
        }
        person_variant_map = {1: 'V1', 2: 'V1', 3: 'V1', 4: 'V1'}
        df = aggregate_variant_metrics(eval_results, person_variant_map, {1, 2}, {2})
        # Reply rate: 1/4 = 25% (person 1 replied, person 2 opted out)
        assert df.iloc[0]['_reply_rate'] == 25.0
        # Opt-out rate: 1/4 = 25%
        assert df.iloc[0]['_optout_rate'] == 25.0

    def test_empty_data(self):
        df = aggregate_variant_metrics({}, {}, set(), set())
        assert df.empty


class TestVariantThresholds:
    """Tests for confidence thresholds in aggregation."""

    def test_call_commitment_threshold_85(self):
        eval_results = {1: _make_eval_result(call_c='YES', confidence=85)}
        df = aggregate_variant_metrics(eval_results, {1: 'V1'}, {1}, set())
        assert df.iloc[0]['_call_commitment_rate'] == 100.0

    def test_call_commitment_threshold_84_not_counted(self):
        eval_results = {1: _make_eval_result(call_c='YES', confidence=84)}
        df = aggregate_variant_metrics(eval_results, {1: 'V1'}, {1}, set())
        assert df.iloc[0]['_call_commitment_rate'] == 0.0

    def test_letter_followthrough_threshold_85(self):
        # Need letter_c=YES so denominator > 0 for follow-through rate
        eval_results = {1: _make_eval_result(letter_c='YES', letter_f='YES', confidence=85)}
        df = aggregate_variant_metrics(eval_results, {1: 'V1'}, {1}, set())
        assert df.iloc[0]['_letter_followthrough_rate'] == 100.0

    def test_meeting_commitment_threshold_85(self):
        eval_results = {1: _make_eval_result(meeting_c='YES', confidence=85)}
        df = aggregate_variant_metrics(eval_results, {1: 'V1'}, {1}, set())
        assert df.iloc[0]['_meeting_commitment_rate'] == 100.0

    def test_optout_from_set_not_llm(self):
        """Opt-out is determined by the optout_ids set, not LLM results."""
        eval_results = {1: _make_eval_result()}
        # Person 1 is in optout_ids set
        df = aggregate_variant_metrics(eval_results, {1: 'V1'}, {1}, {1})
        assert df.iloc[0]['_optout_rate'] == 100.0
        # Active = 0 since the only responder opted out
        assert df.iloc[0]['_active_responders_rate'] == 0.0

    def test_reply_from_set_not_llm(self):
        """Reply is determined by responder_ids set, not LLM results."""
        eval_results = {}  # No LLM results at all
        # Person 1 is a responder (not opted out), person 2 is not
        df = aggregate_variant_metrics(eval_results, {1: 'V1', 2: 'V1'}, {1}, set())
        assert df.iloc[0]['_reply_rate'] == 50.0  # 1/2 recipients


class TestVariantPromptTemplate:
    """Tests for the variant evaluation prompt."""

    def test_prompt_has_conversation_placeholder(self):
        assert '{conversation}' in DEFAULT_VARIANT_EVAL_PROMPT

    def test_prompt_mentions_all_action_types(self):
        prompt_upper = DEFAULT_VARIANT_EVAL_PROMPT.upper()
        assert 'CALL_COMMITMENT' in prompt_upper
        assert 'CALL_FOLLOWTHROUGH' in prompt_upper
        assert 'LETTER_COMMITMENT' in prompt_upper
        assert 'LETTER_FOLLOWTHROUGH' in prompt_upper
        assert 'MEETING_COMMITMENT' in prompt_upper
        assert 'MEETING_FOLLOWTHROUGH' in prompt_upper

    def test_prompt_does_not_mention_reply_or_optout_as_dimensions(self):
        """REPLY and OPTOUT are handled programmatically, not in the prompt."""
        # Should not have "REPLY:" or "OPTOUT:" as structured output lines
        assert 'REPLY: YES' not in DEFAULT_VARIANT_EVAL_PROMPT
        assert 'OPTOUT: YES' not in DEFAULT_VARIANT_EVAL_PROMPT

    def test_prompt_format_works(self):
        result = DEFAULT_VARIANT_EVAL_PROMPT.format(conversation="YOU: Hi\nTHEM: Hello")
        assert "YOU: Hi" in result
        assert "THEM: Hello" in result

    def test_prompt_has_example_format(self):
        assert 'Example:' in DEFAULT_VARIANT_EVAL_PROMPT
        assert 'CALL_COMMITMENT: YES' in DEFAULT_VARIANT_EVAL_PROMPT
        assert 'MEETING_COMMITMENT: YES' in DEFAULT_VARIANT_EVAL_PROMPT
