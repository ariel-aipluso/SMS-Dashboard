"""
Pytest configuration and shared fixtures for SMS Dashboard tests.
"""
import pytest
import pandas as pd


@pytest.fixture
def sample_people_df():
    """Create a sample people DataFrame for testing."""
    return pd.DataFrame([
        {'id': 1, 'name': 'Alice', 'phone': '555-1234', 'phone_number_type': 'mobile'},
        {'id': 2, 'name': 'Bob', 'phone': '555-5678', 'phone_number_type': 'mobile'},
        {'id': 3, 'name': 'Carol', 'phone': '555-9999', 'phone_number_type': 'landline'},
        {'id': 4, 'name': 'Dave', 'phone': '', 'phone_number_type': 'mobile'},
        {'id': 5, 'name': 'Eve', 'phone': 'nan', 'phone_number_type': 'mobile'},
    ])


@pytest.fixture
def sample_messages_df():
    """Create a sample messages DataFrame for testing."""
    return pd.DataFrame([
        {'person_id': 1, 'direction': 'outgoing', 'body': 'Hello, this is a reminder!', 'created_at': '2026-01-01 10:00:00'},
        {'person_id': 1, 'direction': 'incoming', 'body': 'Thanks!', 'created_at': '2026-01-01 10:05:00'},
        {'person_id': 2, 'direction': 'outgoing', 'body': 'Hello, this is a reminder!', 'created_at': '2026-01-01 10:00:00'},
        {'person_id': 2, 'direction': 'incoming', 'body': 'STOP', 'created_at': '2026-01-01 10:10:00'},
        {'person_id': 3, 'direction': 'outgoing', 'body': 'Hello, this is a reminder!', 'created_at': '2026-01-01 10:00:00'},
    ])


@pytest.fixture
def optout_messages_df():
    """Create messages with various opt-out scenarios."""
    return pd.DataFrame([
        # Clear opt-outs
        {'person_id': 1, 'direction': 'incoming', 'body': 'STOP', 'created_at': '2026-01-01'},
        {'person_id': 2, 'direction': 'incoming', 'body': 'stop', 'created_at': '2026-01-01'},
        {'person_id': 3, 'direction': 'incoming', 'body': 'Unsubscribe please', 'created_at': '2026-01-01'},
        # False positives (keyword match but not real opt-out)
        {'person_id': 4, 'direction': 'incoming', 'body': 'Just stop spreading BULL', 'created_at': '2026-01-01'},
        {'person_id': 5, 'direction': 'incoming', 'body': 'Can you stop by later?', 'created_at': '2026-01-01'},
        # Non opt-outs
        {'person_id': 6, 'direction': 'incoming', 'body': 'Yes I will be there', 'created_at': '2026-01-01'},
        {'person_id': 7, 'direction': 'incoming', 'body': 'Thanks for the info', 'created_at': '2026-01-01'},
    ])


@pytest.fixture
def commitment_messages_df():
    """Create messages with various commitment scenarios."""
    return pd.DataFrame([
        # Clear commitments
        {'person_id': 1, 'direction': 'incoming', 'body': "I'll be there!", 'created_at': '2026-01-01'},
        {'person_id': 2, 'direction': 'incoming', 'body': 'Count me in', 'created_at': '2026-01-01'},
        {'person_id': 3, 'direction': 'incoming', 'body': 'Yes, absolutely!', 'created_at': '2026-01-01'},
        # False positives (keyword match but not real commitment)
        {'person_id': 4, 'direction': 'incoming', 'body': 'Yes I got your message', 'created_at': '2026-01-01'},
        {'person_id': 5, 'direction': 'incoming', 'body': 'Yes, what do you want?', 'created_at': '2026-01-01'},
        # Non commitments
        {'person_id': 6, 'direction': 'incoming', 'body': 'Maybe later', 'created_at': '2026-01-01'},
        {'person_id': 7, 'direction': 'incoming', 'body': 'What time does it start?', 'created_at': '2026-01-01'},
    ])


@pytest.fixture
def multiplier_messages_df():
    """Create messages with various network multiplier scenarios."""
    return pd.DataFrame([
        # Clear multipliers
        {'person_id': 1, 'direction': 'incoming', 'body': "I'll invite my friends", 'created_at': '2026-01-01'},
        {'person_id': 2, 'direction': 'incoming', 'body': "Bringing my wife", 'created_at': '2026-01-01'},
        {'person_id': 3, 'direction': 'incoming', 'body': "I'll spread the word", 'created_at': '2026-01-01'},
        # False positives that were fixed
        {'person_id': 4, 'direction': 'incoming', 'body': 'Can you tell me more?', 'created_at': '2026-01-01'},
        {'person_id': 5, 'direction': 'incoming', 'body': 'My friend said hi', 'created_at': '2026-01-01'},
        # Non multipliers
        {'person_id': 6, 'direction': 'incoming', 'body': "I'll be there", 'created_at': '2026-01-01'},
        {'person_id': 7, 'direction': 'incoming', 'body': 'Count me in', 'created_at': '2026-01-01'},
    ])


@pytest.fixture
def stop_keywords():
    """Return the stop keywords used for opt-out detection."""
    return ['stop', 'unsubscribe', 'opt out', 'opt-out', 'remove', 'quit', 'cancel']


@pytest.fixture
def commitment_keywords():
    """Return the commitment keywords."""
    return [
        "yes", "i will", "i'll", "count me in", "sign me up",
        "i'm in", "absolutely", "definitely", "for sure",
        "commit", "committed", "attend", "participate", "join",
        "coming", "be there"
    ]


@pytest.fixture
def multiplier_keywords():
    """Return the network multiplier keywords."""
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
