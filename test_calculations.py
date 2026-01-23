#!/usr/bin/env python3
"""
Test script to verify textable phone number and response rate calculations.
Tests both the OLD buggy behavior and NEW fixed behavior.
"""

import pandas as pd


def is_valid_phone(phone):
    """Check if a phone value is a valid, non-empty phone number (NEW LOGIC)."""
    if pd.isna(phone):
        return False
    s = str(phone).strip().lower()
    # Check for empty or placeholder values
    if s == '' or s in ('nan', 'none', 'null', 'n/a', 'na', '-'):
        return False
    # Must have at least some digits to be a phone number
    digits = ''.join(c for c in s if c.isdigit())
    return len(digits) >= 7  # Minimum reasonable phone number length


def calculate_metrics_old(people_df, messages_df, verbose=False):
    """OLD buggy calculation logic."""
    phone_column = None
    for phone_col in ['phone', 'phone_number', 'mobile', 'cell']:
        if phone_col in people_df.columns:
            phone_column = phone_col
            break

    if phone_column:
        people_with_phones = people_df[
            people_df[phone_column].notna() &
            (people_df[phone_column] != '') &
            (people_df[phone_column].astype(str).str.strip() != '')
        ]
        if 'phone_number_type' in people_df.columns:
            people_with_phones = people_with_phones[
                ~people_with_phones['phone_number_type'].isin(['landline', 'fixedVoip'])
            ]
        contactable_recipients = len(people_with_phones)
    else:
        contactable_recipients = len(people_df)

    all_responders = set(messages_df[messages_df['direction'] == 'incoming']['person_id'].unique())
    total_responders = len(all_responders)
    response_rate = (total_responders / contactable_recipients * 100) if contactable_recipients > 0 else 0

    return contactable_recipients, total_responders, response_rate


def calculate_metrics_new(people_df, messages_df, verbose=False):
    """NEW fixed calculation logic."""
    phone_col_candidates = ['phone', 'phone_number', 'mobile', 'cell']

    # Find which phone columns exist in the data
    available_phone_cols = [col for col in phone_col_candidates if col in people_df.columns]

    if available_phone_cols:
        # Check if person has a valid phone in ANY of the available columns
        def has_valid_phone(row):
            for col in available_phone_cols:
                if is_valid_phone(row[col]):
                    return True
            return False

        # Filter to people with at least one valid phone number
        people_with_phones = people_df[people_df.apply(has_valid_phone, axis=1)]

        if 'phone_number_type' in people_df.columns:
            people_with_phones = people_with_phones[
                ~people_with_phones['phone_number_type'].isin(['landline', 'fixedVoip'])
            ]
        contactable_recipients = len(people_with_phones)
        textable_people_ids = set(people_with_phones['id'].tolist())
    else:
        contactable_recipients = len(people_df)
        textable_people_ids = set(people_df['id'].tolist())

    # Only count responders who are in the textable list
    all_responders = set(messages_df[messages_df['direction'] == 'incoming']['person_id'].unique())
    textable_responders = all_responders & textable_people_ids
    total_responders = len(textable_responders)
    response_rate = (total_responders / contactable_recipients * 100) if contactable_recipients > 0 else 0

    if verbose:
        print(f"Available phone columns: {available_phone_cols}")
        print(f"Textable people IDs: {textable_people_ids}")
        print(f"All responders: {all_responders}")
        print(f"Textable responders: {textable_responders}")

    return contactable_recipients, total_responders, response_rate


def test_scenario(name, people_df, messages_df, expected_old, expected_new):
    """Run a test scenario comparing old and new logic."""
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")

    old_textable, old_responders, old_rate = calculate_metrics_old(people_df.copy(), messages_df.copy())
    new_textable, new_responders, new_rate = calculate_metrics_new(people_df.copy(), messages_df.copy())

    print(f"\nOLD (buggy) logic:")
    print(f"  Textable: {old_textable}, Responders: {old_responders}, Rate: {old_rate:.1f}%")
    print(f"  Expected: {expected_old}")

    print(f"\nNEW (fixed) logic:")
    print(f"  Textable: {new_textable}, Responders: {new_responders}, Rate: {new_rate:.1f}%")
    print(f"  Expected: {expected_new}")

    old_pass = (old_textable, old_responders, round(old_rate, 1)) == expected_old
    new_pass = (new_textable, new_responders, round(new_rate, 1)) == expected_new

    print(f"\nOLD matches expected: {'✅ YES' if old_pass else '❌ NO'}")
    print(f"NEW matches expected: {'✅ YES' if new_pass else '❌ NO'}")

    return new_pass


def main():
    print("="*70)
    print("SMS Dashboard Calculation Tests - OLD vs NEW Logic")
    print("="*70)

    all_passed = True

    # Test 1: Basic standard format (should be same for both)
    people_1 = pd.DataFrame([
        {'id': 1, 'name': 'Alice', 'phone': '555-1234', 'phone_number_type': 'mobile'},
        {'id': 2, 'name': 'Bob', 'phone': '555-5678', 'phone_number_type': 'mobile'},
        {'id': 3, 'name': 'Carol', 'phone': '555-9999', 'phone_number_type': 'landline'},
        {'id': 4, 'name': 'Dave', 'phone': '', 'phone_number_type': 'mobile'},
    ])
    messages_1 = pd.DataFrame([
        {'person_id': 1, 'direction': 'incoming', 'body': 'Hi!', 'created_at': '2026-01-01'},
    ])
    # OLD: 2 textable, 1 responder, 50%
    # NEW: 2 textable, 1 responder, 50%
    all_passed &= test_scenario("Basic standard format",
                                 people_1, messages_1,
                                 (2, 1, 50.0), (2, 1, 50.0))

    # Test 2: 'nan' string in phone (BUG FIX)
    people_2 = pd.DataFrame([
        {'id': 1, 'name': 'Alice', 'phone': '555-1234', 'phone_number_type': 'mobile'},
        {'id': 2, 'name': 'Bob', 'phone': 'nan', 'phone_number_type': 'mobile'},
    ])
    messages_2 = pd.DataFrame([
        {'person_id': 1, 'direction': 'incoming', 'body': 'Yes!', 'created_at': '2026-01-01'},
        {'person_id': 2, 'direction': 'incoming', 'body': 'Yes!', 'created_at': '2026-01-01'},
    ])
    # OLD (buggy): 2 textable (includes 'nan'), 2 responders, 100%
    # NEW (fixed): 1 textable, 1 responder (Bob excluded), 100%
    all_passed &= test_scenario("'nan' string in phone",
                                 people_2, messages_2,
                                 (2, 2, 100.0), (1, 1, 100.0))

    # Test 3: fixedVoip responder (BUG FIX - response rate > 100%)
    people_3 = pd.DataFrame([
        {'id': 1, 'name': 'Alice', 'phone': '555-1234', 'phone_number_type': 'mobile'},
        {'id': 2, 'name': 'Bob', 'phone': '555-5678', 'phone_number_type': 'fixedVoip'},
    ])
    messages_3 = pd.DataFrame([
        {'person_id': 1, 'direction': 'incoming', 'body': 'Yes!', 'created_at': '2026-01-01'},
        {'person_id': 2, 'direction': 'incoming', 'body': 'Yes!', 'created_at': '2026-01-01'},
    ])
    # OLD (buggy): 1 textable, 2 responders, 200% (impossible rate!)
    # NEW (fixed): 1 textable, 1 responder (Bob excluded), 100%
    all_passed &= test_scenario("fixedVoip responder",
                                 people_3, messages_3,
                                 (1, 2, 200.0), (1, 1, 100.0))

    # Test 4: Orphan responder (not in people list)
    people_4 = pd.DataFrame([
        {'id': 1, 'name': 'Alice', 'phone': '555-1234', 'phone_number_type': 'mobile'},
    ])
    messages_4 = pd.DataFrame([
        {'person_id': 1, 'direction': 'incoming', 'body': 'Yes!', 'created_at': '2026-01-01'},
        {'person_id': 999, 'direction': 'incoming', 'body': 'Hi!', 'created_at': '2026-01-01'},
    ])
    # OLD (buggy): 1 textable, 2 responders, 200%
    # NEW (fixed): 1 textable, 1 responder, 100%
    all_passed &= test_scenario("Orphan responder",
                                 people_4, messages_4,
                                 (1, 2, 200.0), (1, 1, 100.0))

    # Test 5: Both phone and phone_number columns (BUG FIX)
    people_5 = pd.DataFrame([
        {'id': 1, 'name': 'Alice', 'phone': '555-1234', 'phone_number': '', 'phone_number_type': 'mobile'},
        {'id': 2, 'name': 'Bob', 'phone': '', 'phone_number': '555-5678', 'phone_number_type': 'mobile'},
    ])
    messages_5 = pd.DataFrame([
        {'person_id': 1, 'direction': 'incoming', 'body': 'Yes!', 'created_at': '2026-01-01'},
        {'person_id': 2, 'direction': 'incoming', 'body': 'Yes!', 'created_at': '2026-01-01'},
    ])
    # OLD (buggy): 1 textable (only Alice - picks 'phone' column first), 2 responders, 200%
    # NEW (fixed): 2 textable (checks ALL phone columns per person), 2 responders, 100%
    all_passed &= test_scenario("Both phone columns",
                                 people_5, messages_5,
                                 (1, 2, 200.0), (2, 2, 100.0))

    # Test 6: N/A and None strings
    people_6 = pd.DataFrame([
        {'id': 1, 'name': 'Alice', 'phone': '555-1234', 'phone_number_type': 'mobile'},
        {'id': 2, 'name': 'Bob', 'phone': 'N/A', 'phone_number_type': 'mobile'},
        {'id': 3, 'name': 'Carol', 'phone': 'None', 'phone_number_type': 'mobile'},
    ])
    messages_6 = pd.DataFrame([
        {'person_id': 1, 'direction': 'incoming', 'body': 'Yes!', 'created_at': '2026-01-01'},
    ])
    # OLD (buggy): 3 textable (includes N/A, None), 1 responder, 33.3%
    # NEW (fixed): 1 textable, 1 responder, 100%
    all_passed &= test_scenario("N/A and None strings",
                                 people_6, messages_6,
                                 (3, 1, 33.3), (1, 1, 100.0))

    # Test 7: Phone numbers as floats
    people_7 = pd.DataFrame([
        {'id': 1, 'name': 'Alice', 'phone': 5551234.0, 'phone_number_type': 'mobile'},
        {'id': 2, 'name': 'Bob', 'phone': 5555678.0, 'phone_number_type': 'mobile'},
    ])
    messages_7 = pd.DataFrame([
        {'person_id': 1, 'direction': 'incoming', 'body': 'Yes!', 'created_at': '2026-01-01'},
    ])
    # Both should work - floats are valid phone numbers
    all_passed &= test_scenario("Phone as floats",
                                 people_7, messages_7,
                                 (2, 1, 50.0), (2, 1, 50.0))

    # Test 8: Short invalid phone numbers
    people_8 = pd.DataFrame([
        {'id': 1, 'name': 'Alice', 'phone': '555-1234', 'phone_number_type': 'mobile'},
        {'id': 2, 'name': 'Bob', 'phone': '123', 'phone_number_type': 'mobile'},  # Too short
        {'id': 3, 'name': 'Carol', 'phone': 'x', 'phone_number_type': 'mobile'},  # No digits
    ])
    messages_8 = pd.DataFrame([
        {'person_id': 1, 'direction': 'incoming', 'body': 'Yes!', 'created_at': '2026-01-01'},
    ])
    # OLD (buggy): 3 textable, 1 responder, 33.3%
    # NEW (fixed): 1 textable (only valid phone), 1 responder, 100%
    all_passed &= test_scenario("Short/invalid phones",
                                 people_8, messages_8,
                                 (3, 1, 33.3), (1, 1, 100.0))

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if all_passed:
        print("✅ All NEW logic tests passed!")
        print("\nFixes applied:")
        print("1. Added is_valid_phone() to properly validate phone numbers")
        print("2. String 'nan', 'N/A', 'None', etc. are now correctly excluded")
        print("3. Short/invalid phone numbers are now correctly excluded")
        print("4. Phone columns are checked per-person (not just one column)")
        print("5. Responders are filtered to only include textable people")
        print("6. Response rate can no longer exceed 100%")
    else:
        print("❌ Some NEW logic tests failed - check implementation")


if __name__ == '__main__':
    main()
