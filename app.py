import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import importlib.util
import re
import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Check LLM availability without importing (saves ~100-150MB on idle)
ANTHROPIC_AVAILABLE = importlib.util.find_spec("anthropic") is not None
OPENAI_AVAILABLE = importlib.util.find_spec("openai") is not None


def get_centralized_api_key(provider: str) -> str | None:
    """Get centralized API key from secrets or environment variables.

    Resolution order:
    1. Streamlit secrets (`.streamlit/secrets.toml`)
    2. Environment variable (ANTHROPIC_API_KEY or OPENAI_API_KEY)
    """
    env_var = f"{provider.upper()}_API_KEY"

    # Try environment variable first (works in deployment)
    if os.getenv(env_var):
        return os.getenv(env_var)

    # Try Streamlit secrets (local development)
    try:
        return st.secrets.get("api_keys", {}).get(provider.lower())
    except Exception:
        return None


st.set_page_config(
    page_title="SMS Campaign Analytics",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def get_version_date():
    """Get the last git commit date for version display."""
    import subprocess
    try:
        result = subprocess.run(
            ['git', 'log', '-1', '--format=%cd', '--date=format:%B %d, %Y'],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return "January 26, 2026"  # Fallback date

st.title("📱 SMS Campaign Analytics Dashboard")
st.markdown("*Interactive SMS Campaign Analysis Tool*")
st.caption(f"Version: {get_version_date()}")

# File upload section
with st.sidebar.expander("📂 Data Upload", expanded=True):
    messages_files = st.file_uploader(
        "Upload Messages CSV(s)", type=['csv'], accept_multiple_files=True,
        help="Upload one or more message exports (Campaign, Flow, All Messages). Files with the same message IDs will be merged."
    )
    people_file = st.file_uploader("Upload People CSV", type=['csv'])

def normalize_phone(phone):
    """Normalize phone number for comparison - handles floats, +1, formatting."""
    if pd.isna(phone):
        return None
    s = str(phone)
    # Remove .0 suffix from float conversion
    if s.endswith('.0'):
        s = s[:-2]
    # Remove all formatting
    s = s.replace('+', '').replace('-', '').replace(' ', '').replace('(', '').replace(')', '')
    # Normalize to 10 digits (remove leading 1 for US numbers)
    if len(s) == 11 and s.startswith('1'):
        s = s[1:]
    return s


def is_valid_phone(phone):
    """Check if a phone value is a valid, non-empty phone number."""
    if pd.isna(phone):
        return False
    s = str(phone).strip().lower()
    # Check for empty or placeholder values
    if s == '' or s in ('nan', 'none', 'null', 'n/a', 'na', '-'):
        return False
    # Must have at least some digits to be a phone number
    digits = ''.join(c for c in s if c.isdigit())
    return len(digits) >= 7  # Minimum reasonable phone number length

def get_person_name(person_id, people_df):
    """Get display name for a person, checking first_name/last_name before name column."""
    # Handle type mismatches by trying both direct comparison and string comparison
    person_row = people_df[people_df['id'] == person_id]
    if person_row.empty:
        # Try string comparison as fallback for type mismatches
        person_id_str = str(person_id)
        person_row = people_df[people_df['id'].astype(str) == person_id_str]
    if not person_row.empty:
        # Helper to check if value is valid (not NaN and not string 'nan')
        def is_valid(val):
            return pd.notna(val) and str(val).lower() != 'nan' and str(val).strip() != ''

        # Try first_name/last_name first (more specific)
        if 'first_name' in people_df.columns or 'last_name' in people_df.columns:
            first_name = person_row['first_name'].iloc[0] if 'first_name' in people_df.columns else None
            last_name = person_row['last_name'].iloc[0] if 'last_name' in people_df.columns else None
            parts = []
            if is_valid(first_name):
                parts.append(str(first_name))
            if is_valid(last_name):
                parts.append(str(last_name))
            if parts:
                return ' '.join(parts)
        # Fall back to name column
        if 'name' in people_df.columns:
            name_val = person_row['name'].iloc[0]
            if is_valid(name_val):
                return str(name_val)
    return None

def reconcile_message_files(uploaded_files):
    """Load and reconcile multiple message CSV files using 'id' as primary key."""
    if not uploaded_files:
        return pd.DataFrame(), []

    file_info = []
    dataframes = []

    # Only keep columns the app actually uses (flow exports have ~130MB of unused state_data)
    _msg_keep_cols = {'id', 'person_id', 'direction', 'body', 'created_at', 'sent_at', 'status',
                      'message_variant_name', 'message_node_name', 'conversation_id', 'from', 'to',
                      'phone_number_type', 'automation_id', 'automation_name', 'broadcast_id', 'broadcast_name'}

    for f in uploaded_files:
        chunks = pd.read_csv(f, chunksize=50000, low_memory=False)
        df = pd.concat(chunks, ignore_index=True)
        df = df[[c for c in df.columns if c in _msg_keep_cols]]

        cols = set(df.columns)
        has_variant = 'message_variant_name' in cols
        has_person_id = 'person_id' in cols
        has_automation_name = 'automation_name' in cols

        if has_variant and has_automation_name:
            export_type = 'Flow'
        elif has_variant:
            export_type = 'Campaign'
        elif has_person_id:
            export_type = 'All Messages'
        else:
            export_type = 'Unknown'

        file_info.append({
            'filename': f.name,
            'rows': len(df),
            'export_type': export_type,
        })
        dataframes.append(df)

    if len(dataframes) == 1:
        return dataframes[0], file_info

    has_id_col = all('id' in df.columns for df in dataframes)

    if has_id_col:
        base = dataframes[0].set_index('id')
        for df in dataframes[1:]:
            other = df.set_index('id')
            base = base.combine_first(other)
        messages_df = base.reset_index()
    else:
        messages_df = pd.concat(dataframes, ignore_index=True)

    return messages_df, file_info

# LLM-based verification using Claude or OpenAI (conversation-aware)
def format_conversation_for_llm(person_messages_df):
    """Format a person's conversation for LLM analysis."""
    lines = []
    for _, msg in person_messages_df.iterrows():
        direction = "THEM" if msg['direction'] == 'incoming' else "YOU"
        body = msg.get('body', '') or ''
        lines.append(f"{direction}: {body}")
    return "\n".join(lines)

PROMPT_VERSION = "v19"  # Increment this when changing the prompt to bust cache

# Default prompts for AI verification
DEFAULT_COMMITMENT_PROMPT = """Did THEM agree to take an action they were asked to do?

Actions include: making a call, attending an event, gathering others.
Also count: alternative actions if they already completed the original ask.

CRITICAL RULE - Check the SEQUENCE:
1. Find where YOU asked THEM to take action (call, attend, etc.)
2. Did THEM respond AFTER that action request?
3. If THEM's last message was BEFORE the action request, answer NO

COMMON MISTAKE TO AVOID:
- YOU: "Do you think X is fair?"
- THEM: "No, it's not fair!" (this is an OPINION, not a commitment)
- YOU: "Would you make a call?"
- THEM: [no response]
This is NO 95 - they only answered the opinion question, never the action request.

YES = They responded to an action request with agreement
NO = They only expressed opinions, never responded to action request, or declined

Score (0-100) = Your confidence. Be consistent: similar conversations should get similar scores.

Note: Unanswered follow-ups about timing after committing don't cancel a commitment.

CONVERSATION:
{conversation}

RESPOND WITH: YES ## or NO ## followed by a one-sentence summary explaining why.
Example: YES 92 - THEM agreed to attend the rally after being asked to join.
Example: NO 95 - THEM only answered the opinion question; they never responded to the action request."""

DEFAULT_VARIANT_EVAL_PROMPT = """Analyze this SMS conversation for action commitments and follow-through. A single conversation may have MULTIPLE action types (e.g., someone commits to both a call AND a meeting).

CONVERSATION:
{conversation}

For each dimension, answer YES or NO, give a confidence score (0-100), and a brief explanation justifying your assessment.

DIMENSIONS:
1. CALL_COMMITMENT: Did THEM commit to making a phone call AFTER being asked?
   - Only count if YOU requested a call and THEM agreed. Opinions are NOT commitments.
2. CALL_FOLLOWTHROUGH: Did THEM report actually making the call?
   - Evidence like "I called", "I just spoke with them", "I left a message".
   - NO if no call commitment, or committed but never reported calling.
3. LETTER_COMMITMENT: Did THEM commit to writing a letter or sending an email AFTER being asked?
   - Only count if YOU requested a letter/email and THEM agreed.
4. LETTER_FOLLOWTHROUGH: Did THEM report actually sending the letter or email?
   - Evidence like "I sent the email", "I mailed the letter", "letter is in the mail".
   - NO if no letter commitment, or committed but never reported sending.
5. MEETING_COMMITMENT: Did THEM commit to attending an in-person meeting or event AFTER being asked?
   - Only count if YOU requested attendance and THEM agreed.
6. MEETING_FOLLOWTHROUGH: Did THEM report actually attending the meeting or event?
   - Evidence like "I went to the meeting", "I was there", "just got back from the event".
   - NO if no meeting commitment, or committed but never reported attending.

RESPOND in this exact format (6 lines):
CALL_COMMITMENT: YES|NO <confidence> - <explanation>
CALL_FOLLOWTHROUGH: YES|NO <confidence> - <explanation>
LETTER_COMMITMENT: YES|NO <confidence> - <explanation>
LETTER_FOLLOWTHROUGH: YES|NO <confidence> - <explanation>
MEETING_COMMITMENT: YES|NO <confidence> - <explanation>
MEETING_FOLLOWTHROUGH: YES|NO <confidence> - <explanation>

Example:
CALL_COMMITMENT: YES 90 - THEM agreed to call their representative after YOU asked.
CALL_FOLLOWTHROUGH: YES 85 - THEM said "I just called and left a voicemail."
LETTER_COMMITMENT: NO 95 - No letter or email was requested in the conversation.
LETTER_FOLLOWTHROUGH: NO 99 - No letter commitment was made.
MEETING_COMMITMENT: YES 88 - THEM agreed to attend the rally after being asked.
MEETING_FOLLOWTHROUGH: NO 90 - THEM committed to attend but never reported going."""

# Commitment analysis functions
def analyze_commitments_with_anthropic(conversations: dict, api_key: str, custom_prompt: str = None, progress_callback=None) -> dict:
    """
    Use Claude to analyze conversations and detect genuine commitments to action.
    """
    if not ANTHROPIC_AVAILABLE or not api_key:
        return {}

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    results = {}
    prompt_template = custom_prompt if custom_prompt else DEFAULT_COMMITMENT_PROMPT

    def _eval_one(person_id, conversation):
        prompt = prompt_template.format(conversation=conversation)
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = response.content[0].text.strip()
            response_upper = response_text.upper()

            match = re.search(r'(YES|NO)[\s:\-\(]*(\d+)', response_upper)
            if match:
                answer = match.group(1)
                confidence = int(match.group(2))
                is_committed = (answer == "YES" and confidence >= 85)

                summary_match = re.search(r'(?:YES|NO)\s*\d+\s*[-–—]\s*(.+)', response_text, re.IGNORECASE)
                summary = summary_match.group(1).strip() if summary_match else None

                return person_id, {"is_committed": is_committed, "confidence": confidence, "raw_answer": answer, "summary": summary}
            else:
                yes_match = re.search(r'\bYES\b', response_upper)
                no_match = re.search(r'\bNO\b', response_upper)

                if yes_match and not no_match:
                    return person_id, {"is_committed": False, "confidence": 50, "raw_answer": "YES", "note": "no_confidence", "summary": None}
                elif no_match:
                    return person_id, {"is_committed": False, "confidence": 50, "raw_answer": "NO", "note": "no_confidence", "summary": None}
                else:
                    return person_id, {"is_committed": False, "confidence": 0, "raw_answer": "NO", "note": "parse_failed", "summary": None}

        except Exception as e:
            return person_id, {"is_committed": "unknown", "error": str(e)}

    completed = 0
    total = len(conversations)
    with ThreadPoolExecutor(max_workers=VARIANT_EVAL_MAX_WORKERS) as executor:
        futures = {executor.submit(_eval_one, pid, conv): pid for pid, conv in conversations.items()}
        for future in as_completed(futures):
            person_id, result = future.result()
            results[person_id] = result
            completed += 1
            if progress_callback:
                progress_callback(completed, total)

    return results

def analyze_commitments_with_openai(conversations: dict, api_key: str, custom_prompt: str = None, progress_callback=None) -> dict:
    """
    Use OpenAI to analyze conversations and detect genuine commitments to action.
    """
    if not OPENAI_AVAILABLE or not api_key:
        return {}

    import openai
    client = openai.OpenAI(api_key=api_key)
    results = {}
    prompt_template = custom_prompt if custom_prompt else DEFAULT_COMMITMENT_PROMPT

    def _eval_one(person_id, conversation):
        prompt = prompt_template.format(conversation=conversation)
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = response.choices[0].message.content.strip()
            response_upper = response_text.upper()

            match = re.search(r'(YES|NO)[\s:\-\(]*(\d+)', response_upper)
            if match:
                answer = match.group(1)
                confidence = int(match.group(2))
                is_committed = (answer == "YES" and confidence >= 85)

                summary_match = re.search(r'(?:YES|NO)\s*\d+\s*[-–—]\s*(.+)', response_text, re.IGNORECASE)
                summary = summary_match.group(1).strip() if summary_match else None

                return person_id, {"is_committed": is_committed, "confidence": confidence, "raw_answer": answer, "summary": summary}
            else:
                yes_match = re.search(r'\bYES\b', response_upper)
                no_match = re.search(r'\bNO\b', response_upper)

                if yes_match and not no_match:
                    return person_id, {"is_committed": False, "confidence": 50, "raw_answer": "YES", "note": "no_confidence", "summary": None}
                elif no_match:
                    return person_id, {"is_committed": False, "confidence": 50, "raw_answer": "NO", "note": "no_confidence", "summary": None}
                else:
                    return person_id, {"is_committed": False, "confidence": 0, "raw_answer": "NO", "note": "parse_failed", "summary": None}

        except Exception as e:
            return person_id, {"is_committed": "unknown", "error": str(e)}

    completed = 0
    total = len(conversations)
    with ThreadPoolExecutor(max_workers=VARIANT_EVAL_MAX_WORKERS) as executor:
        futures = {executor.submit(_eval_one, pid, conv): pid for pid, conv in conversations.items()}
        for future in as_completed(futures):
            person_id, result = future.result()
            results[person_id] = result
            completed += 1
            if progress_callback:
                progress_callback(completed, total)

    return results

# Variant evaluation functions
VARIANT_EVAL_DIMENSIONS = [
    'CALL_COMMITMENT', 'CALL_FOLLOWTHROUGH',
    'LETTER_COMMITMENT', 'LETTER_FOLLOWTHROUGH',
    'MEETING_COMMITMENT', 'MEETING_FOLLOWTHROUGH',
]

def parse_variant_eval_response(response_text: str) -> dict:
    """Parse the 8-dimensional variant evaluation response."""
    result = {}
    for dimension in VARIANT_EVAL_DIMENSIONS:
        pattern = rf'{dimension}:\s*(YES|NO)\s*(\d+)\s*[-–—]\s*(.+)'
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            result[dimension.lower()] = {
                'answer': match.group(1).upper(),
                'confidence': int(match.group(2)),
                'summary': match.group(3).strip()
            }
        else:
            # Fallback: look for just YES/NO on a line mentioning this dimension
            line_pattern = rf'{dimension}:\s*(YES|NO)\s*(\d+)?'
            line_match = re.search(line_pattern, response_text, re.IGNORECASE)
            if line_match:
                result[dimension.lower()] = {
                    'answer': line_match.group(1).upper(),
                    'confidence': int(line_match.group(2)) if line_match.group(2) else 50,
                    'summary': None
                }
            else:
                result[dimension.lower()] = {
                    'answer': 'NO',
                    'confidence': 0,
                    'summary': None
                }
    return result

def _variant_eval_fallback():
    """Return default fallback result for all variant eval dimensions."""
    return {
        dim.lower(): {'answer': 'NO', 'confidence': 0, 'summary': None}
        for dim in VARIANT_EVAL_DIMENSIONS
    }

VARIANT_EVAL_MAX_WORKERS = 10
VARIANT_EVAL_MAX_RETRIES = 3

def _api_call_with_retry(fn, max_retries=VARIANT_EVAL_MAX_RETRIES):
    """Call fn() with exponential backoff on rate limit / server errors."""
    for attempt in range(max_retries):
        try:
            return fn(), None
        except Exception as e:
            err_str = str(e).lower()
            is_retryable = any(k in err_str for k in ['rate', '429', '529', 'overloaded', 'timeout', 'capacity'])
            if is_retryable and attempt < max_retries - 1:
                time.sleep(2 ** attempt + 1)
                continue
            return None, str(e)

def analyze_variant_eval_with_anthropic(conversations: dict, api_key: str, custom_prompt: str = None, progress_callback=None) -> dict:
    """Use Claude to evaluate conversations concurrently for variant comparison."""
    if not ANTHROPIC_AVAILABLE or not api_key:
        return {}

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    prompt_template = custom_prompt if custom_prompt else DEFAULT_VARIANT_EVAL_PROMPT

    def _eval_one(person_id, conversation):
        prompt = prompt_template.format(conversation=conversation)
        def _call():
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=450,
                messages=[{"role": "user", "content": prompt}]
            )
            return parse_variant_eval_response(response.content[0].text.strip())
        result, err = _api_call_with_retry(_call)
        if err:
            return person_id, None, err
        return person_id, result, None

    results = {}
    errors = []
    completed = 0
    total = len(conversations)
    with ThreadPoolExecutor(max_workers=VARIANT_EVAL_MAX_WORKERS) as executor:
        futures = {executor.submit(_eval_one, pid, conv): pid for pid, conv in conversations.items()}
        for future in as_completed(futures):
            person_id, result, err = future.result()
            if err:
                results[person_id] = _variant_eval_fallback()
                errors.append((person_id, err))
            else:
                results[person_id] = result
            completed += 1
            if progress_callback:
                progress_callback(completed, total)

    if errors:
        sample = errors[0][1]
        st.warning(f"Variant eval API errors for {len(errors)} conversation(s) after {VARIANT_EVAL_MAX_RETRIES} retries. Sample error: {sample}")
    return results

def analyze_variant_eval_with_openai(conversations: dict, api_key: str, custom_prompt: str = None, progress_callback=None) -> dict:
    """Use OpenAI to evaluate conversations concurrently for variant comparison."""
    if not OPENAI_AVAILABLE or not api_key:
        return {}

    import openai
    client = openai.OpenAI(api_key=api_key)
    prompt_template = custom_prompt if custom_prompt else DEFAULT_VARIANT_EVAL_PROMPT

    def _eval_one(person_id, conversation):
        prompt = prompt_template.format(conversation=conversation)
        def _call():
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=450,
                messages=[{"role": "user", "content": prompt}]
            )
            return parse_variant_eval_response(response.choices[0].message.content.strip())
        result, err = _api_call_with_retry(_call)
        if err:
            return person_id, None, err
        return person_id, result, None

    results = {}
    errors = []
    completed = 0
    total = len(conversations)
    with ThreadPoolExecutor(max_workers=VARIANT_EVAL_MAX_WORKERS) as executor:
        futures = {executor.submit(_eval_one, pid, conv): pid for pid, conv in conversations.items()}
        for future in as_completed(futures):
            person_id, result, err = future.result()
            if err:
                results[person_id] = _variant_eval_fallback()
                errors.append((person_id, err))
            else:
                results[person_id] = result
            completed += 1
            if progress_callback:
                progress_callback(completed, total)

    if errors:
        sample = errors[0][1]
        st.warning(f"Variant eval API errors for {len(errors)} conversation(s) after {VARIANT_EVAL_MAX_RETRIES} retries. Sample error: {sample}")
    return results

def get_person_variant(messages_df):
    """Map person_id -> message_variant_name from initial broadcast messages only.
    If message_variant_name column is missing, assigns all recipients to 'All'."""
    # Base filter: outgoing, exclude undelivered/failed messages
    base_mask = messages_df['direction'] == 'outgoing'
    if 'status' in messages_df.columns:
        base_mask = base_mask & (~messages_df['status'].fillna('').str.lower().isin(['undelivered', 'failed']))

    if 'message_variant_name' not in messages_df.columns or messages_df['message_variant_name'].dropna().astype(str).str.strip().replace('', pd.NA).dropna().empty:
        # No variant info — assign all outgoing-message recipients to a single group
        outgoing = messages_df[base_mask]
        if 'message_node_name' in outgoing.columns:
            outgoing = outgoing[outgoing['message_node_name'].isna() | (outgoing['message_node_name'].astype(str).str.strip() == '')]
        person_ids = outgoing['person_id'].dropna().unique()
        return {pid: 'All' for pid in person_ids}

    mask = (
        base_mask &
        (messages_df['message_variant_name'].notna()) &
        (messages_df['message_variant_name'].astype(str).str.strip() != '')
    )
    # Exclude AI flow responses (message_node_name set) to only count broadcast recipients
    if 'message_node_name' in messages_df.columns:
        mask = mask & (messages_df['message_node_name'].isna() | (messages_df['message_node_name'].astype(str).str.strip() == ''))
    outgoing = messages_df[mask].sort_values('created_at')
    return outgoing.drop_duplicates('person_id', keep='first').set_index('person_id')['message_variant_name'].to_dict()

ACTION_TYPES = ['call', 'letter', 'meeting']

def aggregate_variant_metrics(eval_results: dict, person_variant_map: dict, responder_ids: set, optout_ids: set, person_reply_counts: dict = None) -> pd.DataFrame:
    """Aggregate per-conversation LLM evaluation results by variant into a summary DataFrame."""
    OTHER_THRESHOLD = 85
    if person_reply_counts is None:
        person_reply_counts = {}

    def _empty_counters():
        d = {'recipients': 0, 'responders': 0, 'reply': 0, 'optout': 0, 'active_reply_counts': []}
        for action in ACTION_TYPES:
            d[f'{action}_commitment'] = 0
            d[f'{action}_followthrough'] = 0
        return d

    variant_data = {}
    for person_id, variant_name in person_variant_map.items():
        if variant_name not in variant_data:
            variant_data[variant_name] = _empty_counters()
        # Everyone in person_variant_map received an outgoing message with this variant
        variant_data[variant_name]['recipients'] += 1
        if person_id in optout_ids:
            variant_data[variant_name]['optout'] += 1
        if person_id in responder_ids:
            variant_data[variant_name]['responders'] += 1
            if person_id not in optout_ids:
                variant_data[variant_name]['reply'] += 1
                # Track reply counts for active responders (not opted out)
                count = person_reply_counts.get(person_id) or person_reply_counts.get(str(person_id), 0)
                if count > 0:
                    variant_data[variant_name]['active_reply_counts'].append(count)

        if person_id in eval_results:
            r = eval_results[person_id]
            for action in ACTION_TYPES:
                if r.get(f'{action}_commitment', {}).get('answer') == 'YES' and r.get(f'{action}_commitment', {}).get('confidence', 0) >= OTHER_THRESHOLD:
                    variant_data[variant_name][f'{action}_commitment'] += 1
                if r.get(f'{action}_followthrough', {}).get('answer') == 'YES' and r.get(f'{action}_followthrough', {}).get('confidence', 0) >= OTHER_THRESHOLD:
                    variant_data[variant_name][f'{action}_followthrough'] += 1

    def _fmt(count, denom):
        return f"{(count / denom * 100):.1f}% ({count})" if denom > 0 else "0.0% (0)"

    def _rate(count, denom):
        return (count / denom * 100) if denom > 0 else 0

    rows = []
    for variant_name, d in sorted(variant_data.items()):
        recipients = d['recipients']
        active = d['reply']  # responders who have not opted out
        active_counts = d['active_reply_counts']
        median_replies = float(np.median(active_counts)) if active_counts else 0.0
        row = {
            'Variant': variant_name,
            'Total Recipients': recipients,
            'Total Responders': d['responders'],
            'Reply Rate': _fmt(d['responders'], recipients),
            'Opt-out Rate': _fmt(d['optout'], recipients),
            'Active Responders': _fmt(active, recipients),
            'Median Active Responses': median_replies,
            '_reply_rate': _rate(d['responders'], recipients),
            '_optout_rate': _rate(d['optout'], recipients),
            '_active_responders_rate': _rate(active, recipients),
        }
        for action in ACTION_TYPES:
            label = action.title()
            commit_count = d[f'{action}_commitment']
            row[f'{label} Commit'] = _fmt(commit_count, active)
            row[f'{label} Follow-through'] = _fmt(d[f'{action}_followthrough'], commit_count)
            row[f'_{action}_commitment_rate'] = _rate(commit_count, active)
            row[f'_{action}_followthrough_rate'] = _rate(d[f'{action}_followthrough'], commit_count)
        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()

# Sidebar: Date Filter (available before data upload)
st.sidebar.markdown("---")
st.sidebar.header("📅 Date Filter")

date_filter_type = st.sidebar.radio(
    "Filter conversations by date",
    ["All dates", "Before date", "After date", "Date range"],
    horizontal=False,
    key="date_filter_type"
)

# Store date filter values in session state (will be applied after data loads)
if date_filter_type == "Before date":
    filter_before_date = st.sidebar.date_input("Show conversations before", value=None, key="filter_before_date")
elif date_filter_type == "After date":
    filter_after_date = st.sidebar.date_input("Show conversations after", value=None, key="filter_after_date")
elif date_filter_type == "Date range":
    range_col1, range_col2 = st.sidebar.columns(2)
    with range_col1:
        filter_start_date = st.date_input("From", value=None, key="filter_start_date")
    with range_col2:
        filter_end_date = st.date_input("To", value=None, key="filter_end_date")

# Sidebar: LLM Configuration
st.sidebar.markdown("---")
st.sidebar.header("🤖 AI Verification")
use_llm_commitment = False
use_llm_variant_eval = False
llm_provider = None
llm_api_key = None

if ANTHROPIC_AVAILABLE or OPENAI_AVAILABLE:
    st.sidebar.caption("Select which features to enable:")
    use_llm_commitment = st.sidebar.checkbox(
        "AI commitment verification",
        value=False,
        help="Uses AI to detect commitments in all conversations"
    )
    use_llm_variant_eval = st.sidebar.checkbox(
        "AI variant evaluation",
        value=False,
        help="Uses AI to evaluate reply, opt-out, commitment, and follow-through per conversation grouped by message variant"
    )

    if use_llm_commitment or use_llm_variant_eval:
        # Check for centralized API keys
        centralized_anthropic = get_centralized_api_key("anthropic") if ANTHROPIC_AVAILABLE else None
        centralized_openai = get_centralized_api_key("openai") if OPENAI_AVAILABLE else None

        # Determine if we have a centralized key available
        has_centralized_key = bool(centralized_anthropic or centralized_openai)

        # Build list of available providers from centralized keys
        available_providers = []
        if centralized_anthropic:
            available_providers.append("Anthropic (Claude)")
        if centralized_openai:
            available_providers.append("OpenAI (GPT-4o-mini)")

        if available_providers:
            st.sidebar.success("AI features enabled")
            if len(available_providers) > 1:
                provider_choice = st.sidebar.radio(
                    "AI Provider",
                    available_providers,
                    key="llm_provider_choice"
                )
            else:
                provider_choice = available_providers[0]
        else:
            st.sidebar.caption("Enter an API key to enable AI:")
            provider_choice = None

        # Optional override section
        with st.sidebar.expander("Use your own API key (optional)"):
            st.caption("Override the default with your own key")
            anthropic_key_override = None
            openai_key_override = None

            if ANTHROPIC_AVAILABLE:
                anthropic_key_override = st.text_input(
                    "Anthropic API Key",
                    type="password",
                    help="From console.anthropic.com (uses Claude Haiku)"
                )

            if OPENAI_AVAILABLE:
                openai_key_override = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    help="From platform.openai.com (uses GPT-4o-mini)"
                )

        # Resolve final key: user override > centralized provider choice
        if anthropic_key_override:
            llm_provider = "anthropic"
            llm_api_key = anthropic_key_override
        elif openai_key_override:
            llm_provider = "openai"
            llm_api_key = openai_key_override
        elif provider_choice and "Anthropic" in provider_choice:
            llm_provider = "anthropic"
            llm_api_key = centralized_anthropic
        elif provider_choice and "OpenAI" in provider_choice:
            llm_provider = "openai"
            llm_api_key = centralized_openai
        else:
            st.sidebar.warning("Enter an API key to enable AI verification")
            use_llm_commitment = False
            use_llm_variant_eval = False

        # Auto-reset prompts when PROMPT_VERSION changes
        if st.session_state.get('prompt_version') != PROMPT_VERSION:
            st.session_state['custom_commitment_prompt'] = DEFAULT_COMMITMENT_PROMPT
            st.session_state['custom_variant_eval_prompt'] = DEFAULT_VARIANT_EVAL_PROMPT
            st.session_state['prompt_version'] = PROMPT_VERSION

        # Editable prompts section
        with st.sidebar.expander("✏️ Customize AI Prompts"):
            st.caption("Edit the prompts used for AI verification. Use {conversation} as placeholder for the conversation text.")

            custom_commitment_prompt = st.text_area(
                "Commitment Verification Prompt",
                value=st.session_state.get('custom_commitment_prompt', DEFAULT_COMMITMENT_PROMPT),
                height=200,
                key="commitment_prompt_input",
                help="Prompt used to verify commitment messages"
            )
            st.session_state['custom_commitment_prompt'] = custom_commitment_prompt

            custom_variant_eval_prompt = st.text_area(
                "Variant Evaluation Prompt",
                value=st.session_state.get('custom_variant_eval_prompt', DEFAULT_VARIANT_EVAL_PROMPT),
                height=300,
                key="variant_eval_prompt_input",
                help="Prompt used for multi-dimensional variant evaluation"
            )
            st.session_state['custom_variant_eval_prompt'] = custom_variant_eval_prompt

            if st.button("Reset to Defaults", key="reset_prompts"):
                st.session_state['custom_commitment_prompt'] = DEFAULT_COMMITMENT_PROMPT
                st.session_state['custom_variant_eval_prompt'] = DEFAULT_VARIANT_EVAL_PROMPT
                st.rerun()
else:
    st.sidebar.info("Install `anthropic` or `openai` package to enable AI verification")

# Get prompts (use defaults if not customized)
commitment_prompt_template = st.session_state.get('custom_commitment_prompt', DEFAULT_COMMITMENT_PROMPT)
variant_eval_prompt_template = st.session_state.get('custom_variant_eval_prompt', DEFAULT_VARIANT_EVAL_PROMPT)

if messages_files and people_file:
    # Load data (messages first - higher priority)
    try:
        messages_df, file_info = reconcile_message_files(messages_files)

        people_chunks = pd.read_csv(people_file, chunksize=50000, low_memory=False)
        people_df = pd.concat(people_chunks, ignore_index=True)

        # Drop unused columns to save memory (~80-90MB on large people files)
        _keep_cols = {'id', 'first_name', 'last_name', 'name', 'tags',
                      'sms_subscribed', 'sms_opt_out_at', 'sms_opt_out_reason',
                      'phone', 'phone_number', 'mobile', 'cell'}
        people_df = people_df[[c for c in people_df.columns if c in _keep_cols]]

        # Parse tags early for exclusion filter
        excluded_tags = []
        if 'tags' in people_df.columns:
            # Parse tags (assuming comma-separated)
            people_df['tag_list'] = people_df['tags'].fillna('').astype(str).apply(
                lambda x: [tag.strip() for tag in x.split(',') if tag.strip()]
            )

            # Get all unique tags
            all_tags_set = set()
            for tag_list in people_df['tag_list']:
                all_tags_set.update(tag_list)
            all_tags_list = sorted([tag for tag in all_tags_set if tag])

            # Sidebar: Exclude Test Data section
            if all_tags_list:
                st.sidebar.markdown("---")
                st.sidebar.header("🧪 Exclude Test Data")
                excluded_tags = st.sidebar.multiselect(
                    "Exclude people with these tags",
                    all_tags_list,
                    help="People with any of these tags will be excluded from all analytics"
                )

                if excluded_tags:
                    # Filter out people with excluded tags
                    def has_excluded_tag(tag_list):
                        return any(tag in excluded_tags for tag in tag_list)

                    excluded_people_mask = people_df['tag_list'].apply(has_excluded_tag)
                    excluded_count = excluded_people_mask.sum()

                    # Get IDs of excluded people before filtering
                    excluded_people_ids = set(people_df[excluded_people_mask]['id'].tolist())

                    # Filter people
                    people_df = people_df[~excluded_people_mask]

                    # Filter messages if person_id column exists
                    # (alternative format will have person_id added later, filter will be applied then)
                    if 'person_id' in messages_df.columns:
                        messages_df = messages_df[~messages_df['person_id'].isin(excluded_people_ids)]

                    # Store excluded IDs for later filtering (after format conversion)
                    st.session_state['excluded_people_ids'] = excluded_people_ids

                    st.sidebar.caption(f"Excluding {excluded_count:,} people")

        # Apply date filter from sidebar settings
        date_col = 'created_at' if 'created_at' in messages_df.columns else 'sent_at' if 'sent_at' in messages_df.columns else None

        if date_col and date_filter_type != "All dates":
            # Ensure date column is datetime
            messages_df[date_col] = pd.to_datetime(messages_df[date_col], errors='coerce')
            original_count = len(messages_df)

            if date_filter_type == "Before date" and st.session_state.get('filter_before_date'):
                filter_date = st.session_state['filter_before_date']
                messages_df = messages_df[messages_df[date_col].dt.date < filter_date]

            elif date_filter_type == "After date" and st.session_state.get('filter_after_date'):
                filter_date = st.session_state['filter_after_date']
                messages_df = messages_df[messages_df[date_col].dt.date > filter_date]

            elif date_filter_type == "Date range":
                start_date = st.session_state.get('filter_start_date')
                end_date = st.session_state.get('filter_end_date')
                if start_date and end_date:
                    messages_df = messages_df[(messages_df[date_col].dt.date >= start_date) & (messages_df[date_col].dt.date <= end_date)]

            if len(messages_df) != original_count:
                st.info(f"📅 Date filter applied: showing {len(messages_df):,} of {original_count:,} messages")

        # Collect loading messages for collapsed display
        loading_messages = []

        # Report file reconciliation details
        if len(file_info) > 1:
            for fi in file_info:
                loading_messages.append(("info", f"📄 {fi['filename']}: {fi['rows']:,} rows ({fi['export_type']} export)"))
            loading_messages.append(("success", f"🔗 Reconciled {len(file_info)} files into {len(messages_df):,} unique messages"))
        elif len(file_info) == 1:
            loading_messages.append(("info", f"📄 {file_info[0]['filename']} ({file_info[0]['export_type']} export)"))

        # Handle different message CSV formats
        if 'conversation_id' in messages_df.columns and 'from' in messages_df.columns:
            # Alternative format: conversation_id with from/to columns
            loading_messages.append(("info", "🔄 Detected alternative message format - converting to standard format..."))

            # Extract recipient phone numbers and create phone-to-person mapping
            recipient_phones = set()
            phone_to_conversation = {}
            conversation_ids = set()

            for _, row in messages_df.iterrows():
                if row['direction'] == 'outgoing':
                    # For outgoing messages, recipient is in 'to' field
                    recipient_phone = row['to']
                else:
                    # For incoming messages, sender is in 'from' field (but they're the recipient of the campaign)
                    recipient_phone = row['from']

                recipient_phones.add(recipient_phone)
                phone_to_conversation[recipient_phone] = row['conversation_id']
                conversation_ids.add(row['conversation_id'])

            # Try to reconcile with existing people data
            conversation_to_person_id = {}
            phone_to_person_id = {}
            matched_by_id = 0
            matched_by_phone = 0

            if not people_df.empty and 'id' in people_df.columns:
                # First, try direct ID matching: conversation_id -> people.id
                people_ids = set(people_df['id'].astype(str).tolist())
                for conv_id in conversation_ids:
                    conv_id_str = str(conv_id)
                    if conv_id_str in people_ids:
                        conversation_to_person_id[conv_id] = conv_id
                        matched_by_id += 1

                if matched_by_id > 0:
                    loading_messages.append(("success", f"🔗 Matched {matched_by_id} conversation IDs directly to people records"))

                # For unmatched conversations, try phone number matching
                unmatched_conversations = conversation_ids - set(conversation_to_person_id.keys())

                if unmatched_conversations:
                    # Find phone column in people data
                    phone_column = None
                    for col in ['phone', 'phone_number', 'mobile', 'cell']:
                        if col in people_df.columns:
                            phone_column = col
                            break

                    if phone_column:
                        # Build a lookup from normalized phone -> person ID
                        people_phone_lookup = {}
                        for _, person in people_df.iterrows():
                            person_phone = person[phone_column]
                            normalized = normalize_phone(person_phone)
                            if normalized:
                                people_phone_lookup[normalized] = person['id']

                        # Match recipient phones to people
                        for recipient_phone in recipient_phones:
                            normalized_recipient = normalize_phone(recipient_phone)
                            if normalized_recipient and normalized_recipient in people_phone_lookup:
                                phone_to_person_id[recipient_phone] = people_phone_lookup[normalized_recipient]
                                matched_by_phone += 1

                        if matched_by_phone > 0:
                            loading_messages.append(("success", f"🔗 Matched {matched_by_phone} additional records by phone number"))

            # Create person_id mapping for messages
            def get_person_id_for_row(row):
                # Preserve existing person_id from merged All Messages export
                if 'person_id' in row.index and pd.notna(row.get('person_id')):
                    return row['person_id']
                conv_id = row['conversation_id']
                phone = row['to'] if row['direction'] == 'outgoing' else row['from']

                # First try conversation_id direct match
                if conv_id in conversation_to_person_id:
                    return conversation_to_person_id[conv_id]
                # Then try phone match
                if phone in phone_to_person_id:
                    return phone_to_person_id[phone]
                # Fall back to conversation_id as person_id
                return conv_id

            # Add person_id to messages based on reconciliation
            messages_df['person_id'] = messages_df.apply(get_person_id_for_row, axis=1)

            # Determine which conversation_ids need synthetic people records
            all_matched_conv_ids = set(conversation_to_person_id.keys())
            # Also add conversation IDs that were matched via phone
            for phone, person_id in phone_to_person_id.items():
                conv_id = phone_to_conversation.get(phone)
                if conv_id:
                    all_matched_conv_ids.add(conv_id)

            unmatched_conv_ids = conversation_ids - all_matched_conv_ids

            if unmatched_conv_ids:
                if people_df.empty:
                    loading_messages.append(("warning", "📱 Creating synthetic people records from phone numbers in messages..."))
                    synthetic_people = []
                    for conv_id in sorted(unmatched_conv_ids, key=str):
                        # Find a phone for this conversation
                        phone = next((p for p, c in phone_to_conversation.items() if c == conv_id), None)
                        synthetic_people.append({
                            'id': conv_id,
                            'phone': phone,
                            'name': f"Contact {str(phone)[-4:]}" if phone else f"Contact {str(conv_id)[-4:]}",
                            'phone_number_type': messages_df[
                                messages_df['conversation_id'] == conv_id
                            ]['phone_number_type'].iloc[0] if 'phone_number_type' in messages_df.columns else 'mobile'
                        })
                    people_df = pd.DataFrame(synthetic_people)
                else:
                    loading_messages.append(("info", f"📱 Adding {len(unmatched_conv_ids)} missing people records from messages..."))
                    # Add missing people to existing dataframe
                    new_people = []
                    for conv_id in unmatched_conv_ids:
                        # Find a phone for this conversation
                        phone = next((p for p, c in phone_to_conversation.items() if c == conv_id), None)
                        new_people.append({
                            'id': conv_id,
                            'phone': phone,
                            'name': f"Contact {str(phone)[-4:]}" if phone else f"Contact {str(conv_id)[-4:]}",
                            'phone_number_type': messages_df[
                                messages_df['conversation_id'] == conv_id
                            ]['phone_number_type'].iloc[0] if 'phone_number_type' in messages_df.columns else 'mobile'
                        })

                    # Add missing people to existing people_df
                    if new_people:
                        new_people_df = pd.DataFrame(new_people)
                        people_df = pd.concat([people_df, new_people_df], ignore_index=True)

        # Convert datetime columns
        if 'created_at' in messages_df.columns:
            messages_df['created_at'] = pd.to_datetime(messages_df['created_at'])
        if 'sent_at' in messages_df.columns:
            messages_df['sent_at'] = pd.to_datetime(messages_df['sent_at'])

        # Apply tag-based exclusion filter for messages (for alternative format that now has person_id)
        if 'excluded_people_ids' in st.session_state and st.session_state['excluded_people_ids']:
            excluded_ids = st.session_state['excluded_people_ids']
            # Filter by both direct ID and string comparison for type safety
            messages_df = messages_df[
                ~messages_df['person_id'].isin(excluded_ids) &
                ~messages_df['person_id'].astype(str).isin([str(x) for x in excluded_ids])
            ]

        loading_messages.append(("success", f"✅ Loaded {len(messages_df)} messages and {len(people_df)} people records"))

        # Display loading messages in collapsed expander
        with st.expander("📋 Data Loading Details", expanded=False):
            for msg_type, msg_text in loading_messages:
                if msg_type == "success":
                    st.success(msg_text)
                elif msg_type == "info":
                    st.info(msg_text)
                elif msg_type == "warning":
                    st.warning(msg_text)

        # Create top-level tabs
        tab_campaign, tab_variant = st.tabs(["📈 Campaign Analytics", "📊 Variant Evaluation"])

        with tab_campaign:

            # Data overview
            st.header("📊 Campaign Summary")

            col1, col2, col3, col4 = st.columns(4)

            # Total recipients = people who received an outgoing message in this campaign
            # Exclude undelivered/failed messages — these were never received
            outgoing_mask = messages_df['direction'] == 'outgoing'
            if 'status' in messages_df.columns:
                outgoing_mask = outgoing_mask & (~messages_df['status'].fillna('').str.lower().isin(['undelivered', 'failed']))
            outgoing_person_ids = set(messages_df[outgoing_mask]['person_id'].unique())
            total_recipients = len(outgoing_person_ids)

            # Determine broadcast time (first outgoing message) for opt-out scoping
            outgoing_msgs = messages_df[outgoing_mask]
            broadcast_time = pd.to_datetime(outgoing_msgs['created_at'].min()) if not outgoing_msgs.empty else None

            # Calculate opt-outs AFTER broadcast from people file, or keyword fallback
            opted_out_people_set = set()
            optout_source = "keyword"
            if not people_df.empty and 'sms_subscribed' in people_df.columns and 'sms_opt_out_at' in people_df.columns and broadcast_time is not None:
                # Only count people who opted out AFTER the broadcast was sent
                people_with_optout = people_df[
                    (people_df['sms_subscribed'] == False) &
                    (people_df['sms_opt_out_at'].notna())
                ].copy()
                people_with_optout['_opt_out_at'] = pd.to_datetime(people_with_optout['sms_opt_out_at'], errors='coerce')
                opted_out_after = set(people_with_optout[people_with_optout['_opt_out_at'] > broadcast_time]['id'])
                opted_out_people_set = opted_out_after & outgoing_person_ids
                optout_source = "people_file"
            elif not people_df.empty and 'sms_subscribed' in people_df.columns:
                # No opt_out_at column — fall back to all unsubscribed recipients
                opted_out_people_set = set(people_df[people_df['sms_subscribed'] == False]['id']) & outgoing_person_ids
                optout_source = "people_file"
            elif 'body' in messages_df.columns:
                stop_keywords = ['stop', 'unsubscribe', 'opt out', 'opt-out', 'remove', 'quit', 'cancel',
                                 'stopall', 'end', 'revoke', 'optout']
                incoming_msgs = messages_df[messages_df['direction'] == 'incoming'].copy()
                body_stripped = incoming_msgs['body'].fillna('').str.lower().str.strip().str.replace(r'[^\w\s-]', '', regex=True).str.strip()
                opted_out_people_set = set(incoming_msgs[body_stripped.isin(stop_keywords)]['person_id'].unique())
            opt_outs = len(opted_out_people_set)

            # Calculate responders from incoming messages
            all_responders = set(messages_df[messages_df['direction'] == 'incoming']['person_id'].unique())
            total_responders = len(all_responders)

            # Total Response Rate: people who replied 1+ times / total recipients
            total_response_rate = (total_responders / total_recipients * 100) if total_recipients > 0 else 0

            # Active responders: replied AND not opted out after broadcast
            active_responders_set = all_responders - opted_out_people_set
            active_responders = len(active_responders_set)
            active_responder_pct = (active_responders / total_recipients * 100) if total_recipients > 0 else 0

            with col1:
                st.metric("Total Recipients", f"{total_recipients:,}")
            with col2:
                st.metric("Total Response Rate", f"{total_response_rate:.1f}%")
                st.caption(f"{total_responders:,} replied / {total_recipients:,} recipients")
            with col3:
                st.metric("Active Responders", f"{active_responders:,} ({active_responder_pct:.1f}%)")
                st.caption("Responders who have not opted out")
            with col4:
                optout_pct = (opt_outs / total_recipients * 100) if total_recipients > 0 else 0
                st.metric("Opt-outs", f"{opt_outs:,} ({optout_pct:.1f}%)")
                if optout_source == "people_file":
                    st.caption("Opted out after broadcast")
                else:
                    st.caption("STOP keyword matches")

            # Engagement depth analysis
            st.header("📈 Engagement Depth Analysis")
            st.caption("Analyzes message counts from active responders (excludes opted-out users)")

            # Calculate response counts per person (exclude opt-outs from engagement analysis)
            engaged_messages = messages_df[
                (messages_df['direction'] == 'incoming') &
                (~messages_df['person_id'].isin(opted_out_people_set))
            ]
            person_response_counts = engaged_messages.groupby('person_id').size().reset_index(name='response_count')

            # Create engagement buckets
            engagement_buckets = {
                '1 Response': len(person_response_counts[person_response_counts['response_count'] == 1]),
                '2+ Responses': len(person_response_counts[person_response_counts['response_count'] >= 2]),
                '3+ Responses': len(person_response_counts[person_response_counts['response_count'] >= 3])
            }

            col1, col2 = st.columns(2)

            with col1:
                # Engagement depth chart
                st.subheader("Response Engagement Levels")
                chart_df = pd.DataFrame({
                    'Engagement Level': list(engagement_buckets.keys()),
                    'Number of People': list(engagement_buckets.values())
                }).set_index('Engagement Level')
                st.bar_chart(chart_df)

            with col2:
                # Response distribution
                if not person_response_counts.empty:
                    st.subheader("Distribution of Response Counts")
                    hist_data = person_response_counts['response_count'].value_counts().sort_index()
                    hist_df = pd.DataFrame({
                        'Number of People': hist_data.values
                    }, index=hist_data.index)
                    hist_df.index.name = 'Number of Responses'
                    st.bar_chart(hist_df)

            # Opt-out timing analysis
            st.header("⏰ Opt-out Timing Analysis")

            if opted_out_people_set:
                # Stop keywords to identify opt-out-only replies vs substantive conversation
                _stop_keywords = {'stop', 'unsubscribe', 'opt out', 'opt-out', 'remove', 'quit', 'cancel',
                                  'stopall', 'end', 'revoke', 'optout'}

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Opt-outs", f"{opt_outs:,}")
                    st.caption(f"Of {total_recipients:,} total recipients")
                with col2:
                    optout_rate = (opt_outs / total_recipients * 100) if total_recipients > 0 else 0
                    st.metric("Opt-out Rate", f"{optout_rate:.1f}%")
                    st.caption("% of recipients who opted out after broadcast")

                # Classify ALL opted-out people by engagement
                opt_out_categories = []
                opt_out_details = []

                # Build people lookup for opt-out reason/timestamp
                _people_optout_lookup = {}
                if not people_df.empty and 'sms_opt_out_reason' in people_df.columns:
                    for _, row in people_df[people_df['id'].isin(opted_out_people_set)].iterrows():
                        _people_optout_lookup[row['id']] = {
                            'reason': row.get('sms_opt_out_reason', ''),
                            'opt_out_at': pd.to_datetime(row.get('sms_opt_out_at'), errors='coerce')
                        }

                for person_id in opted_out_people_set:
                    person_messages = messages_df[messages_df['person_id'] == person_id].sort_values('created_at')
                    person_incoming = person_messages[person_messages['direction'] == 'incoming']

                    optout_info = _people_optout_lookup.get(person_id, {})
                    opt_out_time = optout_info.get('opt_out_at')
                    opt_out_reason = optout_info.get('reason', '')
                    if pd.isna(opt_out_reason):
                        opt_out_reason = ''

                    num_replies = len(person_incoming)
                    bot_messages = len(person_messages[person_messages['direction'] == 'outgoing'])

                    # Classify: check if any incoming messages are substantive (not just stop keywords)
                    has_substantive_reply = False
                    if num_replies > 0:
                        for _, msg in person_incoming.iterrows():
                            body = str(msg.get('body', '') or '').strip()
                            body_normalized = body.lower().strip()
                            # Remove punctuation for keyword comparison
                            body_clean = pd.Series([body_normalized]).str.replace(r'[^\w\s-]', '', regex=True).str.strip().iloc[0]
                            if body_clean and body_clean not in _stop_keywords:
                                has_substantive_reply = True
                                break

                    engagement_category = "Replied before opting out" if has_substantive_reply else "Opt-out only"

                    # Calculate hours from first outgoing to opt-out
                    first_bot_msg = person_messages[person_messages['direction'] == 'outgoing']
                    hours = None
                    if len(first_bot_msg) > 0 and pd.notna(opt_out_time):
                        first_bot_time = first_bot_msg.iloc[0]['created_at']
                        if pd.notna(first_bot_time):
                            hours = (opt_out_time - first_bot_time).total_seconds() / 3600

                    opt_out_categories.append(engagement_category)
                    opt_out_details.append({
                        'person_id': person_id,
                        'engagement_category': engagement_category,
                        'replies': num_replies,
                        'bot_messages': bot_messages,
                        'hours_elapsed': hours,
                        'opt_out_reason': opt_out_reason,
                    })

                with st.expander(f"View {opt_outs:,} Opt-out Details"):
                    if opt_out_categories:
                        opt_out_df = pd.DataFrame(opt_out_details)
                        category_counts = pd.Series(opt_out_categories).value_counts()

                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("Opt-out Breakdown by Engagement")
                            optout_chart_df = pd.DataFrame({
                                'Number of Opt-outs': category_counts.values
                            }, index=category_counts.index)
                            optout_chart_df.index.name = 'Engagement Type'
                            st.bar_chart(optout_chart_df)

                        with col2:
                            st.subheader("Key Insights")

                            optout_only_count = len(opt_out_df[opt_out_df['engagement_category'] == 'Opt-out only'])
                            replied_count = len(opt_out_df[opt_out_df['engagement_category'] == 'Replied before opting out'])

                            st.metric("Opt-out only", f"{optout_only_count:,}",
                                     help="Unsubscribed without any substantive reply — their only action was to opt out")
                            st.metric("Replied before opting out", f"{replied_count:,}",
                                     help="Had a real conversation before unsubscribing")

                            valid_hours = opt_out_df['hours_elapsed'].dropna()
                            if len(valid_hours) > 0:
                                avg_hours = valid_hours.mean()
                                st.metric("Average time to opt-out", f"{avg_hours:.1f} hours")

                            if optout_only_count > 0:
                                pct_immediate = (optout_only_count / len(opt_out_df)) * 100
                                if pct_immediate > 50:
                                    st.warning(f"⚠️ {pct_immediate:.0f}% opted out without any substantive reply")

                        # Opt-out reason breakdown
                        if opt_out_df['opt_out_reason'].fillna('').str.strip().replace('', pd.NA).dropna().any():
                            reason_counts = opt_out_df['opt_out_reason'].replace('', 'Unknown').value_counts()
                            st.subheader("Opt-out Reasons")
                            st.dataframe(reason_counts.reset_index().rename(columns={'index': 'Reason', 'opt_out_reason': 'Reason', 'count': 'Count'}), hide_index=True)

                        # Detailed table
                        st.subheader("Individual Opt-out Details")

                        def get_person_name_for_table(person_id):
                            name = get_person_name(person_id, people_df)
                            return name if name else f"Person {person_id}"

                        opt_out_df['person_name'] = opt_out_df['person_id'].apply(get_person_name_for_table)
                        opt_out_df['hours_rounded'] = opt_out_df['hours_elapsed'].round(1)

                        display_cols = ['person_name', 'engagement_category', 'replies', 'bot_messages', 'hours_rounded', 'opt_out_reason']
                        col_config = {
                            'person_name': 'Name',
                            'engagement_category': 'Engagement',
                            'replies': 'Replies',
                            'bot_messages': 'Bot Messages',
                            'hours_rounded': 'Hours to Opt-out',
                            'opt_out_reason': 'Opt-out Reason'
                        }
                        opt_out_df = opt_out_df.sort_values(['engagement_category', 'replies'], ascending=[True, False])
                        st.dataframe(opt_out_df[display_cols].rename(columns=col_config), hide_index=True)
            else:
                st.info("No opt-outs found in this campaign")

            # Committed responders detection
            st.header("✅ Committed Responders")

            commitment_keywords = ["yes", "i will", "i'll", "count me in", "sign me up", "i'm in", "absolutely", "definitely", "for sure", "commit", "committed", "attend", "participate", "join", "coming", "be there"]

            if 'body' in messages_df.columns:
                incoming_messages = messages_df[messages_df['direction'] == 'incoming'].copy()

                # Create cache key for commitment detection
                commitment_cache_key = f"commit_{len(messages_df)}_{use_llm_commitment}_{llm_provider}_{hash(commitment_prompt_template)}_{len(opted_out_people_set)}"

                # Check if we have cached results
                if 'commitment_cache_key' in st.session_state and st.session_state.get('commitment_cache_key') == commitment_cache_key:
                    committed_people = st.session_state['committed_people']
                    rejected_people = st.session_state['rejected_people']
                    llm_verified_count = st.session_state.get('llm_verified_count', 0)
                    llm_rejected_count = st.session_state.get('llm_rejected_count', 0)
                    llm_unknown_count = st.session_state.get('llm_unknown_count', 0)
                else:
                    llm_verified_count = 0
                    llm_rejected_count = 0
                    llm_unknown_count = 0
                    committed_people = []
                    rejected_people = []

                    if use_llm_commitment and llm_provider and llm_api_key:
                        # AI mode: Analyze ALL non-opted-out responders
                        # Get all unique responders who haven't opted out
                        all_responder_ids = set(incoming_messages['person_id'].unique())
                        non_opted_out_responders = all_responder_ids - opted_out_people_set

                        if non_opted_out_responders:
                            # Build conversations for all non-opted-out responders
                            commitment_conversations = {}
                            responder_info = {}  # Store info for building results

                            for person_id in non_opted_out_responders:
                                person_msgs = messages_df[messages_df['person_id'] == person_id].sort_values('created_at')
                                commitment_conversations[person_id] = format_conversation_for_llm(person_msgs)

                                # Get first incoming message for date/message info
                                first_incoming = incoming_messages[incoming_messages['person_id'] == person_id].iloc[0]
                                person_name = get_person_name(person_id, people_df)
                                if not person_name:
                                    person_name = f"Person {person_id}"

                                responder_info[person_id] = {
                                    'person_name': person_name,
                                    'person_id': person_id,
                                    'commitment_date': first_incoming['created_at'],
                                    'raw_message': first_incoming['body']
                                }

                            provider_name = "Claude" if llm_provider == "anthropic" else "GPT-4o-mini"
                            n_to_analyze = len(commitment_conversations)
                            progress_bar = st.progress(0, text=f"Analyzing 0/{n_to_analyze} conversations with {provider_name} (0%)...")
                            def _update_commitment_progress(completed, total):
                                pct = int(completed / total * 100)
                                progress_bar.progress(completed / total, text=f"Analyzing {completed}/{total} conversations with {provider_name} ({pct}%)...")
                            if llm_provider == "anthropic":
                                llm_results = analyze_commitments_with_anthropic(commitment_conversations, llm_api_key, commitment_prompt_template, progress_callback=_update_commitment_progress)
                            else:
                                llm_results = analyze_commitments_with_openai(commitment_conversations, llm_api_key, commitment_prompt_template, progress_callback=_update_commitment_progress)
                            progress_bar.empty()

                            # Process LLM results
                            for person_id, info in responder_info.items():
                                result = llm_results.get(person_id, {"is_committed": "unknown"})
                                status = result.get('is_committed')
                                confidence = result.get('confidence')
                                raw_answer = result.get('raw_answer', '')
                                summary = result.get('summary')

                                person_data = info.copy()
                                # Format confidence display
                                if confidence is not None:
                                    person_data['confidence'] = f"{raw_answer} {confidence}%"
                                else:
                                    person_data['confidence'] = f"{raw_answer} (no score)"

                                # Store AI summary (fallback to raw_message if no summary)
                                person_data['ai_summary'] = summary if summary else person_data['raw_message']

                                if status == "unknown":
                                    person_data['detection_method'] = 'AI Unknown'
                                    committed_people.append(person_data)
                                    llm_unknown_count += 1
                                elif status:
                                    person_data['detection_method'] = 'AI Verified'
                                    committed_people.append(person_data)
                                    llm_verified_count += 1
                                else:
                                    person_data['detection_method'] = 'AI Rejected'
                                    rejected_people.append(person_data)
                                    llm_rejected_count += 1

                    else:
                        # No LLM - use keyword matching
                        keyword_committed_people = []
                        seen_people = set()

                        for _, msg in incoming_messages.iterrows():
                            if any(keyword in str(msg['body']).lower() for keyword in commitment_keywords):
                                if msg['person_id'] not in seen_people:
                                    seen_people.add(msg['person_id'])

                                    person_name = get_person_name(msg['person_id'], people_df)
                                    if not person_name:
                                        person_name = f"Person {msg['person_id']}"

                                    keyword_committed_people.append({
                                        'person_name': person_name,
                                        'person_id': msg['person_id'],
                                        'commitment_date': msg['created_at'],
                                        'raw_message': msg['body'],
                                        'detection_method': 'Keyword',
                                        'confidence': 'N/A'
                                    })

                        committed_people = keyword_committed_people

                    # Cache the results
                    st.session_state['commitment_cache_key'] = commitment_cache_key
                    st.session_state['committed_people'] = committed_people
                    st.session_state['rejected_people'] = rejected_people
                    st.session_state['llm_verified_count'] = llm_verified_count
                    st.session_state['llm_rejected_count'] = llm_rejected_count
                    st.session_state['llm_unknown_count'] = llm_unknown_count

                # Show LLM results (whether cached or fresh)
                if use_llm_commitment and llm_provider and llm_api_key:
                    if llm_verified_count > 0:
                        st.success(f"🤖 AI detected {llm_verified_count} commitment(s)")
                    if llm_rejected_count > 0:
                        st.info(f"🤖 AI found {llm_rejected_count} responder(s) with unclear commitments")
                    if llm_unknown_count > 0:
                        st.warning(f"⚠️ {llm_unknown_count} conversation(s) could not be analyzed (API error)")

                if committed_people or rejected_people:
                    # Summary stats
                    if rejected_people:
                        col1, col2, col3 = st.columns(3)
                    else:
                        col1, col2 = st.columns(2)
                        col3 = None

                    with col1:
                        total_committed = len(committed_people)
                        st.metric("Committed Responders", f"{total_committed}")
                    with col2:
                        if total_committed > 0:
                            commitment_rate = (total_committed / active_responders * 100) if active_responders > 0 else 0
                            st.metric("Commitment Rate", f"{commitment_rate:.1f}%")
                            st.caption("% of active responders who committed to action")
                    if col3:
                        with col3:
                            st.metric("AI Rejected (False Positives)", f"{len(rejected_people)}")
                            st.caption("Keyword matches that AI determined weren't commitments")

                    # Collapsible table section - include both committed and rejected
                    all_people = committed_people + rejected_people
                    expander_label = f"View {len(committed_people)} Committed" + (f" + {len(rejected_people)} AI-Rejected" if rejected_people else "") + " Details"
                    with st.expander(expander_label):
                        all_df = pd.DataFrame(all_people)
                        all_df['commitment_date_formatted'] = all_df['commitment_date'].dt.strftime('%Y-%m-%d %H:%M')

                        # Use AI summary if available, otherwise raw message
                        if use_llm_commitment and llm_provider and llm_api_key and 'ai_summary' in all_df.columns:
                            display_cols = ['person_name', 'commitment_date_formatted', 'detection_method', 'confidence', 'ai_summary']
                            col_names = ['Person', 'Date', 'Status', 'Confidence', 'AI Summary']
                            column_config = {
                                'Status': st.column_config.TextColumn(
                                    help="AI Verified = YES with Confidence ≥ 85%"
                                ),
                                'AI Summary': st.column_config.TextColumn(width=800)
                            }
                        else:
                            display_cols = ['person_name', 'commitment_date_formatted', 'raw_message']
                            col_names = ['Person', 'Date', 'Message']
                            column_config = {
                                'Message': st.column_config.TextColumn(width=400)
                            }

                        display_df = all_df[display_cols].copy()
                        display_df.columns = col_names

                        st.dataframe(display_df, width='stretch', column_config=column_config)
                else:
                    st.info("No action commitments detected in the conversation data")

            # Network multiplier detection
            st.header("🌐 Network Multiplier Detection")

            multiplier_keywords = [
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

            if 'body' in messages_df.columns:
                incoming_messages = messages_df[messages_df['direction'] == 'incoming']

                # Find unique people who sent multiplier messages with their first occurrence
                multiplier_people = []
                seen_people = set()

                for _, msg in incoming_messages.iterrows():
                    if any(keyword in str(msg['body']).lower() for keyword in multiplier_keywords):
                        if msg['person_id'] not in seen_people:
                            seen_people.add(msg['person_id'])

                            # Get person name
                            person_name = get_person_name(msg['person_id'], people_df)
                            if not person_name:
                                person_name = f"Person {msg['person_id']}"

                            multiplier_people.append({
                                'person_name': person_name,
                                'person_id': msg['person_id'],
                                'commitment_date': msg['created_at'],
                                'raw_message': msg['body']
                            })

                if multiplier_people:
                    # Summary stats first
                    col1, col2 = st.columns(2)
                    with col1:
                        total_multipliers = len(multiplier_people)
                        st.metric("Network Multipliers", f"{total_multipliers}")
                    with col2:
                        if total_multipliers > 0:
                            multiplier_rate = (total_multipliers / active_responders * 100) if active_responders > 0 else 0
                            st.metric("Multiplier Rate", f"{multiplier_rate:.1f}%")
                            st.caption("% of active responders who committed to connecting others")

                    # Collapsible table section
                    with st.expander(f"View {len(multiplier_people)} Network Multiplier Details"):
                        # Display as a clean table
                        multiplier_df = pd.DataFrame(multiplier_people)

                        # Format the date for display
                        multiplier_df['commitment_date_formatted'] = multiplier_df['commitment_date'].dt.strftime('%Y-%m-%d %H:%M')

                        # Display table with raw message
                        display_df = multiplier_df[['person_name', 'commitment_date_formatted', 'raw_message']].copy()
                        display_df.columns = ['Person', 'Commitment Date', 'Message']

                        st.dataframe(display_df, width='stretch')
                else:
                    st.info("No network multiplier commitments detected in the conversation data")

            # Tag-based segment comparison
            st.header("🏷️ Tag-Based Segment Analysis")

            if 'tags' in people_df.columns:
                # Parse tags (assuming comma-separated)
                people_df['tag_list'] = people_df['tags'].fillna('').astype(str).apply(lambda x: [tag.strip() for tag in x.split(',') if tag.strip()])

                # Get all unique tags
                all_tags = set()
                for tag_list in people_df['tag_list']:
                    all_tags.update(tag_list)
                all_tags = [tag for tag in all_tags if tag]

                if all_tags:
                    selected_tags = st.multiselect("Select tags to compare", all_tags)

                    if selected_tags:
                        tag_comparison = {}

                        for tag in selected_tags:
                            tagged_people = people_df[people_df['tag_list'].apply(lambda x: tag in x)]['id'].tolist()
                            tagged_responses = messages_df[
                                (messages_df['person_id'].isin(tagged_people)) &
                                (messages_df['direction'] == 'incoming')
                            ]['person_id'].nunique()

                            response_rate = (tagged_responses / len(tagged_people) * 100) if len(tagged_people) > 0 else 0

                            tag_comparison[tag] = {
                                'recipients': len(tagged_people),
                                'responders': tagged_responses,
                                'response_rate': response_rate
                            }

                        # Display comparison
                        comparison_df = pd.DataFrame(tag_comparison).T
                        comparison_df = comparison_df.round(2)
                        st.dataframe(comparison_df)

            # CSV Export section
            st.header("📤 Export as CSV")

            # Calculate sets for filtering
            all_responder_ids_export = set(messages_df[messages_df['direction'] == 'incoming']['person_id'].unique())
            non_responder_ids_export = outgoing_person_ids - all_responder_ids_export
            committed_ids_export = set()
            if 'committed_people' in st.session_state and st.session_state['committed_people']:
                committed_ids_export = {p['person_id'] for p in st.session_state['committed_people']}

            # Filter and export type selection
            export_col1, export_col2 = st.columns(2)
            with export_col1:
                export_filter = st.selectbox(
                    "Filter by",
                    ["All Responders", "Opted Out", "Active (Not Opted Out)", "Committed", "Non-Responders"],
                    key="export_filter"
                )
            with export_col2:
                export_type = st.selectbox(
                    "Export type",
                    ["Responder Details", "Full Conversation"],
                    key="export_type"
                )

            # Apply filter
            if export_filter == "All Responders":
                export_ids = list(all_responder_ids_export)
                filter_label = "all_responders"
            elif export_filter == "Opted Out":
                export_ids = list(opted_out_people_set)
                filter_label = "opted_out"
            elif export_filter == "Active (Not Opted Out)":
                export_ids = list(all_responder_ids_export - opted_out_people_set)
                filter_label = "active"
            elif export_filter == "Committed":
                export_ids = list(committed_ids_export)
                filter_label = "committed"
            else:  # Non-Responders
                export_ids = list(non_responder_ids_export)
                filter_label = "non_responders"

            st.info(f"Found {len(export_ids)} people matching filter: {export_filter}")

            if len(export_ids) > 0:
                # Tag input
                export_tag = st.text_input("Tag for export:", value=f"{filter_label}_v1", key="export_tag")

                if st.button("📥 Generate CSV Export"):
                    export_data = []

                    for person_id in export_ids:
                        # Handle type mismatches with string comparison fallback
                        person_row = people_df[people_df['id'] == person_id]
                        if person_row.empty:
                            person_row = people_df[people_df['id'].astype(str) == str(person_id)]

                        # Extract person info
                        first_name = ""
                        last_name = ""
                        phone = ""

                        if not person_row.empty:
                            person = person_row.iloc[0]

                            # Handle different name column formats
                            if 'first_name' in people_df.columns and pd.notna(person['first_name']):
                                first_name = str(person['first_name']).strip()
                            if 'last_name' in people_df.columns and pd.notna(person['last_name']):
                                last_name = str(person['last_name']).strip()
                            if 'name' in people_df.columns and not first_name and not last_name:
                                name_val = person['name']
                                if pd.notna(name_val):
                                    full_name = str(name_val).strip()
                                    name_parts = full_name.split(' ', 1)
                                    first_name = name_parts[0] if len(name_parts) > 0 else ''
                                    last_name = name_parts[1] if len(name_parts) > 1 else ''

                            # Handle different phone column formats
                            for phone_col in ['phone', 'phone_number', 'mobile', 'cell']:
                                if phone_col in people_df.columns:
                                    phone_val = person[phone_col]
                                    if pd.notna(phone_val) and str(phone_val).strip():
                                        phone = str(phone_val).strip()
                                        # Remove .0 suffix from float conversion
                                        if phone.endswith('.0'):
                                            phone = phone[:-2]
                                        break

                        # Build export row
                        row_data = {
                            'first_name': first_name,
                            'last_name': last_name,
                            'phone_number': phone,
                            'tag': export_tag
                        }

                        # Add conversation if full export type selected
                        if export_type == "Full Conversation":
                            person_messages = messages_df[
                                messages_df['person_id'] == person_id
                            ].sort_values('created_at')

                            if len(person_messages) > 0:
                                conversation_array = []
                                for _, msg in person_messages.iterrows():
                                    direction = "YOU" if msg['direction'] == 'outgoing' else "THEM"
                                    timestamp = msg['created_at'].strftime("%Y-%m-%d %H:%M") if pd.notna(msg['created_at']) else ""
                                    conversation_array.append({
                                        'direction': direction,
                                        'timestamp': timestamp,
                                        'message': msg['body']
                                    })
                                row_data['conversation'] = json.dumps(conversation_array)
                            else:
                                row_data['conversation'] = "[]"

                        export_data.append(row_data)

                    if export_data:
                        export_df = pd.DataFrame(export_data)

                        # Convert to CSV
                        csv_data = export_df.to_csv(index=False)

                        st.download_button(
                            label="⬇️ Download CSV",
                            data=csv_data,
                            file_name=f"{filter_label}_{export_tag}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )

                        # Preview the data
                        st.subheader("Preview of Export Data:")
                        st.dataframe(export_df.head(10))
                        st.caption(f"Showing first 10 of {len(export_df)} records")
                    else:
                        st.warning("No data available for export")

            # Responder details with conversation viewer
            st.header("💬 Responder Details & Conversations")

            # Get all responder IDs
            all_responder_ids = set(messages_df[messages_df['direction'] == 'incoming']['person_id'].unique())

            # Get committed person IDs (from the committed_people list if it exists)
            committed_ids = set()
            if 'committed_people' in st.session_state and st.session_state['committed_people']:
                committed_ids = {p['person_id'] for p in st.session_state['committed_people']}

            # Calculate non-responders
            non_responder_ids = outgoing_person_ids - all_responder_ids

            # Filter options
            filter_col1, filter_col2 = st.columns([2, 3])
            with filter_col1:
                conversation_filter = st.radio(
                    "Filter by",
                    ["All Responders", "Opted Out", "Active (Not Opted Out)", "Committed", "Non-Responders"],
                    horizontal=True
                )

            # Apply filter
            if conversation_filter == "All Responders":
                filtered_ids = list(all_responder_ids)
                filter_description = f"{len(filtered_ids)} responders"
            elif conversation_filter == "Opted Out":
                filtered_ids = list(opted_out_people_set)
                filter_description = f"{len(filtered_ids)} opted out"
            elif conversation_filter == "Active (Not Opted Out)":
                filtered_ids = list(all_responder_ids - opted_out_people_set)
                filter_description = f"{len(filtered_ids)} active responders"
            elif conversation_filter == "Committed":
                filtered_ids = list(committed_ids)
                filter_description = f"{len(filtered_ids)} committed"
            else:  # Non-Responders
                filtered_ids = list(non_responder_ids)
                filter_description = f"{len(filtered_ids)} non-responders"

            with filter_col2:
                st.caption(f"Showing: {filter_description}")

            if len(filtered_ids) > 0:
                # Create a mapping of person_id to name if available
                def get_display_name(person_id):
                    name = get_person_name(person_id, people_df)
                    if name:
                        return f"{name} (ID: {person_id})"
                    return f"Person {person_id}"

                selected_person = st.selectbox(
                    "Select a person to view conversation",
                    filtered_ids,
                    format_func=get_display_name
                )

                if selected_person:
                    # Get conversation for selected person
                    conversation = messages_df[
                        messages_df['person_id'] == selected_person
                    ].sort_values('created_at')

                    st.subheader(f"Conversation with {get_display_name(selected_person)}")

                    if len(conversation) > 0:
                        for _, msg in conversation.iterrows():
                            direction_icon = "👤" if msg['direction'] == 'incoming' else "🤖"
                            timestamp = msg['created_at'].strftime("%Y-%m-%d %H:%M") if pd.notna(msg['created_at']) else "Unknown time"

                            with st.chat_message("user" if msg['direction'] == 'incoming' else "assistant"):
                                st.write(f"**{direction_icon} {timestamp}**")
                                st.write(msg['body'])
                    else:
                        st.info("No messages found for this person (non-responder)")
            else:
                st.info(f"No people found matching filter: {conversation_filter}")

        with tab_variant:

            if not use_llm_variant_eval or not llm_api_key:
                st.info("Enable **AI variant evaluation** in the sidebar and configure an API key to use this feature.")
            else:
                # Derive person -> variant mapping from first outgoing message
                person_variant_map = get_person_variant(messages_df)

                if not person_variant_map:
                    st.warning("No outgoing messages found to evaluate.")
                else:
                    variant_names = sorted(set(person_variant_map.values()))
                    n_conversations = len(person_variant_map)
                    n_with_replies = len(set(messages_df[messages_df['direction'] == 'incoming']['person_id'].unique()) & set(person_variant_map.keys()))

                    # Caching
                    variant_eval_cache_key = (
                        f"variant_eval_{len(messages_df)}_{len(people_df)}"
                        f"_{llm_provider}_{hash(variant_eval_prompt_template)}"
                    )

                    if ('variant_eval_cache_key' in st.session_state
                            and st.session_state.get('variant_eval_cache_key') == variant_eval_cache_key
                            and 'variant_eval_results' in st.session_state):
                        eval_results = st.session_state['variant_eval_results']
                        variant_summary_df = st.session_state['variant_eval_aggregated']
                        cached_responder_ids = st.session_state.get('variant_eval_responder_ids', set())
                        cached_optout_ids = st.session_state.get('variant_eval_optout_ids', set())
                    else:
                        eval_results = None
                        variant_summary_df = None
                        cached_responder_ids = set()
                        cached_optout_ids = set()

                    # Show run button if no cached results
                    if eval_results is None:
                        n_no_replies = n_conversations - n_with_replies
                        variant_label = f"across {len(variant_names)} variants" if len(variant_names) > 1 else f"({variant_names[0]})"
                        st.caption(f"{n_conversations} recipients {variant_label} — {n_with_replies} replied ({n_no_replies} didn't reply, still counted toward totals). AI evaluation will run on {n_with_replies} conversations via {llm_provider.title()}.")
                        run_variant_eval = st.button(f"Run Variant Evaluation ({n_with_replies} AI calls)", key="run_variant_eval")

                        if run_variant_eval:
                            # Split into conversations with replies (need LLM) vs no replies (skip LLM)
                            responder_pids = set(messages_df[messages_df['direction'] == 'incoming']['person_id'].unique())
                            conversations_with_replies = {}
                            eval_results = {}
                            for person_id in person_variant_map:
                                if person_id not in responder_pids:
                                    # No incoming messages — all dimensions are NO, skip LLM
                                    eval_results[person_id] = _variant_eval_fallback()
                                    continue
                                person_msgs = messages_df[messages_df['person_id'] == person_id].sort_values('created_at')
                                if not person_msgs.empty:
                                    conversations_with_replies[person_id] = format_conversation_for_llm(person_msgs)

                            if conversations_with_replies:
                                n_to_eval = len(conversations_with_replies)
                                progress_bar = st.progress(0, text=f"Evaluating 0/{n_to_eval} conversations (0%)...")
                                def _update_progress(completed, total):
                                    pct = int(completed / total * 100)
                                    progress_bar.progress(completed / total, text=f"Evaluating {completed}/{total} conversations ({pct}%)...")
                                if llm_provider == "anthropic":
                                    llm_results = analyze_variant_eval_with_anthropic(conversations_with_replies, llm_api_key, variant_eval_prompt_template, progress_callback=_update_progress)
                                else:
                                    llm_results = analyze_variant_eval_with_openai(conversations_with_replies, llm_api_key, variant_eval_prompt_template, progress_callback=_update_progress)
                                progress_bar.empty()
                                eval_results.update(llm_results)

                            # Reuse opt-out set from Campaign Analytics (broadcast-time-scoped)
                            variant_optout_ids = opted_out_people_set & set(person_variant_map.keys())

                            # Count incoming messages per person for median calculation
                            _reply_counts = messages_df[messages_df['direction'] == 'incoming'].groupby('person_id').size().to_dict()

                            variant_summary_df = aggregate_variant_metrics(eval_results, person_variant_map, responder_pids, variant_optout_ids, _reply_counts)

                            # Cache results
                            st.session_state['variant_eval_cache_key'] = variant_eval_cache_key
                            st.session_state['variant_eval_results'] = eval_results
                            st.session_state['variant_eval_aggregated'] = variant_summary_df
                            st.session_state['variant_eval_responder_ids'] = responder_pids
                            st.session_state['variant_eval_optout_ids'] = variant_optout_ids

                    # Display results if available
                    if eval_results is not None and variant_summary_df is not None and not variant_summary_df.empty:
                        # Summary metrics
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.metric("Variants", len(variant_names))
                        with m2:
                            st.metric("Conversations Evaluated", len(eval_results))
                        with m3:
                            avg_reply = variant_summary_df['_reply_rate'].mean()
                            st.metric("Avg Reply Rate", f"{avg_reply:.1f}%")

                        # Main comparison table
                        display_cols = ['Variant', 'Total Recipients', 'Total Responders', 'Reply Rate', 'Opt-out Rate',
                                        'Active Responders', 'Median Active Responses',
                                        'Call Commit', 'Call Follow-through',
                                        'Letter Commit', 'Letter Follow-through',
                                        'Meeting Commit', 'Meeting Follow-through']
                        st.dataframe(
                            variant_summary_df[display_cols],
                            width='stretch',
                            column_config={
                                'Variant': st.column_config.TextColumn('Variant', help='Message variant name from outgoing messages'),
                                'Total Recipients': st.column_config.NumberColumn('Total Recipients', help='People who received this variant (with valid phone numbers)'),
                                'Total Responders': st.column_config.NumberColumn('Total Responders', help='People who sent at least one reply'),
                                'Reply Rate': st.column_config.TextColumn('Reply Rate', help='Substantive reply rate — denominator: Total Recipients'),
                                'Opt-out Rate': st.column_config.TextColumn('Opt-out Rate', help='Opt-out rate — denominator: Total Recipients'),
                                'Active Responders': st.column_config.TextColumn('Active Responders', help='Responders who did not opt out — denominator: Total Recipients'),
                                'Median Active Responses': st.column_config.NumberColumn('Median Active Responses', help='Active response messages per active conversation'),
                                'Call Commit': st.column_config.TextColumn('Call Commit', help='Committed to making a phone call — denominator: Active Responders'),
                                'Call Follow-through': st.column_config.TextColumn('Call Follow-through', help='Reported making the call — denominator: Call Commit count'),
                                'Letter Commit': st.column_config.TextColumn('Letter Commit', help='Committed to writing a letter or email — denominator: Active Responders'),
                                'Letter Follow-through': st.column_config.TextColumn('Letter Follow-through', help='Reported sending the letter/email — denominator: Letter Commit count'),
                                'Meeting Commit': st.column_config.TextColumn('Meeting Commit', help='Committed to attending an in-person meeting — denominator: Active Responders'),
                                'Meeting Follow-through': st.column_config.TextColumn('Meeting Follow-through', help='Reported attending the meeting — denominator: Meeting Commit count'),
                            },
                            hide_index=True,
                        )

                        # Export results for sharing (include person names for standalone viewing)
                        _person_names = {}
                        for pid in person_variant_map:
                            name = get_person_name(pid, people_df)
                            if name:
                                _person_names[str(pid)] = name
                        # Build reply counts for export (from messages_df if available, else from cached)
                        _export_reply_counts = messages_df[messages_df['direction'] == 'incoming'].groupby('person_id').size().to_dict() if 'direction' in messages_df.columns else {}
                        export_eval_data = {
                            'eval_results': {str(k): v for k, v in eval_results.items()},
                            'person_variant_map': {str(k): v for k, v in person_variant_map.items()},
                            'responder_ids': [str(x) for x in st.session_state.get('variant_eval_responder_ids', set())],
                            'optout_ids': [str(x) for x in st.session_state.get('variant_eval_optout_ids', set())],
                            'person_reply_counts': {str(k): int(v) for k, v in _export_reply_counts.items()},
                            'person_names': _person_names,
                            'metadata': {
                                'timestamp': pd.Timestamp.now().isoformat(),
                                'provider': llm_provider if llm_provider else 'imported',
                                'n_conversations': len(eval_results),
                            }
                        }
                        col_export, col_rerun = st.columns(2)
                        with col_export:
                            st.download_button(
                                "⬇️ Export Variant Eval Results",
                                data=json.dumps(export_eval_data, default=str),
                                file_name=f"variant_eval_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                        with col_rerun:
                            if st.button("🔄 Re-run with AI", key="rerun_variant_eval"):
                                for k in ['variant_eval_cache_key', 'variant_eval_results', 'variant_eval_aggregated',
                                           'variant_eval_responder_ids', 'variant_eval_optout_ids']:
                                    st.session_state.pop(k, None)
                                st.rerun()

                        # Expandable per-variant details (active responders only)
                        display_responder_ids = st.session_state.get('variant_eval_responder_ids', set())
                        display_optout_ids = st.session_state.get('variant_eval_optout_ids', set())
                        for variant_name in variant_names:
                            variant_people = [pid for pid, v in person_variant_map.items() if v == variant_name]
                            # Filter to active responders: replied and not opted out
                            variant_active = {
                                pid: eval_results[pid] for pid in variant_people
                                if pid in eval_results
                                and pid in display_responder_ids
                                and pid not in display_optout_ids
                            }

                            with st.expander(f"View {variant_name} - {len(variant_active)} active responders"):
                                detail_rows = []
                                for pid, r in variant_active.items():
                                    person_name = get_person_name(pid, people_df) or f"Person {pid}"
                                    row = {'Person': person_name}
                                    for action in ACTION_TYPES:
                                        label = action.title()
                                        commit_key = f'{action}_commitment'
                                        follow_key = f'{action}_followthrough'
                                        commit_data = r.get(commit_key, {})
                                        follow_data = r.get(follow_key, {})
                                        row[f'{label} Commit'] = f"{commit_data.get('answer', 'N/A')} ({commit_data.get('confidence', 0)}%)"
                                        row[f'{label} Follow-through'] = f"{follow_data.get('answer', 'N/A')} ({follow_data.get('confidence', 0)}%)"
                                        if commit_data.get('answer') == 'YES' and commit_data.get('summary'):
                                            row[f'{label} Commit'] += f" - {commit_data['summary']}"
                                        if follow_data.get('answer') == 'YES' and follow_data.get('summary'):
                                            row[f'{label} Follow-through'] += f" - {follow_data['summary']}"
                                    detail_rows.append(row)
                                if detail_rows:
                                    st.dataframe(pd.DataFrame(detail_rows), width='stretch', hide_index=True)
                                else:
                                    st.caption("No active responders for this variant.")

    except Exception as e:
        import traceback
        st.error(f"Error loading data: {str(e)}")
        st.code(traceback.format_exc())
        st.info("Please ensure your CSV files have the expected column structure.")

else:
    st.header("📊 Import Variant Evaluation Results")
    st.caption("Have a results JSON from a previous run? Upload it here — no CSV files needed.")
    json_only_file = st.file_uploader(
        "Drop JSON file here", type=['json'],
        key="json_only_import"
    )

    if json_only_file:
        try:
            imported_data = json.loads(json_only_file.read())
            eval_results = {k: v for k, v in imported_data['eval_results'].items()}
            person_variant_map = imported_data['person_variant_map']
            responder_pids = set(imported_data.get('responder_ids', []))
            variant_optout_ids = set(imported_data.get('optout_ids', []))
            variant_names = sorted(set(person_variant_map.values()))

            _imported_reply_counts = imported_data.get('person_reply_counts', {})
            variant_summary_df = aggregate_variant_metrics(eval_results, person_variant_map, responder_pids, variant_optout_ids, _imported_reply_counts)

            meta = imported_data.get('metadata', {})
            st.success(f"Loaded {meta.get('n_conversations', len(eval_results))} evaluation results (from {meta.get('timestamp', 'unknown')}, via {meta.get('provider', 'unknown')})")

            # Display results
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Variants", len(variant_names))
            with m2:
                st.metric("Conversations Evaluated", len(eval_results))
            with m3:
                avg_reply = variant_summary_df['_reply_rate'].mean()
                st.metric("Avg Reply Rate", f"{avg_reply:.1f}%")

            display_cols = ['Variant', 'Total Recipients', 'Total Responders', 'Reply Rate', 'Opt-out Rate',
                            'Active Responders', 'Median Active Responses',
                            'Call Commit', 'Call Follow-through',
                            'Letter Commit', 'Letter Follow-through',
                            'Meeting Commit', 'Meeting Follow-through']
            st.dataframe(
                variant_summary_df[display_cols],
                width='stretch',
                column_config={
                    'Variant': st.column_config.TextColumn('Variant', help='Message variant name from outgoing messages'),
                    'Total Recipients': st.column_config.NumberColumn('Total Recipients', help='People who received this variant (with valid phone numbers)'),
                    'Total Responders': st.column_config.NumberColumn('Total Responders', help='People who sent at least one reply'),
                    'Reply Rate': st.column_config.TextColumn('Reply Rate', help='Substantive reply rate — denominator: Total Recipients'),
                    'Opt-out Rate': st.column_config.TextColumn('Opt-out Rate', help='Opt-out rate — denominator: Total Recipients'),
                    'Active Responders': st.column_config.TextColumn('Active Responders', help='Responders who did not opt out — denominator: Total Recipients'),
                    'Median Active Responses': st.column_config.NumberColumn('Median Active Responses', help='Active response messages per active conversation'),
                    'Call Commit': st.column_config.TextColumn('Call Commit', help='Committed to making a phone call — denominator: Active Responders'),
                    'Call Follow-through': st.column_config.TextColumn('Call Follow-through', help='Reported making the call — denominator: Call Commit count'),
                    'Letter Commit': st.column_config.TextColumn('Letter Commit', help='Committed to writing a letter or email — denominator: Active Responders'),
                    'Letter Follow-through': st.column_config.TextColumn('Letter Follow-through', help='Reported sending the letter/email — denominator: Letter Commit count'),
                    'Meeting Commit': st.column_config.TextColumn('Meeting Commit', help='Committed to attending an in-person meeting — denominator: Active Responders'),
                    'Meeting Follow-through': st.column_config.TextColumn('Meeting Follow-through', help='Reported attending the meeting — denominator: Meeting Commit count'),
                },
                hide_index=True,
            )

            # Per-variant details
            _standalone_names = imported_data.get('person_names', {})
            for vname in variant_names:
                v_people = [pid for pid, v in person_variant_map.items() if v == vname]
                v_active = {
                    pid: eval_results[pid] for pid in v_people
                    if pid in eval_results and pid in responder_pids and pid not in variant_optout_ids
                }
                with st.expander(f"View {vname} - {len(v_active)} active responders"):
                    detail_rows = []
                    for pid, r in v_active.items():
                        row = {'Person': _standalone_names.get(str(pid), f"Person {pid}")}
                        for action in ACTION_TYPES:
                            label = action.title()
                            commit_data = r.get(f'{action}_commitment', {})
                            follow_data = r.get(f'{action}_followthrough', {})
                            row[f'{label} Commit'] = f"{commit_data.get('answer', 'N/A')} ({commit_data.get('confidence', 0)}%)"
                            row[f'{label} Follow-through'] = f"{follow_data.get('answer', 'N/A')} ({follow_data.get('confidence', 0)}%)"
                            if commit_data.get('answer') == 'YES' and commit_data.get('summary'):
                                row[f'{label} Commit'] += f" - {commit_data['summary']}"
                            if follow_data.get('answer') == 'YES' and follow_data.get('summary'):
                                row[f'{label} Follow-through'] += f" - {follow_data['summary']}"
                        detail_rows.append(row)
                    if detail_rows:
                        st.dataframe(pd.DataFrame(detail_rows), width='stretch', hide_index=True)
                    else:
                        st.caption("No active responders for this variant.")
        except Exception as e:
            st.error(f"Error loading evaluation results: {e}")
    if not json_only_file:
        st.markdown("---")
        st.header("📂 Upload Files for Your Own Analysis")
        st.info("👈 Upload Messages CSV file(s) and a People CSV file in the sidebar to begin")
        with st.expander("Expected CSV Structure"):
            st.markdown("""
            **Messages CSV(s)** — upload one or more of these export types:

            *Campaign Export:*
            `id, conversation_id, message_variant_name, direction, body, from, to, created_at, ...`

            *Flow Export:*
            Same as Campaign, plus: `automation_id, automation_name, broadcast_id, broadcast_name, ...`

            *All Messages Export:*
            `id, conversation_id, direction, body, from, to, person_id, created_at, ...`

            When multiple files are uploaded, they are merged by message `id`. This lets you
            combine `message_variant_name` (from Campaign/Flow) with `person_id` (from All Messages).

            **People CSV should contain:**
            - id (person identifier)
            - phone/phone_number/mobile/cell (phone number)
            - name or first_name/last_name (optional)
            - tags (optional, comma-separated)

            *Note: People records can be auto-generated from phone numbers when not in the people file.*
            """)