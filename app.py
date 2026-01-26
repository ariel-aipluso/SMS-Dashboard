import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
import os
import json
import subprocess

# Optional: LLM providers for AI-based verification
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

st.set_page_config(
    page_title="SMS Campaign Analytics",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_version_date():
    """Get the last git commit date for version display."""
    try:
        # Get the last commit date in a readable format
        result = subprocess.run(
            ['git', 'log', '-1', '--format=%cd', '--date=format:%B %d, %Y'],
            capture_output=True,
            text=True,
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
st.sidebar.header("📂 Data Upload")
messages_file = st.sidebar.file_uploader("Upload Messages CSV", type=['csv'])
people_file = st.sidebar.file_uploader("Upload People CSV", type=['csv'])

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
    person_row = people_df[people_df['id'] == person_id]
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

# LLM-based verification using Claude or OpenAI (conversation-aware)
def format_conversation_for_llm(person_messages_df):
    """Format a person's conversation for LLM analysis."""
    lines = []
    for _, msg in person_messages_df.iterrows():
        direction = "THEM" if msg['direction'] == 'incoming' else "YOU"
        body = msg.get('body', '') or ''
        lines.append(f"{direction}: {body}")
    return "\n".join(lines)

PROMPT_VERSION = "v5"  # Increment this when changing the prompt to bust cache

# Default prompts for AI verification
DEFAULT_OPTOUT_PROMPT = """Look at this SMS conversation. Did the person (THEM) send "STOP" to unsubscribe?

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

DEFAULT_COMMITMENT_PROMPT = """Analyze this SMS conversation. Did the person (THEM) make a genuine commitment to take action?

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

# Cache disabled temporarily for debugging
def analyze_conversations_with_anthropic(conversations: dict, api_key: str, custom_prompt: str = None) -> dict:
    """
    Use Claude to analyze conversations INDIVIDUALLY and detect opt-outs.
    """
    st.write(f"**Debug:** analyze_conversations_with_anthropic called with {len(conversations)} conversations")
    st.write(f"**Debug:** ANTHROPIC_AVAILABLE={ANTHROPIC_AVAILABLE}, api_key={'SET' if api_key else 'EMPTY'}")

    if not ANTHROPIC_AVAILABLE or not api_key:
        st.write("**Debug:** Returning early - anthropic not available or no API key")
        return {}

    client = anthropic.Anthropic(api_key=api_key)
    results = {}

    # Initialize debug storage - store detailed info for each call
    st.session_state.debug_llm_calls = []
    st.write(f"**Debug:** Starting to process {len(conversations)} conversations...")

    # Use custom prompt or default
    prompt_template = custom_prompt if custom_prompt else DEFAULT_OPTOUT_PROMPT

    # Process each conversation individually for accuracy
    for person_id, conversation in conversations.items():
        prompt = prompt_template.format(conversation=conversation)

        try:
            response = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = response.content[0].text.strip().upper()
            opted_out = response_text.startswith("YES")

            st.write(f"**Debug API Response:** Person {person_id} -> `{response_text}` -> opted_out={opted_out}")

            # Store detailed debug info
            st.session_state.debug_llm_calls.append({
                'person_id': person_id,
                'conversation': conversation,
                'raw_response': response_text,
                'opted_out': opted_out
            })

            results[person_id] = {"opted_out": opted_out, "opt_out_message": None}

        except Exception as e:
            st.warning(f"Anthropic API error for {person_id}: {e}")
            st.session_state.debug_llm_calls.append({
                'person_id': person_id,
                'conversation': conversation,
                'raw_response': f"ERROR: {e}",
                'opted_out': "unknown"
            })
            results[person_id] = {"opted_out": "unknown", "opt_out_message": None, "error": str(e)}

    return results

# Cache disabled temporarily for debugging
def analyze_conversations_with_openai(conversations: dict, api_key: str, custom_prompt: str = None) -> dict:
    """
    Use OpenAI to analyze conversations INDIVIDUALLY and detect opt-outs.
    """
    if not OPENAI_AVAILABLE or not api_key:
        return {}

    client = openai.OpenAI(api_key=api_key)
    results = {}

    # Use custom prompt or default
    prompt_template = custom_prompt if custom_prompt else DEFAULT_OPTOUT_PROMPT

    for person_id, conversation in conversations.items():
        prompt = prompt_template.format(conversation=conversation)

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = response.choices[0].message.content.strip().upper()
            opted_out = response_text.startswith("YES")

            results[person_id] = {"opted_out": opted_out, "opt_out_message": None}

        except Exception as e:
            st.warning(f"OpenAI API error for {person_id}: {e}")
            results[person_id] = {"opted_out": "unknown", "opt_out_message": None, "error": str(e)}

    return results

# Commitment analysis functions
def analyze_commitments_with_anthropic(conversations: dict, api_key: str, custom_prompt: str = None) -> dict:
    """
    Use Claude to analyze conversations and detect genuine commitments to action.
    """
    if not ANTHROPIC_AVAILABLE or not api_key:
        return {}

    client = anthropic.Anthropic(api_key=api_key)
    results = {}

    # Initialize debug storage
    if 'debug_commitment_calls' not in st.session_state:
        st.session_state.debug_commitment_calls = []
    st.session_state.debug_commitment_calls = []

    # Use custom prompt or default
    prompt_template = custom_prompt if custom_prompt else DEFAULT_COMMITMENT_PROMPT

    for person_id, conversation in conversations.items():
        prompt = prompt_template.format(conversation=conversation)

        try:
            response = client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = response.content[0].text.strip().upper()
            is_committed = response_text.startswith("YES")

            st.session_state.debug_commitment_calls.append({
                'person_id': person_id,
                'conversation': conversation,
                'raw_response': response_text,
                'is_committed': is_committed
            })

            results[person_id] = {"is_committed": is_committed}

        except Exception as e:
            st.session_state.debug_commitment_calls.append({
                'person_id': person_id,
                'conversation': conversation,
                'raw_response': f"ERROR: {e}",
                'is_committed': "unknown"
            })
            results[person_id] = {"is_committed": "unknown", "error": str(e)}

    return results

def analyze_commitments_with_openai(conversations: dict, api_key: str, custom_prompt: str = None) -> dict:
    """
    Use OpenAI to analyze conversations and detect genuine commitments to action.
    """
    if not OPENAI_AVAILABLE or not api_key:
        return {}

    client = openai.OpenAI(api_key=api_key)
    results = {}

    if 'debug_commitment_calls' not in st.session_state:
        st.session_state.debug_commitment_calls = []
    st.session_state.debug_commitment_calls = []

    # Use custom prompt or default
    prompt_template = custom_prompt if custom_prompt else DEFAULT_COMMITMENT_PROMPT

    for person_id, conversation in conversations.items():
        prompt = prompt_template.format(conversation=conversation)

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = response.choices[0].message.content.strip().upper()
            is_committed = response_text.startswith("YES")

            st.session_state.debug_commitment_calls.append({
                'person_id': person_id,
                'conversation': conversation,
                'raw_response': response_text,
                'is_committed': is_committed
            })

            results[person_id] = {"is_committed": is_committed}

        except Exception as e:
            st.session_state.debug_commitment_calls.append({
                'person_id': person_id,
                'conversation': conversation,
                'raw_response': f"ERROR: {e}",
                'is_committed': "unknown"
            })
            results[person_id] = {"is_committed": "unknown", "error": str(e)}

    return results

def detect_optouts(messages_df, use_llm=False, llm_provider=None, api_key=None, custom_prompt=None):
    """
    Detect opt-out messages using keyword matching and optionally LLM verification.
    LLM is only used to verify keyword-detected opt-outs (cost optimization).
    """
    stop_keywords = ['stop', 'unsubscribe', 'opt out', 'opt-out', 'remove', 'quit', 'cancel']

    incoming_messages = messages_df[messages_df['direction'] == 'incoming'].copy()

    # Apply keyword matching first
    incoming_messages['is_stop_keyword'] = incoming_messages['body'].str.lower().str.contains(
        '|'.join(stop_keywords), na=False
    )

    incoming_messages['is_stop_llm_verified'] = False
    incoming_messages['is_ai_rejected'] = False  # Track false positives
    incoming_messages['detection_method'] = None

    # Get people flagged by keyword matching
    keyword_opted_out_people = set(incoming_messages[incoming_messages['is_stop_keyword']]['person_id'].unique())

    st.write(f"**Debug Flow:** use_llm={use_llm}, llm_provider={llm_provider}, api_key={'SET' if api_key else 'EMPTY'}, keyword_flagged={len(keyword_opted_out_people) if keyword_opted_out_people else 0}")

    if use_llm and llm_provider and api_key and keyword_opted_out_people:
        # Only analyze conversations that were flagged by keywords (cost optimization)
        conversations = {}
        for person_id in keyword_opted_out_people:
            person_msgs = messages_df[messages_df['person_id'] == person_id].sort_values('created_at')
            conversations[person_id] = format_conversation_for_llm(person_msgs)

        st.write(f"**Debug:** Built {len(conversations)} conversations to analyze")
        provider_name = "Claude" if llm_provider == "anthropic" else "GPT-4o-mini"
        with st.spinner(f"Verifying {len(conversations)} keyword-flagged conversations with {provider_name}..."):
            if llm_provider == "anthropic":
                llm_results = analyze_conversations_with_anthropic(conversations, api_key, custom_prompt)
            else:
                llm_results = analyze_conversations_with_openai(conversations, api_key, custom_prompt)

        # LLM verification: confirm or reject keyword matches
        verified_count = 0
        rejected_count = 0

        # Process ALL keyword-flagged people, not just those in llm_results
        unknown_count = 0
        for person_id in keyword_opted_out_people:
            person_incoming = incoming_messages[incoming_messages['person_id'] == person_id]
            keyword_flagged_msgs = person_incoming[person_incoming['is_stop_keyword']]

            # Get LLM result, default to "unknown" if not found
            result = llm_results.get(person_id, {"opted_out": "unknown", "opt_out_message": None})
            opted_out_status = result.get('opted_out')

            if opted_out_status == "unknown":
                # API error - keep keyword match but mark as unverified
                for idx in keyword_flagged_msgs.index:
                    incoming_messages.loc[idx, 'is_stop_llm_unknown'] = True
                unknown_count += 1
            elif opted_out_status:
                # LLM confirms this is a real opt-out
                for idx in keyword_flagged_msgs.index:
                    incoming_messages.loc[idx, 'is_stop_llm_verified'] = True
                verified_count += 1
            else:
                # LLM says this is NOT an opt-out (false positive from keywords)
                for idx in keyword_flagged_msgs.index:
                    incoming_messages.loc[idx, 'is_stop_keyword'] = False
                    incoming_messages.loc[idx, 'is_ai_rejected'] = True  # Mark as rejected
                rejected_count += 1

        if verified_count > 0:
            st.success(f"🤖 AI verified {verified_count} opt-out(s)")
        if rejected_count > 0:
            st.info(f"🤖 AI rejected {rejected_count} false positive(s) from keyword matching")
        if unknown_count > 0:
            st.warning(f"⚠️ {unknown_count} conversation(s) could not be analyzed (API error) - kept as keyword match")

        # Debug output - show detailed LLM analysis
        with st.expander("🔍 Debug: LLM Analysis Details", expanded=True):
            if 'debug_llm_calls' in st.session_state and st.session_state.debug_llm_calls:
                st.write("**Each LLM call with conversation and response:**")
                for call in st.session_state.debug_llm_calls:
                    with st.container():
                        if call['opted_out'] == "unknown":
                            status = "⚠️ UNKNOWN (API error)"
                        elif call['opted_out']:
                            status = "✅ OPT-OUT"
                        else:
                            status = "❌ NOT opt-out"
                        st.markdown(f"### Person {call['person_id']}: {status}")
                        st.write(f"**Raw LLM Response:** `{call['raw_response']}`")
                        st.write("**Conversation sent:**")
                        st.code(call['conversation'])
                        st.divider()
            else:
                st.write("No LLM calls recorded")

    # Final opt-out status
    incoming_messages['is_stop'] = incoming_messages['is_stop_keyword']
    incoming_messages['detection_method'] = incoming_messages.apply(
        lambda row: 'AI Rejected' if row.get('is_ai_rejected', False)
                   else ('AI Verified' if row.get('is_stop_llm_verified', False)
                   else ('AI Unknown' if row.get('is_stop_llm_unknown', False)
                   else ('Keyword' if row['is_stop_keyword'] else None))), axis=1
    )

    return incoming_messages

# Sidebar: LLM Configuration
st.sidebar.markdown("---")
st.sidebar.header("🤖 AI Verification")
use_llm_detection = False
llm_provider = None
llm_api_key = None

if ANTHROPIC_AVAILABLE or OPENAI_AVAILABLE:
    use_llm_detection = st.sidebar.checkbox(
        "Enable AI verification",
        help="Uses AI to verify keyword matches and reduce false positives"
    )
    if use_llm_detection:
        st.sidebar.caption("AI verification is used for:")
        st.sidebar.markdown("• Opt-out detection\n• Commitment verification")
        st.sidebar.caption("Enter at least one API key:")

        anthropic_key = None
        openai_key = None

        if ANTHROPIC_AVAILABLE:
            anthropic_key = st.sidebar.text_input(
                "Anthropic API Key",
                type="password",
                help="From console.anthropic.com (uses Claude Haiku)"
            )

        if OPENAI_AVAILABLE:
            openai_key = st.sidebar.text_input(
                "OpenAI API Key",
                type="password",
                help="From platform.openai.com (uses GPT-4o-mini)"
            )

        # Use whichever key is provided (prefer Anthropic if both)
        if anthropic_key:
            llm_provider = "anthropic"
            llm_api_key = anthropic_key
        elif openai_key:
            llm_provider = "openai"
            llm_api_key = openai_key
        else:
            st.sidebar.warning("Enter an API key to enable AI verification")
            use_llm_detection = False

        # Editable prompts section
        with st.sidebar.expander("✏️ Customize AI Prompts"):
            st.caption("Edit the prompts used for AI verification. Use {conversation} as placeholder for the conversation text.")

            custom_optout_prompt = st.text_area(
                "Opt-out Detection Prompt",
                value=st.session_state.get('custom_optout_prompt', DEFAULT_OPTOUT_PROMPT),
                height=200,
                key="optout_prompt_input",
                help="Prompt used to verify opt-out messages"
            )
            st.session_state['custom_optout_prompt'] = custom_optout_prompt

            custom_commitment_prompt = st.text_area(
                "Commitment Verification Prompt",
                value=st.session_state.get('custom_commitment_prompt', DEFAULT_COMMITMENT_PROMPT),
                height=200,
                key="commitment_prompt_input",
                help="Prompt used to verify commitment messages"
            )
            st.session_state['custom_commitment_prompt'] = custom_commitment_prompt

            if st.button("Reset to Defaults", key="reset_prompts"):
                st.session_state['custom_optout_prompt'] = DEFAULT_OPTOUT_PROMPT
                st.session_state['custom_commitment_prompt'] = DEFAULT_COMMITMENT_PROMPT
                st.rerun()
else:
    st.sidebar.info("Install `anthropic` or `openai` package to enable AI verification")

# Get prompts (use defaults if not customized)
optout_prompt_template = st.session_state.get('custom_optout_prompt', DEFAULT_OPTOUT_PROMPT)
commitment_prompt_template = st.session_state.get('custom_commitment_prompt', DEFAULT_COMMITMENT_PROMPT)

if messages_file and people_file:
    # Load data
    try:
        messages_df = pd.read_csv(messages_file)
        people_df = pd.read_csv(people_file)

        # Collect loading messages for collapsed display
        loading_messages = []

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

        # Data overview
        st.header("📊 Campaign Summary")

        col1, col2, col3, col4, col5 = st.columns(5)

        # Calculate metrics - exclude people without phone numbers
        # Find people with valid phone numbers
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

            # Also exclude landline and fixed VoIP numbers that can't receive SMS
            if 'phone_number_type' in people_df.columns:
                people_with_phones = people_with_phones[
                    ~people_with_phones['phone_number_type'].isin(['landline', 'fixedVoip'])
                ]

            contactable_recipients = len(people_with_phones)
        else:
            # If no phone column found, assume all can be contacted
            people_with_phones = people_df
            contactable_recipients = len(people_df)

        # Get IDs of textable people for filtering responders
        textable_people_ids = set(people_with_phones['id'].tolist())

        total_recipients = len(people_df)

        # Calculate opt-outs first
        stop_keywords = ['stop', 'unsubscribe', 'opt out', 'opt-out', 'remove', 'quit', 'cancel']
        opted_out_people_set = set()

        if 'body' in messages_df.columns:
            incoming_stop_messages = messages_df[
                (messages_df['direction'] == 'incoming') &
                (messages_df['body'].str.lower().str.contains('|'.join(stop_keywords), na=False))
            ]
            opted_out_people_set = set(incoming_stop_messages['person_id'].unique())
            opt_outs = len(opted_out_people_set)
        else:
            opt_outs = 0

        # Calculate responders - only count those who are in the textable list
        all_responders = set(messages_df[messages_df['direction'] == 'incoming']['person_id'].unique())
        # Filter to only count responders who were textable
        textable_responders = all_responders & textable_people_ids
        total_responders = len(textable_responders)

        # Calculate active responders (responded but didn't opt out)
        active_responders_set = textable_responders - opted_out_people_set
        active_responders = len(active_responders_set)

        # Calculate textable opted-out people (for active response rate denominator)
        textable_opted_out = opted_out_people_set & textable_people_ids
        textable_non_opted_out = contactable_recipients - len(textable_opted_out)

        # Total Response Rate: all responders / all textable (includes opt-outs)
        total_response_rate = (total_responders / contactable_recipients * 100) if contactable_recipients > 0 else 0

        # Active Response Rate: active responders / textable non-opted-out
        active_response_rate = (active_responders / textable_non_opted_out * 100) if textable_non_opted_out > 0 else 0

        with col1:
            st.metric("Total Recipients", f"{total_recipients:,}")
            if available_phone_cols and contactable_recipients != total_recipients:
                st.caption(f"📱 {contactable_recipients:,} textable")
        with col2:
            st.metric("Total Response Rate", f"{total_response_rate:.1f}%")
            st.caption(f"All responders / {contactable_recipients:,} textable")
        with col3:
            st.metric("Active Response Rate", f"{active_response_rate:.1f}%")
            st.caption(f"Active / {textable_non_opted_out:,} non-opted-out")
        with col4:
            st.metric("Total Responders", f"{total_responders:,}")
            st.caption(f"Active: {active_responders:,}")
        with col5:
            st.metric("Opt-outs", f"{opt_outs:,}")

        # Engagement depth analysis
        st.header("📈 Engagement Depth Analysis")

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
            fig = px.bar(
                x=list(engagement_buckets.keys()),
                y=list(engagement_buckets.values()),
                title="Response Engagement Levels",
                labels={'x': 'Engagement Level', 'y': 'Number of People'}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Response distribution
            if not person_response_counts.empty:
                fig = px.histogram(
                    person_response_counts,
                    x='response_count',
                    nbins=int(min(20, person_response_counts['response_count'].max())),
                    title="Distribution of Response Counts",
                    labels={'response_count': 'Number of Responses', 'count': 'Number of People'}
                )
                st.plotly_chart(fig, use_container_width=True)

        # Opt-out timing analysis
        st.header("⏰ Opt-out Timing Analysis")
        detection_method = "AI + keyword matching" if use_llm_detection else "keyword matching"
        st.caption(f"Analyzing opt-outs by engagement level ({detection_method})")

        if 'body' in messages_df.columns:
            # Detect opt-outs using keyword matching or LLM
            incoming_messages = detect_optouts(messages_df, use_llm=use_llm_detection, llm_provider=llm_provider, api_key=llm_api_key, custom_prompt=optout_prompt_template)
            stop_messages = incoming_messages[incoming_messages['is_stop']]
            opted_out_people = stop_messages['person_id'].unique().tolist()

            # Get AI-rejected false positives for display
            ai_rejected_messages = incoming_messages[incoming_messages['is_ai_rejected'] == True] if 'is_ai_rejected' in incoming_messages.columns else pd.DataFrame()
            ai_rejected_people = ai_rejected_messages['person_id'].unique().tolist() if not ai_rejected_messages.empty else []

            if opted_out_people or ai_rejected_people:
                # Summary stats first (matching committed responders format)
                if ai_rejected_people:
                    col1, col2, col3 = st.columns(3)
                else:
                    col1, col2 = st.columns(2)
                    col3 = None

                with col1:
                    total_opt_outs = len(opted_out_people)
                    st.metric("Verified Opt-outs", f"{total_opt_outs}")
                with col2:
                    opt_out_rate = (total_opt_outs / contactable_recipients * 100) if contactable_recipients > 0 else 0
                    st.metric("Opt-out Rate", f"{opt_out_rate:.1f}%")
                    st.caption("% of textable numbers who sent STOP")
                if col3:
                    with col3:
                        st.metric("AI Rejected (False Positives)", f"{len(ai_rejected_people)}")
                        st.caption("Keyword matches that AI determined weren't opt-outs")

                # Analyze opt-outs by conversation engagement
                opt_out_categories = []
                opt_out_details = []

                for person_id in opted_out_people:
                    person_messages = messages_df[messages_df['person_id'] == person_id].sort_values('created_at')

                    # Find the first STOP message for this person
                    person_stop_msg = stop_messages[stop_messages['person_id'] == person_id].iloc[0]
                    stop_time = person_stop_msg['created_at']

                    # Get all incoming messages for this person
                    person_incoming = person_messages[person_messages['direction'] == 'incoming']

                    # Check if they had any non-STOP incoming messages before opting out
                    # (i.e., did they reply or ask questions before sending STOP?)
                    incoming_before_stop = person_incoming[person_incoming['created_at'] <= stop_time]

                    # Check for non-STOP messages (replies/questions)
                    non_stop_incoming = incoming_before_stop[
                        ~incoming_before_stop['body'].str.lower().str.contains(
                            '|'.join(stop_keywords), na=False
                        )
                    ]

                    had_conversation = len(non_stop_incoming) > 0
                    num_replies_before_stop = len(non_stop_incoming)

                    # Count outgoing (bot) messages before STOP for context
                    messages_before_stop = person_messages[person_messages['created_at'] <= stop_time]
                    bot_messages_before = len(messages_before_stop[messages_before_stop['direction'] == 'outgoing'])

                    # Find first bot message time for time calculation
                    first_bot_msg = person_messages[person_messages['direction'] == 'outgoing']
                    if len(first_bot_msg) > 0:
                        first_bot_time = first_bot_msg.iloc[0]['created_at']

                        if pd.notna(first_bot_time) and pd.notna(stop_time):
                            time_diff = stop_time - first_bot_time
                            hours = time_diff.total_seconds() / 3600

                            # Categorize by conversation engagement
                            if had_conversation:
                                engagement_category = "After conversation"
                            else:
                                engagement_category = "Without conversation"

                            opt_out_categories.append(engagement_category)
                            # Get detection method if available
                            detection = person_stop_msg.get('detection_method', 'Keyword') if 'detection_method' in person_stop_msg.index else 'Keyword'
                            opt_out_details.append({
                                'person_id': person_id,
                                'engagement_category': engagement_category,
                                'replies_before_stop': num_replies_before_stop,
                                'bot_messages_before': bot_messages_before,
                                'hours_elapsed': hours,
                                'stop_message': person_stop_msg['body'],
                                'stop_time': stop_time,
                                'detection_method': detection
                            })

                # Also add AI-rejected false positives to the details
                for person_id in ai_rejected_people:
                    person_messages = messages_df[messages_df['person_id'] == person_id].sort_values('created_at')
                    person_rejected_msg = ai_rejected_messages[ai_rejected_messages['person_id'] == person_id].iloc[0]

                    # Get message info
                    first_bot_msg = person_messages[person_messages['direction'] == 'outgoing']
                    if len(first_bot_msg) > 0:
                        first_bot_time = first_bot_msg.iloc[0]['created_at']
                        msg_time = person_rejected_msg['created_at']

                        if pd.notna(first_bot_time) and pd.notna(msg_time):
                            hours = (msg_time - first_bot_time).total_seconds() / 3600
                            bot_messages_before = len(person_messages[
                                (person_messages['direction'] == 'outgoing') &
                                (person_messages['created_at'] <= msg_time)
                            ])

                            opt_out_details.append({
                                'person_id': person_id,
                                'engagement_category': 'N/A (False Positive)',
                                'replies_before_stop': 0,
                                'bot_messages_before': bot_messages_before,
                                'hours_elapsed': hours,
                                'stop_message': person_rejected_msg['body'],
                                'stop_time': msg_time,
                                'detection_method': 'AI Rejected'
                            })

                # Collapsible detailed analysis section
                total_analyzed = len(opted_out_people) + len(ai_rejected_people)
                expander_label = f"View {len(opted_out_people)} Opt-out(s)" + (f" + {len(ai_rejected_people)} AI-Rejected" if ai_rejected_people else "") + " Details"
                with st.expander(expander_label):
                    if opt_out_categories:
                        category_counts = pd.Series(opt_out_categories).value_counts()

                        col1, col2 = st.columns(2)

                        with col1:
                            # Define colors for categories
                            colors = {'Without conversation': '#EF553B', 'After conversation': '#636EFA'}
                            fig = px.bar(
                                x=category_counts.index,
                                y=category_counts.values,
                                title="Opt-out Breakdown by Engagement",
                                labels={'x': 'Engagement Type', 'y': 'Number of Opt-outs'},
                                color=category_counts.index,
                                color_discrete_map=colors
                            )
                            fig.update_layout(showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            st.subheader("Key Insights")

                            # Calculate statistics
                            opt_out_df = pd.DataFrame(opt_out_details)

                            # Count by category
                            no_convo_count = len(opt_out_df[opt_out_df['engagement_category'] == 'Without conversation'])
                            after_convo_count = len(opt_out_df[opt_out_df['engagement_category'] == 'After conversation'])

                            st.metric("Opted out without conversation", f"{no_convo_count}",
                                     help="People who only sent STOP without any prior replies")
                            st.metric("Opted out after conversation", f"{after_convo_count}",
                                     help="People who replied/asked questions before opting out")

                            avg_hours = opt_out_df['hours_elapsed'].mean()
                            st.metric("Average time to STOP", f"{avg_hours:.1f} hours")

                            # Warning for immediate opt-outs
                            if no_convo_count > 0:
                                pct_immediate = (no_convo_count / len(opt_out_df)) * 100
                                if pct_immediate > 50:
                                    st.warning(f"⚠️ {pct_immediate:.0f}% opted out without any conversation")

                        # Detailed table
                        st.subheader("Individual Opt-out Details")
                        opt_out_summary = pd.DataFrame(opt_out_details)

                        # Add person names to the summary
                        def get_person_name_for_table(person_id):
                            name = get_person_name(person_id, people_df)
                            return name if name else f"Person {person_id}"

                        opt_out_summary['person_name'] = opt_out_summary['person_id'].apply(get_person_name_for_table)
                        opt_out_summary['hours_rounded'] = opt_out_summary['hours_elapsed'].round(1)

                        display_cols = ['person_name', 'engagement_category', 'replies_before_stop', 'bot_messages_before', 'hours_rounded', 'stop_message']
                        col_config = {
                            'person_name': 'Name',
                            'engagement_category': 'Engagement',
                            'replies_before_stop': 'Replies Before STOP',
                            'bot_messages_before': 'Bot Messages',
                            'hours_rounded': 'Hours to STOP',
                            'stop_message': 'STOP Message'
                        }
                        # Add detection method column if LLM detection was used
                        if use_llm_detection and 'detection_method' in opt_out_summary.columns:
                            display_cols.insert(2, 'detection_method')
                            col_config['detection_method'] = 'Detected By'
                        st.dataframe(opt_out_summary[display_cols].rename(columns=col_config))
            else:
                st.info("No STOP messages found in the conversation data")

        # Committed responders detection
        st.header("✅ Committed Responders")

        commitment_keywords = ["yes", "i will", "i'll", "count me in", "sign me up", "i'm in", "absolutely", "definitely", "for sure", "commit", "committed", "attend", "participate", "join", "coming", "be there"]

        if 'body' in messages_df.columns:
            incoming_messages = messages_df[messages_df['direction'] == 'incoming'].copy()

            # Step 1: Keyword matching to find potential commitments
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
                            'detection_method': 'Keyword'
                        })

            # Step 2: LLM verification if enabled
            llm_verified_count = 0
            llm_rejected_count = 0
            llm_unknown_count = 0
            committed_people = []

            if use_llm_detection and llm_provider and api_key and keyword_committed_people:
                # Build conversations for keyword-flagged people
                commitment_conversations = {}
                for person in keyword_committed_people:
                    person_msgs = messages_df[messages_df['person_id'] == person['person_id']].sort_values('created_at')
                    commitment_conversations[person['person_id']] = format_conversation_for_llm(person_msgs)

                provider_name = "Claude" if llm_provider == "anthropic" else "GPT-4o-mini"
                with st.spinner(f"Verifying {len(commitment_conversations)} potential commitments with {provider_name}..."):
                    if llm_provider == "anthropic":
                        llm_results = analyze_commitments_with_anthropic(commitment_conversations, api_key, commitment_prompt_template)
                    else:
                        llm_results = analyze_commitments_with_openai(commitment_conversations, api_key, commitment_prompt_template)

                # Process LLM results
                for person in keyword_committed_people:
                    result = llm_results.get(person['person_id'], {"is_committed": "unknown"})
                    status = result.get('is_committed')

                    if status == "unknown":
                        person['detection_method'] = 'AI Unknown'
                        committed_people.append(person)
                        llm_unknown_count += 1
                    elif status:
                        person['detection_method'] = 'AI Verified'
                        committed_people.append(person)
                        llm_verified_count += 1
                    else:
                        # LLM rejected - don't include in final list
                        llm_rejected_count += 1

                # Show LLM verification results
                if llm_verified_count > 0:
                    st.success(f"🤖 AI verified {llm_verified_count} commitment(s)")
                if llm_rejected_count > 0:
                    st.info(f"🤖 AI rejected {llm_rejected_count} false positive(s) from keyword matching")
                if llm_unknown_count > 0:
                    st.warning(f"⚠️ {llm_unknown_count} commitment(s) could not be analyzed (API error)")

                # Debug output for commitments
                with st.expander("🔍 Debug: Commitment LLM Analysis"):
                    if 'debug_commitment_calls' in st.session_state and st.session_state.debug_commitment_calls:
                        for call in st.session_state.debug_commitment_calls:
                            if call['is_committed'] == "unknown":
                                status = "⚠️ UNKNOWN (API error)"
                            elif call['is_committed']:
                                status = "✅ COMMITTED"
                            else:
                                status = "❌ NOT committed"
                            st.markdown(f"**Person {call['person_id']}:** {status}")
                            st.write(f"Raw Response: `{call['raw_response']}`")
                            st.code(call['conversation'])
                            st.divider()
            else:
                # No LLM - use keyword results directly
                committed_people = keyword_committed_people

            if committed_people:
                # Summary stats
                col1, col2 = st.columns(2)
                with col1:
                    total_committed = len(committed_people)
                    st.metric("Committed Responders", f"{total_committed}")
                with col2:
                    if total_committed > 0:
                        commitment_rate = (total_committed / active_responders * 100) if active_responders > 0 else 0
                        st.metric("Commitment Rate", f"{commitment_rate:.1f}%")
                        st.caption("% of active responders who committed to action")

                # Collapsible table section
                with st.expander(f"View {len(committed_people)} Committed Responder Details"):
                    committed_df = pd.DataFrame(committed_people)
                    committed_df['commitment_date_formatted'] = committed_df['commitment_date'].dt.strftime('%Y-%m-%d %H:%M')

                    display_cols = ['person_name', 'commitment_date_formatted', 'raw_message']
                    col_names = ['Person', 'Commitment Date', 'Message']

                    # Add detection method if LLM was used
                    if use_llm_detection and llm_provider and api_key:
                        display_cols.insert(2, 'detection_method')
                        col_names.insert(2, 'Verified By')

                    display_df = committed_df[display_cols].copy()
                    display_df.columns = col_names

                    st.dataframe(display_df, use_container_width=True)
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

                    st.dataframe(display_df, use_container_width=True)
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

        # Non-responder export
        st.header("📤 Export Non-Responders")

        # Calculate non-responders (only from contactable people)
        # Reuse the textable_people_ids calculated earlier for consistency
        contactable_people_df = people_with_phones

        all_contactable = set(contactable_people_df['id'].tolist())
        # Use textable_responders calculated earlier for consistency
        non_responders = all_contactable - textable_responders

        st.info(f"Found {len(non_responders)} people with textable numbers who did not respond to the broadcast")

        if len(non_responders) > 0:
            # Tag input
            export_tag = st.text_input("Tag for exported non-responders:", value="no_response_v1")

            if st.button("📥 Export Non-Responders as CSV"):
                # Create export dataframe
                non_responder_data = []

                for person_id in non_responders:
                    person_row = people_df[people_df['id'] == person_id]
                    if not person_row.empty:
                        person = person_row.iloc[0]

                        # Extract name and phone data
                        first_name = ""
                        last_name = ""
                        phone = ""

                        # Handle different name column formats
                        if 'first_name' in people_df.columns:
                            first_name = person.get('first_name', '') or ''
                        if 'last_name' in people_df.columns:
                            last_name = person.get('last_name', '') or ''
                        if 'name' in people_df.columns and not first_name and not last_name:
                            full_name = person.get('name', '') or ''
                            name_parts = full_name.split(' ', 1)
                            first_name = name_parts[0] if len(name_parts) > 0 else ''
                            last_name = name_parts[1] if len(name_parts) > 1 else ''

                        # Handle different phone column formats
                        for phone_col in ['phone', 'phone_number', 'mobile', 'cell']:
                            if phone_col in people_df.columns:
                                phone = person.get(phone_col, '') or ''
                                if phone:
                                    break

                        non_responder_data.append({
                            'first_name': first_name,
                            'last_name': last_name,
                            'phone_number': phone,
                            'tag': export_tag
                        })

                if non_responder_data:
                    export_df = pd.DataFrame(non_responder_data)

                    # Convert to CSV
                    csv_data = export_df.to_csv(index=False)

                    st.download_button(
                        label="⬇️ Download Non-Responders CSV",
                        data=csv_data,
                        file_name=f"non_responders_{export_tag}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

                    # Preview the data
                    st.subheader("Preview of Export Data:")
                    st.dataframe(export_df.head(10))
                    st.caption(f"Showing first 10 of {len(export_df)} non-responders")
                else:
                    st.warning("No data available for export")

        # Responder details with conversation viewer
        st.header("💬 Responder Details & Conversations")

        responder_ids = messages_df[messages_df['direction'] == 'incoming']['person_id'].unique()

        if len(responder_ids) > 0:
            # Create a mapping of person_id to name if available
            def get_display_name(person_id):
                name = get_person_name(person_id, people_df)
                if name:
                    return f"{name} (ID: {person_id})"
                return f"Person {person_id}"

            selected_responder = st.selectbox(
                "Select a responder to view conversation",
                responder_ids,
                format_func=get_display_name
            )

            if selected_responder:
                # Get conversation for selected responder
                conversation = messages_df[
                    messages_df['person_id'] == selected_responder
                ].sort_values('created_at')

                st.subheader(f"Conversation with {get_display_name(selected_responder)}")

                for _, msg in conversation.iterrows():
                    direction_icon = "👤" if msg['direction'] == 'incoming' else "🤖"
                    timestamp = msg['created_at'].strftime("%Y-%m-%d %H:%M") if pd.notna(msg['created_at']) else "Unknown time"

                    with st.chat_message("user" if msg['direction'] == 'incoming' else "assistant"):
                        st.write(f"**{direction_icon} {timestamp}**")
                        st.write(msg['body'])

    except Exception as e:
        import traceback
        st.error(f"Error loading data: {str(e)}")
        st.code(traceback.format_exc())
        st.info("Please ensure your CSV files have the expected column structure.")

else:
    st.info("👆 Please upload both Messages and People CSV files to begin analysis")

    with st.expander("Expected CSV Structure"):
        st.markdown("""
        **Messages CSV should contain (Standard Format):**
        - person_id
        - body (message content)
        - direction (incoming/outgoing)
        - created_at or sent_at (timestamp)

        **Alternative Messages CSV Format:**
        - conversation_id
        - from (sender phone number)
        - to (recipient phone number)
        - body (message content)
        - direction (incoming/outgoing)
        - created_at (timestamp)
        - phone_number_type (optional)

        **People CSV should contain:**
        - id (person identifier)
        - phone/phone_number/mobile/cell (phone number)
        - name or first_name/last_name (optional)
        - phone_number_type (optional)
        - opted_out (boolean, optional)
        - tags (optional, comma-separated)

        *Note: If using alternative message format, people records can be auto-generated from phone numbers.*
        """)