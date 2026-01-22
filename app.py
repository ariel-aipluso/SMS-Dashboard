import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re

st.set_page_config(
    page_title="SMS Campaign Analytics",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📱 SMS Campaign Analytics Dashboard")
st.markdown("*Interactive SMS Campaign Analysis Tool*")
st.caption("Version: Updated with time-based opt-out analysis")

# File upload section
st.sidebar.header("📂 Data Upload")
messages_file = st.sidebar.file_uploader("Upload Messages CSV", type=['csv'])
people_file = st.sidebar.file_uploader("Upload People CSV", type=['csv'])

if messages_file and people_file:
    # Load data
    try:
        messages_df = pd.read_csv(messages_file)
        people_df = pd.read_csv(people_file)

        # Convert datetime columns
        if 'created_at' in messages_df.columns:
            messages_df['created_at'] = pd.to_datetime(messages_df['created_at'])
        if 'sent_at' in messages_df.columns:
            messages_df['sent_at'] = pd.to_datetime(messages_df['sent_at'])

        st.success(f"✅ Loaded {len(messages_df)} messages and {len(people_df)} people records")

        # Data overview
        st.header("📊 Campaign Summary")

        col1, col2, col3, col4, col5 = st.columns(5)

        # Calculate metrics - exclude people without phone numbers
        # Find people with valid phone numbers
        people_with_phones = people_df.copy()
        phone_column = None

        # Detect phone column
        for phone_col in ['phone', 'phone_number', 'mobile', 'cell']:
            if phone_col in people_df.columns:
                phone_column = phone_col
                break

        if phone_column:
            # Filter to only people with non-empty phone numbers
            people_with_phones = people_df[
                people_df[phone_column].notna() &
                (people_df[phone_column] != '') &
                (people_df[phone_column].astype(str).str.strip() != '')
            ]

            # Also exclude landline and fixed VoIP numbers that can't receive SMS
            if 'phone_number_type' in people_df.columns:
                people_with_phones = people_with_phones[
                    ~people_with_phones['phone_number_type'].isin(['landline', 'fixedVoip'])
                ]

            contactable_recipients = len(people_with_phones)
        else:
            # If no phone column found, assume all can be contacted
            contactable_recipients = len(people_df)

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

        # Calculate responders
        all_responders = set(messages_df[messages_df['direction'] == 'incoming']['person_id'].unique())
        total_responders = len(all_responders)

        # Calculate active responders (responded but didn't opt out)
        active_responders_set = all_responders - opted_out_people_set
        active_responders = len(active_responders_set)

        # Response rate based on total responders
        response_rate = (total_responders / contactable_recipients * 100) if contactable_recipients > 0 else 0

        with col1:
            st.metric("Total Recipients", f"{total_recipients:,}")
            if phone_column and contactable_recipients != total_recipients:
                st.caption(f"📱 {contactable_recipients:,} with textable phone numbers")
        with col2:
            st.metric("Response Rate", f"{response_rate:.1f}%")
            st.caption(f"Based on {contactable_recipients:,} textable numbers")
        with col3:
            st.metric("Total Responders", f"{total_responders:,}")
        with col4:
            st.metric("Opt-outs", f"{opt_outs:,}")
        with col5:
            st.metric("Active Responders", f"{active_responders:,}")
            st.caption("Responded but didn't opt out")

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
        st.caption("Detecting STOP messages and analyzing timing relative to initial broadcast")

        if 'body' in messages_df.columns:
            # Find opt-out messages by looking for STOP keywords
            stop_keywords = ['stop', 'unsubscribe', 'opt out', 'opt-out', 'remove', 'quit', 'cancel']

            # Find incoming messages that contain stop words
            incoming_messages = messages_df[messages_df['direction'] == 'incoming'].copy()
            incoming_messages['is_stop'] = incoming_messages['body'].str.lower().str.contains(
                '|'.join(stop_keywords), na=False
            )

            stop_messages = incoming_messages[incoming_messages['is_stop']]
            opted_out_people = stop_messages['person_id'].unique().tolist()

            if opted_out_people:
                # Summary stats first (matching committed responders format)
                col1, col2 = st.columns(2)
                with col1:
                    total_opt_outs = len(opted_out_people)
                    st.metric("Opt-outs", f"{total_opt_outs}")
                with col2:
                    opt_out_rate = (total_opt_outs / contactable_recipients * 100) if contactable_recipients > 0 else 0
                    st.metric("Opt-out Rate", f"{opt_out_rate:.1f}%")
                    st.caption("% of textable numbers who sent STOP")

                # Analyze message sequence position when people opt out
                opt_out_positions = []
                opt_out_details = []

                for person_id in opted_out_people:
                    person_messages = messages_df[messages_df['person_id'] == person_id].sort_values('created_at')

                    # Find the STOP message for this person
                    person_stop_msg = stop_messages[stop_messages['person_id'] == person_id].iloc[0]
                    stop_time = person_stop_msg['created_at']

                    # Count messages before the STOP
                    messages_before_stop = person_messages[
                        person_messages['created_at'] <= stop_time
                    ]

                    # Count outgoing (bot) messages before STOP
                    bot_messages_before = len(messages_before_stop[messages_before_stop['direction'] == 'outgoing'])

                    # Find first bot message time for time calculation
                    first_bot_msg = person_messages[person_messages['direction'] == 'outgoing']
                    if len(first_bot_msg) > 0:
                        first_bot_time = first_bot_msg.iloc[0]['created_at']

                        if pd.notna(first_bot_time) and pd.notna(stop_time):
                            time_diff = stop_time - first_bot_time
                            hours = time_diff.total_seconds() / 3600

                            # Categorize by message position
                            if bot_messages_before == 1:
                                position_category = "After 1st message"
                            elif bot_messages_before <= 3:
                                position_category = "After 2-3 messages"
                            elif bot_messages_before <= 5:
                                position_category = "After 4-5 messages"
                            else:
                                position_category = "After 6+ messages"

                            opt_out_positions.append(position_category)
                            opt_out_details.append({
                                'person_id': person_id,
                                'messages_before_stop': bot_messages_before,
                                'hours_elapsed': hours,
                                'position_category': position_category,
                                'stop_message': person_stop_msg['body'],
                                'stop_time': stop_time
                            })

                # Collapsible detailed analysis section
                with st.expander(f"View {len(opted_out_people)} Opt-out Analysis Details"):
                    if opt_out_positions:
                        position_counts = pd.Series(opt_out_positions).value_counts()

                        col1, col2 = st.columns(2)

                        with col1:
                            fig = px.bar(
                                x=position_counts.index,
                                y=position_counts.values,
                                title="Opt-out Position in Message Sequence",
                                labels={'x': 'Message Position', 'y': 'Number of Opt-outs'}
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            st.subheader("Key Insights")

                            # Calculate statistics
                            opt_out_df = pd.DataFrame(opt_out_details)
                            avg_messages = opt_out_df['messages_before_stop'].mean()
                            avg_hours = opt_out_df['hours_elapsed'].mean()

                            st.metric("Average messages before STOP", f"{avg_messages:.1f}")
                            st.metric("Average time to STOP", f"{avg_hours:.1f} hours")

                            # Early opt-outs warning
                            early_opts = len(opt_out_df[opt_out_df['messages_before_stop'] == 1])
                            if early_opts > 0:
                                st.warning(f"⚠️ {early_opts} people stopped after just 1 message")

                        # Detailed table
                        st.subheader("Individual Opt-out Details")
                        opt_out_summary = pd.DataFrame(opt_out_details)

                        # Add person names to the summary
                        def get_person_name_for_table(person_id):
                            if 'name' in people_df.columns:
                                person_row = people_df[people_df['id'] == person_id]
                                if not person_row.empty and pd.notna(person_row['name'].iloc[0]):
                                    return person_row['name'].iloc[0]
                            elif 'first_name' in people_df.columns or 'last_name' in people_df.columns:
                                person_row = people_df[people_df['id'] == person_id]
                                if not person_row.empty:
                                    first_name = person_row.get('first_name', pd.Series([None])).iloc[0] if 'first_name' in people_df.columns else ""
                                    last_name = person_row.get('last_name', pd.Series([None])).iloc[0] if 'last_name' in people_df.columns else ""
                                    if pd.notna(first_name) or pd.notna(last_name):
                                        return f"{first_name or ''} {last_name or ''}".strip()
                            return f"Person {person_id}"

                        opt_out_summary['person_name'] = opt_out_summary['person_id'].apply(get_person_name_for_table)
                        opt_out_summary['hours_rounded'] = opt_out_summary['hours_elapsed'].round(1)

                        display_cols = ['person_name', 'messages_before_stop', 'hours_rounded', 'stop_message']
                        st.dataframe(opt_out_summary[display_cols])
            else:
                st.info("No STOP messages found in the conversation data")

        # Committed responders detection
        st.header("✅ Committed Responders")

        commitment_keywords = ["yes", "i will", "i'll", "count me in", "sign me up", "i'm in", "absolutely", "definitely", "for sure", "commit", "committed", "attend", "participate", "join", "coming", "be there"]

        if 'body' in messages_df.columns:
            incoming_messages = messages_df[messages_df['direction'] == 'incoming']

            # Find unique people who sent commitment messages with their first occurrence
            committed_people = []
            seen_people = set()

            for _, msg in incoming_messages.iterrows():
                if any(keyword in str(msg['body']).lower() for keyword in commitment_keywords):
                    if msg['person_id'] not in seen_people:
                        seen_people.add(msg['person_id'])

                        # Get person name
                        person_name = "Unknown"
                        if 'name' in people_df.columns:
                            person_row = people_df[people_df['id'] == msg['person_id']]
                            if not person_row.empty and pd.notna(person_row['name'].iloc[0]):
                                person_name = person_row['name'].iloc[0]
                        elif 'first_name' in people_df.columns or 'last_name' in people_df.columns:
                            person_row = people_df[people_df['id'] == msg['person_id']]
                            if not person_row.empty:
                                first_name = person_row.get('first_name', pd.Series([None])).iloc[0] if 'first_name' in people_df.columns else ""
                                last_name = person_row.get('last_name', pd.Series([None])).iloc[0] if 'last_name' in people_df.columns else ""
                                if pd.notna(first_name) or pd.notna(last_name):
                                    person_name = f"{first_name or ''} {last_name or ''}".strip()

                        if person_name == "Unknown" or person_name == "":
                            person_name = f"Person {msg['person_id']}"

                        committed_people.append({
                            'person_name': person_name,
                            'person_id': msg['person_id'],
                            'commitment_date': msg['created_at'],
                            'raw_message': msg['body']
                        })

            if committed_people:
                # Summary stats first
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
                    # Display as a clean table
                    committed_df = pd.DataFrame(committed_people)

                    # Format the date for display
                    committed_df['commitment_date_formatted'] = committed_df['commitment_date'].dt.strftime('%Y-%m-%d %H:%M')

                    # Display table with raw message
                    display_df = committed_df[['person_name', 'commitment_date_formatted', 'raw_message']].copy()
                    display_df.columns = ['Person', 'Commitment Date', 'Message']

                    st.dataframe(display_df, use_container_width=True)
            else:
                st.info("No action commitments detected in the conversation data")

        # Network multiplier detection
        st.header("🌐 Network Multiplier Detection")

        multiplier_keywords = ["invite", "invited", "bring", "friend", "friends", "family", "neighbor", "tell"]

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
                        person_name = "Unknown"
                        if 'name' in people_df.columns:
                            person_row = people_df[people_df['id'] == msg['person_id']]
                            if not person_row.empty and pd.notna(person_row['name'].iloc[0]):
                                person_name = person_row['name'].iloc[0]
                        elif 'first_name' in people_df.columns or 'last_name' in people_df.columns:
                            person_row = people_df[people_df['id'] == msg['person_id']]
                            if not person_row.empty:
                                first_name = person_row.get('first_name', pd.Series([None])).iloc[0] if 'first_name' in people_df.columns else ""
                                last_name = person_row.get('last_name', pd.Series([None])).iloc[0] if 'last_name' in people_df.columns else ""
                                if pd.notna(first_name) or pd.notna(last_name):
                                    person_name = f"{first_name or ''} {last_name or ''}".strip()

                        if person_name == "Unknown" or person_name == "":
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
        # Use the same phone detection logic as the summary
        contactable_people_df = people_df.copy()
        phone_column = None

        for phone_col in ['phone', 'phone_number', 'mobile', 'cell']:
            if phone_col in people_df.columns:
                phone_column = phone_col
                break

        if phone_column:
            contactable_people_df = people_df[
                people_df[phone_column].notna() &
                (people_df[phone_column] != '') &
                (people_df[phone_column].astype(str).str.strip() != '')
            ]

            # Also exclude landline and fixed VoIP numbers that can't receive SMS
            if 'phone_number_type' in people_df.columns:
                contactable_people_df = contactable_people_df[
                    ~contactable_people_df['phone_number_type'].isin(['landline', 'fixedVoip'])
                ]

        all_contactable = set(contactable_people_df['id'].tolist())
        responders = set(messages_df[messages_df['direction'] == 'incoming']['person_id'].unique())
        non_responders = all_contactable - responders

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
                if 'name' in people_df.columns:
                    person_row = people_df[people_df['id'] == person_id]
                    if not person_row.empty and pd.notna(person_row['name'].iloc[0]):
                        return f"{person_row['name'].iloc[0]} (ID: {person_id})"
                elif 'first_name' in people_df.columns or 'last_name' in people_df.columns:
                    person_row = people_df[people_df['id'] == person_id]
                    if not person_row.empty:
                        first_name = person_row.get('first_name', pd.Series([None])).iloc[0] if 'first_name' in people_df.columns else ""
                        last_name = person_row.get('last_name', pd.Series([None])).iloc[0] if 'last_name' in people_df.columns else ""
                        if pd.notna(first_name) or pd.notna(last_name):
                            full_name = f"{first_name or ''} {last_name or ''}".strip()
                            return f"{full_name} (ID: {person_id})"
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
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure your CSV files have the expected column structure.")

else:
    st.info("👆 Please upload both Messages and People CSV files to begin analysis")

    with st.expander("Expected CSV Structure"):
        st.markdown("""
        **Messages CSV should contain:**
        - person_id
        - body (message content)
        - direction (incoming/outgoing)
        - created_at or sent_at (timestamp)

        **People CSV should contain:**
        - id (person identifier)
        - opted_out (boolean)
        - tags (optional, comma-separated)
        """)