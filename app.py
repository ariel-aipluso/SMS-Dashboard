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

        col1, col2, col3, col4 = st.columns(4)

        # Calculate metrics
        total_recipients = len(people_df)
        responded_people = messages_df[messages_df['direction'] == 'incoming']['person_id'].nunique()
        response_rate = (responded_people / total_recipients * 100) if total_recipients > 0 else 0

        # Opt-outs
        opt_outs = len(people_df[people_df['opted_out'] == True]) if 'opted_out' in people_df.columns else 0

        with col1:
            st.metric("Total Recipients", f"{total_recipients:,}")
        with col2:
            st.metric("Response Rate", f"{response_rate:.1f}%")
        with col3:
            st.metric("Responders", f"{responded_people:,}")
        with col4:
            st.metric("Opt-outs", f"{opt_outs:,}")

        # Engagement depth analysis
        st.header("📈 Engagement Depth Analysis")

        # Calculate response counts per person
        person_response_counts = messages_df[messages_df['direction'] == 'incoming'].groupby('person_id').size().reset_index(name='response_count')

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
                    nbins=min(20, person_response_counts['response_count'].max()),
                    title="Distribution of Response Counts",
                    labels={'response_count': 'Number of Responses', 'count': 'Number of People'}
                )
                st.plotly_chart(fig, use_container_width=True)

        # Opt-out timing analysis
        st.header("⏰ Opt-out Timing Analysis")

        if 'opted_out' in people_df.columns and 'body' in messages_df.columns:
            opted_out_people = people_df[people_df['opted_out'] == True]['id'].tolist()

            if opted_out_people:
                # Analyze when people opted out relative to message sequence
                opt_out_analysis = []

                for person_id in opted_out_people:
                    person_messages = messages_df[messages_df['person_id'] == person_id].sort_values('created_at')

                    # Find first static message (from Ed)
                    static_messages = person_messages[person_messages['direction'] == 'outgoing']
                    incoming_messages = person_messages[person_messages['direction'] == 'incoming']

                    if len(static_messages) > 0:
                        first_static = static_messages.iloc[0]['created_at']

                        # Check if they had any incoming messages
                        if len(incoming_messages) > 0:
                            first_response = incoming_messages.iloc[0]['created_at']
                            if first_response > first_static:
                                opt_out_analysis.append("After AI Response")
                            else:
                                opt_out_analysis.append("After 1st Static Message")
                        else:
                            opt_out_analysis.append("After 1st Static Message")

                if opt_out_analysis:
                    opt_out_timing = pd.Series(opt_out_analysis).value_counts()

                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.pie(
                            values=opt_out_timing.values,
                            names=opt_out_timing.index,
                            title="Opt-out Timing Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        st.subheader("Key Insight")
                        after_static = opt_out_timing.get("After 1st Static Message", 0)
                        after_ai = opt_out_timing.get("After AI Response", 0)
                        st.metric("Opted out after 1st static message", f"{after_static}/{len(opted_out_people)}")
                        if after_static > after_ai:
                            st.warning("⚠️ Most opt-outs happen before AI even responds")

        # Network multiplier detection
        st.header("🌐 Network Multiplier Detection")

        multiplier_keywords = ["invite", "invited", "bring", "friend", "friends", "family", "neighbor", "tell"]

        if 'body' in messages_df.columns:
            incoming_messages = messages_df[messages_df['direction'] == 'incoming']

            multiplier_messages = []
            for _, msg in incoming_messages.iterrows():
                if any(keyword in str(msg['body']).lower() for keyword in multiplier_keywords):
                    multiplier_messages.append(msg)

            if multiplier_messages:
                st.success(f"🎯 Found {len(multiplier_messages)} potential network multiplier responses!")

                # Show sample multiplier messages
                with st.expander("View Network Multiplier Messages"):
                    multiplier_df = pd.DataFrame(multiplier_messages)
                    for _, msg in multiplier_df.head(10).iterrows():
                        st.text(f"Person {msg['person_id']}: {msg['body']}")

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

        # Responder details with conversation viewer
        st.header("💬 Responder Details & Conversations")

        responder_ids = messages_df[messages_df['direction'] == 'incoming']['person_id'].unique()

        if len(responder_ids) > 0:
            selected_responder = st.selectbox(
                "Select a responder to view conversation",
                responder_ids,
                format_func=lambda x: f"Person {x}"
            )

            if selected_responder:
                # Get conversation for selected responder
                conversation = messages_df[
                    messages_df['person_id'] == selected_responder
                ].sort_values('created_at')

                st.subheader(f"Conversation with Person {selected_responder}")

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