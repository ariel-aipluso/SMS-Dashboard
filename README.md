# AROS SMS Campaign Analytics Dashboard

An interactive Streamlit dashboard for analyzing SMS outreach campaigns from Daisychain, built for the Alliance to Reclaim Our Schools (AROS).

## Features

- **Campaign Summary**: Key metrics including recipients, response rates, and opt-outs
- **Engagement Depth Analysis**: Breakdown of 1, 2+, and 3+ response patterns
- **Opt-out Timing Analysis**: When people opt out relative to AI vs static messages
- **Network Multiplier Detection**: Identifies people mentioning inviting others
- **Tag-Based Segment Comparison**: Compare performance across different audience segments
- **Conversation Viewer**: Browse individual responder conversations

## Key Insights

The dashboard reveals that **16 of 17 opt-outs occurred after the 1st static message** (before the Ed AI bot even responded), highlighting the importance of the initial outreach message.

## Setup Instructions

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard**:
   ```bash
   streamlit run app.py
   ```

4. **Upload your data**:
   - Export your Messages and People data from Daisychain as CSV files
   - Upload both files using the sidebar file uploaders

## Data Requirements

### Messages CSV
Required columns:
- `person_id`: Identifier linking to people
- `body`: Message content
- `direction`: "incoming" or "outgoing"
- `created_at` or `sent_at`: Timestamp

### People CSV
Required columns:
- `id`: Person identifier
- `opted_out`: Boolean indicating opt-out status

Optional columns:
- `tags`: Comma-separated tags for segmentation

## Context

This dashboard is designed for SMS campaigns using:
- **Daisychain**: SMS platform for outreach
- **Ed AI Bot**: Sends initial static message, then responds dynamically
- **CSV Exports**: Since Daisychain API doesn't expose message/conversation endpoints

## Usage Notes

- The dashboard automatically calculates engagement metrics and identifies patterns
- Network multiplier detection looks for keywords like "invite", "friend", "family"
- Conversation viewer shows the full back-and-forth for each responder
- Tag comparison helps identify which audience segments respond best

## Support

For questions about this dashboard or AROS campaigns, please contact the AROS technical team.