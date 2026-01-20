# SMS Campaign Analytics Dashboard

An interactive Streamlit dashboard for analyzing SMS outreach campaigns with CSV data uploads.

## Features

- **Campaign Summary**: Key metrics including recipients, response rates, and opt-outs
- **Engagement Depth Analysis**: Breakdown of 1, 2+, and 3+ response patterns
- **Opt-out Timing Analysis**: When people opt out relative to AI vs static messages
- **Network Multiplier Detection**: Identifies people mentioning inviting others
- **Tag-Based Segment Comparison**: Compare performance across different audience segments
- **Conversation Viewer**: Browse individual responder conversations

## Key Insights

The dashboard analyzes opt-out timing to reveal when people disengage relative to static vs AI responses, highlighting the importance of initial message optimization.

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
   - Export your Messages and People data as CSV files from your SMS platform
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

This dashboard is designed for SMS campaigns with:
- **SMS Platform Integration**: Works with any platform that can export CSV data
- **AI Bot Analysis**: Analyzes static vs dynamic response patterns
- **CSV-Based Analysis**: Import data from any SMS platform export

## Usage Notes

- The dashboard automatically calculates engagement metrics and identifies patterns
- Network multiplier detection looks for keywords like "invite", "friend", "family"
- Conversation viewer shows the full back-and-forth for each responder
- Tag comparison helps identify which audience segments respond best

## Support

For questions about this dashboard, please open an issue in this repository or contact the development team.