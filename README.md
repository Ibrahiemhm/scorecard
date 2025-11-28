# NFWP State Scorecard Dashboard

A simple, interactive dashboard for visualizing the NFWP State Scorecard data.

## Features

- **Key Metrics Overview**: Total states, SLA status, disbursements, WAG formations, and completion rates
- **Interactive Visualizations**:
  - Completion status by milestone (stacked bar chart)
  - Disbursement by state (bar chart)
  - SLA distribution (pie chart)
  - Milestone completion rates (horizontal bar chart)
  - WAG formation by state
- **Filters**: Filter by SLA status and select specific states
- **Detailed Data Table**: View all state data with color-coded status
- **Export Functionality**: Download filtered data as CSV

## Installation

1. Make sure you have Python installed (Python 3.8 or higher recommended)

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

   Or install individually:
   ```bash
   pip install streamlit pandas plotly openpyxl
   ```

## How to Run

1. Make sure the Excel file `Status NFWP SU STAFFING TO DO PLANS SANI.xlsx` is in the same folder as `dashboard.py`

2. Open your terminal/command prompt and navigate to the folder:
   ```bash
   cd /Users/ibrahim/Desktop/yetunde
   ```

3. Run the dashboard:
   ```bash
   streamlit run dashboard.py
   ```

4. The dashboard will automatically open in your default web browser at `http://localhost:8501`

## Using the Dashboard

### Sidebar Filters
- **SLA Status**: Filter states by whether they have SLA or not
- **Select States**: Choose specific states to view (or select all)

### Key Metrics
- View quick statistics at the top of the dashboard

### Charts
- Hover over charts for detailed information
- Charts update automatically when you change filters

### Data Table
- Scroll through the detailed state data
- Green cells indicate "Yes", red cells indicate "No"

### Export Data
- Click the "Download Filtered Data as CSV" button at the bottom to export the current filtered view

## Troubleshooting

**Dashboard won't start:**
- Make sure all packages are installed: `pip install -r requirements.txt`
- Check that Python is properly installed: `python --version`

**Data not showing:**
- Ensure the Excel file is in the same directory as dashboard.py
- Check that the file name matches exactly: `Status NFWP SU STAFFING TO DO PLANS SANI.xlsx`

**Port already in use:**
- If you see an error about port 8501 being in use, run:
  ```bash
  streamlit run dashboard.py --server.port 8502
  ```

## Updating the Data

When the Excel file is updated:
1. Save the Excel file
2. Refresh the dashboard in your browser (click the "Rerun" button that appears, or press 'R')
3. Streamlit will automatically reload the new data

## Technical Details

- **Framework**: Streamlit
- **Visualization**: Plotly
- **Data Processing**: Pandas
- **Excel Reading**: OpenPyXL

---
*Created for NFWP State Scorecard tracking and reporting*
