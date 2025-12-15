import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="NFWP State Scorecard Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 NFWP State Scorecard Dashboard")
st.markdown("---")

# File options
DATA_FILES = {
    "Original Data (SANI.xlsx)": "Status NFWP SU STAFFING TO DO PLANS SANI.xlsx",
    "Updated Data (updated.xlsx)": "updated.xlsx"
}

# Load and clean data - Original file format
@st.cache_data
def load_original_data():
    file_path = 'Status NFWP SU STAFFING TO DO PLANS SANI.xlsx'

    # Read the scorecard sheet
    df = pd.read_excel(file_path, sheet_name='SCORECARD', skiprows=3)

    # Set proper column names
    df.columns = [
        'Empty1', 'S/N', 'State', 'SLA', 'Disbursement (Million)', 'Start Up Staff',
        'Peer Learning', 'Orientation and AWPB', 'AWPB', 'LGA', 'COM',
        'Full Staff - Others', 'LPIU Staffing', 'WF Staffing', 'WAG Formation',
        'Comment or Remark', 'Empty2', 'Empty3', 'Empty4'
    ]

    # Remove empty columns
    df = df.drop(columns=['Empty1', 'Empty2', 'Empty3', 'Empty4'])

    # Clean rows - keep only valid state data
    df = df[df['State'].notna() & (df['State'] != 'State')]
    df = df[df['S/N'].notna()]

    # Reset index
    df = df.reset_index(drop=True)

    # Categorize entries: State, FCT, or Federal (for original file)
    def categorize_entity(state_name):
        state_lower = str(state_name).lower()
        if state_lower in ['federal', 'federal coordinating unit', 'fcu']:
            return 'Federal'
        elif state_lower in ['fct', 'abuja', 'fc|abuja']:
            return 'FCT'
        else:
            return 'State'

    df['Entity Type'] = df['State'].apply(categorize_entity)

    # Convert Yes/No columns to standardized format
    yes_no_columns = ['SLA', 'Start Up Staff', 'Peer Learning', 'Orientation and AWPB',
                      'AWPB', 'COM', 'Full Staff - Others', 'LPIU Staffing', 'WF Staffing']

    for col in yes_no_columns:
        if col in df.columns:
            df[col] = df[col].fillna('No').astype(str).str.strip()
            # Standardize: anything that's not clearly "No" or empty is considered "Yes"
            df[col] = df[col].apply(lambda x: 'Yes' if x.lower() in ['yes', 'spc', 'po'] or (x != 'No' and x != '' and x != 'nan') else 'No')

    # Clean disbursement column
    df['Disbursement (Million)'] = pd.to_numeric(df['Disbursement (Million)'], errors='coerce').fillna(0)

    # Try to extract WAG Formation numbers
    df['WAG Formation'] = df['WAG Formation'].fillna('0').astype(str)
    df['WAG Count'] = df['WAG Formation'].apply(lambda x: pd.to_numeric(x, errors='coerce') if x.replace('.', '').isdigit() else 0)

    return df

# Load and clean data - Updated file format
@st.cache_data
def load_updated_data():
    file_path = 'updated.xlsx'

    # Read the sheet (data starts at row 1, header at row 0)
    df = pd.read_excel(file_path, sheet_name='Sheet1')

    # Rename columns to match expected format
    df.columns = [
        'S/N', 'State', 'SLA', 'Disbursement (Million)', 'Start Up Staff',
        'Full Staff - Others', 'Peer Learning', 'Orientation and AWPB', 'AWPB',
        'LGA', 'LPIU Staffing', 'COM', 'WF Staffing', 'WAG Formation', 'Comment or Remark'
    ]

    # Clean rows - keep only valid state data
    df = df[df['State'].notna()]
    df = df[df['S/N'].notna()]

    # Reset index
    df = df.reset_index(drop=True)

    # Categorize entries: State, FCT, or Federal
    def categorize_entity(state_name):
        state_clean = str(state_name).strip().lower()
        if state_clean in ['federal', 'federal coordinating unit', 'fcu']:
            return 'Federal'
        elif state_clean.startswith('fc|') or state_clean == 'fct' or state_clean == 'abuja':
            return 'FCT'
        else:
            return 'State'

    df['Entity Type'] = df['State'].apply(categorize_entity)

    # Convert Yes/No columns to standardized format
    yes_no_columns = ['SLA', 'Start Up Staff', 'Peer Learning', 'Orientation and AWPB',
                      'AWPB', 'COM', 'Full Staff - Others', 'LPIU Staffing', 'WF Staffing']

    for col in yes_no_columns:
        if col in df.columns:
            df[col] = df[col].fillna('No').astype(str).str.strip()
            # Standardize: anything that's not clearly "No" or empty is considered "Yes"
            df[col] = df[col].apply(lambda x: 'Yes' if x.lower() in ['yes', 'spc', 'po'] or (x != 'No' and x != '' and x != 'nan') else 'No')

    # Clean disbursement column
    df['Disbursement (Million)'] = pd.to_numeric(df['Disbursement (Million)'], errors='coerce').fillna(0)

    # Try to extract WAG Formation numbers
    df['WAG Formation'] = df['WAG Formation'].fillna('0').astype(str)
    df['WAG Count'] = df['WAG Formation'].apply(lambda x: pd.to_numeric(x, errors='coerce') if x.replace('.', '').isdigit() else 0)

    return df

# Load data based on selection
def load_data(file_choice):
    if file_choice == "Original Data (SANI.xlsx)":
        return load_original_data()
    else:
        return load_updated_data()

# Load data
try:
    # Sidebar - Data Source Selection
    st.sidebar.header("📁 Data Source")
    file_choice = st.sidebar.selectbox(
        "Select Data File",
        options=list(DATA_FILES.keys()),
        index=1  # Default to updated data
    )

    df = load_data(file_choice)

    # Sidebar filters
    st.sidebar.header("🔍 Filters")

    # Federal Coordinating Unit toggle
    include_federal = st.sidebar.checkbox(
        "Include Federal Coordinating Unit",
        value=False,
        help="Include the Federal Coordinating Unit in metrics and charts"
    )

    # Apply Federal filter first
    if include_federal:
        working_df = df.copy()
    else:
        working_df = df[df['Entity Type'] != 'Federal'].copy()

    # SLA filter
    sla_filter = st.sidebar.multiselect(
        "SLA Status",
        options=['Yes', 'No'],
        default=['Yes', 'No']
    )

    # State filter
    all_states = sorted(working_df['State'].unique())
    selected_states = st.sidebar.multiselect(
        "Select States",
        options=all_states,
        default=all_states
    )

    # Apply filters
    filtered_df = working_df[
        (working_df['SLA'].isin(sla_filter)) &
        (working_df['State'].isin(selected_states))
    ]

    # Key Metrics Row
    st.header("📈 Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        # Count by entity type
        state_count = (filtered_df['Entity Type'] == 'State').sum()
        fct_count = (filtered_df['Entity Type'] == 'FCT').sum()
        federal_count = (filtered_df['Entity Type'] == 'Federal').sum()

        # Build the label
        if include_federal and federal_count > 0:
            label = f"{state_count} States + FCT + Federal"
        else:
            label = f"{state_count} States + FCT"

        st.metric("Coverage", len(filtered_df), help=label)
        st.caption(label)

    with col2:
        sla_yes = (filtered_df['SLA'] == 'Yes').sum()
        st.metric("With SLA", sla_yes)

    with col3:
        total_disbursement = filtered_df['Disbursement (Million)'].sum()
        st.metric("Total Disbursement (M)", f"{total_disbursement:.1f}")

    with col4:
        wag_total = filtered_df['WAG Count'].sum()
        st.metric("Total WAG Formations", f"{int(wag_total)}")

    with col5:
        completion_rate = (filtered_df['AWPB'] == 'Yes').sum() / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
        st.metric("AWPB Completion", f"{completion_rate:.1f}%")

    st.markdown("---")

    # Charts Row 1
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Completion Status by Milestone")

        # Calculate completion counts
        milestones = ['Start Up Staff', 'Peer Learning', 'Orientation and AWPB',
                     'AWPB', 'COM', 'Full Staff - Others', 'LPIU Staffing', 'WF Staffing']

        completion_data = []
        for milestone in milestones:
            yes_count = (filtered_df[milestone] == 'Yes').sum()
            no_count = (filtered_df[milestone] == 'No').sum()
            completion_data.append({
                'Milestone': milestone,
                'Completed': yes_count,
                'Not Completed': no_count,
                'Completion %': (yes_count / (yes_count + no_count) * 100) if (yes_count + no_count) > 0 else 0
            })

        completion_df = pd.DataFrame(completion_data)

        fig = go.Figure(data=[
            go.Bar(name='Completed', x=completion_df['Milestone'], y=completion_df['Completed'], marker_color='#2ecc71'),
            go.Bar(name='Not Completed', x=completion_df['Milestone'], y=completion_df['Not Completed'], marker_color='#e74c3c')
        ])

        fig.update_layout(
            barmode='stack',
            xaxis_tickangle=-45,
            height=400,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("💰 Disbursement by State (Top 15)")

        # Get top 15 states by disbursement
        top_states = filtered_df.nlargest(15, 'Disbursement (Million)')[['State', 'Disbursement (Million)']]

        fig = px.bar(
            top_states,
            x='State',
            y='Disbursement (Million)',
            color='Disbursement (Million)',
            color_continuous_scale='Blues'
        )

        fig.update_layout(
            xaxis_tickangle=-45,
            height=400,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    # Charts Row 2
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 SLA Distribution")

        sla_counts = filtered_df['SLA'].value_counts()

        fig = px.pie(
            values=sla_counts.values,
            names=sla_counts.index,
            color=sla_counts.index,
            color_discrete_map={'Yes': '#2ecc71', 'No': '#e74c3c'},
            hole=0.4
        )

        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📋 Milestone Completion Rates")

        fig = go.Figure(data=[
            go.Bar(
                x=completion_df['Completion %'],
                y=completion_df['Milestone'],
                orientation='h',
                marker=dict(
                    color=completion_df['Completion %'],
                    colorscale='RdYlGn',
                    showscale=True
                ),
                text=completion_df['Completion %'].round(1).astype(str) + '%',
                textposition='auto',
            )
        ])

        fig.update_layout(
            height=350,
            xaxis_title="Completion Rate (%)",
            yaxis_title="Milestone",
            xaxis=dict(range=[0, 100])
        )

        st.plotly_chart(fig, use_container_width=True)

    # WAG Formation Analysis
    st.markdown("---")
    st.subheader("👥 WAG Formation by State")

    # Filter states with WAG data
    wag_df = filtered_df[filtered_df['WAG Count'] > 0].sort_values('WAG Count', ascending=False)

    if len(wag_df) > 0:
        fig = px.bar(
            wag_df,
            x='State',
            y='WAG Count',
            color='WAG Count',
            color_continuous_scale='Greens',
            text='WAG Count'
        )

        fig.update_layout(
            xaxis_tickangle=-45,
            height=400,
            showlegend=False
        )

        fig.update_traces(textposition='outside')

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No WAG Formation data available for selected states.")

    # Data Table
    st.markdown("---")
    st.subheader("📋 Detailed State Data")

    # Select columns to display
    display_columns = ['State', 'Entity Type', 'SLA', 'Disbursement (Million)', 'Start Up Staff',
                      'Peer Learning', 'AWPB', 'COM', 'LPIU Staffing', 'WF Staffing',
                      'WAG Formation', 'Comment or Remark']

    st.dataframe(
        filtered_df[display_columns].style.apply(
            lambda x: ['background-color: #d4edda' if v == 'Yes' else
                      'background-color: #f8d7da' if v == 'No' else ''
                      for v in x],
            subset=['SLA', 'Start Up Staff', 'Peer Learning', 'AWPB', 'COM', 'LPIU Staffing', 'WF Staffing']
        ),
        use_container_width=True,
        height=400
    )

    # Export option
    st.markdown("---")
    st.subheader("💾 Export Data")

    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name='scorecard_filtered_data.csv',
        mime='text/csv',
    )

except FileNotFoundError:
    st.error("⚠️ Excel file not found. Please ensure 'Status NFWP SU STAFFING TO DO PLANS SANI.xlsx' is in the same directory as this script.")
except Exception as e:
    st.error(f"⚠️ An error occurred: {str(e)}")
    st.exception(e)

# Footer
st.markdown("---")
st.markdown("*Dashboard created for NFWP State Scorecard tracking*")
