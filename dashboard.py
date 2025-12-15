import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(
    page_title="NFWP State Scorecard Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 NFWP State Scorecard Dashboard")
st.markdown("---")


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

# Define milestones for scoring
MILESTONES = ['SLA', 'Start Up Staff', 'Peer Learning', 'Orientation and AWPB',
              'AWPB', 'COM', 'Full Staff - Others', 'LPIU Staffing', 'WF Staffing']

# Calculate readiness score for each state
def calculate_readiness_score(df):
    """Calculate a readiness score (0-100) based on milestone completion"""
    scores = []
    for _, row in df.iterrows():
        completed = sum(1 for m in MILESTONES if row.get(m) == 'Yes')
        score = (completed / len(MILESTONES)) * 100
        scores.append(score)
    df['Readiness Score'] = scores
    df['Milestones Completed'] = df[MILESTONES].apply(lambda x: (x == 'Yes').sum(), axis=1)
    return df

# Load data based on selection
def load_data(file_choice):
    if file_choice == "Original Data (SANI.xlsx)":
        df = load_original_data()
    else:
        df = load_updated_data()
    return calculate_readiness_score(df)

# Load data
try:
    df = load_data("Updated Data (updated.xlsx)")

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

    # Additional Summary Statistics (Feature 10)
    st.markdown("###")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_readiness = filtered_df['Readiness Score'].mean() if len(filtered_df) > 0 else 0
        st.metric("Avg. Readiness Score", f"{avg_readiness:.1f}%")

    with col2:
        fully_staffed = ((filtered_df['Start Up Staff'] == 'Yes') &
                         (filtered_df['Full Staff - Others'] == 'Yes') &
                         (filtered_df['LPIU Staffing'] == 'Yes')).sum()
        st.metric("Fully Staffed", fully_staffed)

    with col3:
        avg_milestones = filtered_df['Milestones Completed'].mean() if len(filtered_df) > 0 else 0
        st.metric("Avg. Milestones", f"{avg_milestones:.1f}/9")

    with col4:
        states_with_wag = (filtered_df['WAG Count'] > 0).sum()
        st.metric("States with WAG", states_with_wag)

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

    # ===========================================
    # NEW FEATURE 1: State Readiness Score Chart
    # ===========================================
    st.markdown("---")
    st.subheader("🎯 State Readiness Score")
    st.caption("Readiness score based on milestone completion (0-100%)")

    # Sort by readiness score
    readiness_df = filtered_df[['State', 'Readiness Score', 'Milestones Completed']].sort_values(
        'Readiness Score', ascending=True
    )

    fig = go.Figure(data=[
        go.Bar(
            x=readiness_df['Readiness Score'],
            y=readiness_df['State'],
            orientation='h',
            marker=dict(
                color=readiness_df['Readiness Score'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Score %")
            ),
            text=readiness_df['Readiness Score'].round(1).astype(str) + '%',
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Score: %{x:.1f}%<br>Milestones: %{customdata}/9<extra></extra>',
            customdata=readiness_df['Milestones Completed']
        )
    ])

    fig.update_layout(
        height=max(400, len(readiness_df) * 25),
        xaxis_title="Readiness Score (%)",
        xaxis=dict(range=[0, 110]),
        yaxis_title="",
        margin=dict(l=150)
    )

    st.plotly_chart(fig, use_container_width=True)

    # ===========================================
    # NEW FEATURE 2: Progress Funnel Chart
    # ===========================================
    st.markdown("---")
    st.subheader("📉 Implementation Progress Funnel")
    st.caption("Shows how states progress through each milestone stage")

    # Calculate funnel data - ordered by typical implementation sequence
    funnel_milestones = ['SLA', 'Start Up Staff', 'Peer Learning', 'Orientation and AWPB',
                         'AWPB', 'LPIU Staffing', 'COM', 'Full Staff - Others', 'WF Staffing']

    funnel_data = []
    for milestone in funnel_milestones:
        count = (filtered_df[milestone] == 'Yes').sum()
        funnel_data.append({'Stage': milestone, 'Count': count})

    funnel_df = pd.DataFrame(funnel_data)

    fig = go.Figure(go.Funnel(
        y=funnel_df['Stage'],
        x=funnel_df['Count'],
        textposition="inside",
        textinfo="value+percent initial",
        marker=dict(color=['#3498db', '#2980b9', '#1abc9c', '#16a085',
                          '#27ae60', '#2ecc71', '#f39c12', '#e67e22', '#e74c3c']),
        connector=dict(line=dict(color="royalblue", dash="dot", width=3))
    ))

    fig.update_layout(
        height=500,
        title_text="States Completing Each Milestone"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ===========================================
    # NEW FEATURE 4: State Comparison Radar Tool
    # ===========================================
    st.markdown("---")
    st.subheader("🕸️ State Comparison Tool")
    st.caption("Select states to compare their milestone completion")

    # State selector for comparison
    comparison_states = st.multiselect(
        "Select states to compare (2-5 recommended)",
        options=sorted(filtered_df['State'].unique()),
        default=sorted(filtered_df['State'].unique())[:3] if len(filtered_df) >= 3 else list(filtered_df['State'].unique()),
        max_selections=5
    )

    if len(comparison_states) >= 2:
        radar_milestones = ['SLA', 'Start Up Staff', 'Peer Learning', 'AWPB',
                           'COM', 'LPIU Staffing', 'WF Staffing']

        fig = go.Figure()

        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

        for i, state in enumerate(comparison_states):
            state_data = filtered_df[filtered_df['State'] == state].iloc[0]
            values = [1 if state_data[m] == 'Yes' else 0 for m in radar_milestones]
            values.append(values[0])  # Close the radar

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=radar_milestones + [radar_milestones[0]],
                fill='toself',
                name=state,
                line_color=colors[i % len(colors)],
                opacity=0.7
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], tickvals=[0, 1], ticktext=['No', 'Yes'])
            ),
            showlegend=True,
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Please select at least 2 states to compare.")

    # ===========================================
    # NEW FEATURE 3: Staffing Breakdown Chart
    # ===========================================
    st.markdown("---")
    st.subheader("👔 Staffing Breakdown")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Staffing Status Overview**")

        staffing_categories = ['Start Up Staff', 'Full Staff - Others', 'LPIU Staffing', 'WF Staffing']
        staffing_data = []

        for cat in staffing_categories:
            yes_count = (filtered_df[cat] == 'Yes').sum()
            no_count = (filtered_df[cat] == 'No').sum()
            staffing_data.append({
                'Category': cat,
                'Filled': yes_count,
                'Vacant': no_count
            })

        staffing_df = pd.DataFrame(staffing_data)

        fig = go.Figure(data=[
            go.Bar(name='Filled', x=staffing_df['Category'], y=staffing_df['Filled'], marker_color='#27ae60'),
            go.Bar(name='Vacant', x=staffing_df['Category'], y=staffing_df['Vacant'], marker_color='#e74c3c')
        ])

        fig.update_layout(
            barmode='group',
            height=350,
            xaxis_tickangle=-30,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Staffing Completion Rate**")

        staffing_rates = []
        for cat in staffing_categories:
            rate = (filtered_df[cat] == 'Yes').sum() / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
            staffing_rates.append({'Category': cat, 'Rate': rate})

        staffing_rate_df = pd.DataFrame(staffing_rates)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=staffing_rate_df['Rate'].mean(),
            title={'text': "Average Staffing Rate"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#3498db"},
                'steps': [
                    {'range': [0, 33], 'color': "#fadbd8"},
                    {'range': [33, 66], 'color': "#fcf3cf"},
                    {'range': [66, 100], 'color': "#d5f5e3"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))

        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # ===========================================
    # NEW FEATURE 6: Disbursement vs Progress Correlation
    # ===========================================
    st.markdown("---")
    st.subheader("💰 Disbursement vs Progress Correlation")
    st.caption("Analyze relationship between disbursement and milestone completion")

    fig = px.scatter(
        filtered_df,
        x='Disbursement (Million)',
        y='Readiness Score',
        size='Milestones Completed',
        color='Entity Type',
        hover_name='State',
        hover_data=['Milestones Completed', 'SLA'],
        color_discrete_map={'State': '#3498db', 'FCT': '#e74c3c', 'Federal': '#f39c12'},
        size_max=20
    )

    # Add trend line
    if len(filtered_df) > 2:
        z = np.polyfit(filtered_df['Disbursement (Million)'], filtered_df['Readiness Score'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(filtered_df['Disbursement (Million)'].min(),
                            filtered_df['Disbursement (Million)'].max(), 100)
        fig.add_trace(go.Scatter(x=x_line, y=p(x_line), mode='lines',
                                name='Trend', line=dict(dash='dash', color='gray')))

    fig.update_layout(
        height=450,
        xaxis_title="Disbursement (Million)",
        yaxis_title="Readiness Score (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # ===========================================
    # NEW FEATURE 9: WAG Formation Leaders
    # ===========================================
    st.markdown("---")
    st.subheader("🏆 WAG Formation Leaders")

    # Filter states with WAG data
    wag_df = filtered_df[filtered_df['WAG Count'] > 0].sort_values('WAG Count', ascending=False)

    if len(wag_df) > 0:
        col1, col2 = st.columns([2, 1])

        with col1:
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

        with col2:
            st.markdown("**Top WAG Performers**")
            for i, (_, row) in enumerate(wag_df.head(5).iterrows(), 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                st.markdown(f"{medal} **{row['State']}**: {int(row['WAG Count'])} WAGs")

            st.markdown("---")
            st.metric("Total WAG Formations", f"{int(wag_df['WAG Count'].sum())}")
            st.metric("States with WAGs", len(wag_df))
    else:
        st.info("No WAG Formation data available for selected states.")

    # ===========================================
    # NEW FEATURE 7: Gap Analysis Table
    # ===========================================
    st.markdown("---")
    st.subheader("🔍 Gap Analysis")
    st.caption("Identify missing milestones for each state")

    # Create gap analysis data
    gap_data = []
    for _, row in filtered_df.iterrows():
        missing = [m for m in MILESTONES if row[m] == 'No']
        completed = [m for m in MILESTONES if row[m] == 'Yes']
        gap_data.append({
            'State': row['State'],
            'Completed': len(completed),
            'Missing': len(missing),
            'Missing Milestones': ', '.join(missing) if missing else 'All Complete ✓',
            'Readiness': row['Readiness Score']
        })

    gap_df = pd.DataFrame(gap_data).sort_values('Missing', ascending=False)

    # Color code based on missing count
    def highlight_gaps(row):
        if row['Missing'] == 0:
            return ['background-color: #d5f5e3'] * len(row)
        elif row['Missing'] <= 3:
            return ['background-color: #fcf3cf'] * len(row)
        else:
            return ['background-color: #fadbd8'] * len(row)

    st.dataframe(
        gap_df.style.apply(highlight_gaps, axis=1),
        use_container_width=True,
        height=400
    )

    # Gap summary
    col1, col2, col3 = st.columns(3)
    with col1:
        all_complete = (gap_df['Missing'] == 0).sum()
        st.metric("Fully Complete", all_complete, help="States with all milestones completed")
    with col2:
        needs_attention = (gap_df['Missing'] > 5).sum()
        st.metric("Needs Attention", needs_attention, help="States missing more than 5 milestones")
    with col3:
        most_common_gap = pd.Series([m for gaps in gap_df['Missing Milestones']
                                     for m in gaps.split(', ') if m != 'All Complete ✓']).value_counts()
        if len(most_common_gap) > 0:
            st.metric("Most Common Gap", most_common_gap.index[0])

    # ===========================================
    # NEW FEATURE 8: Activity/Comments Feed
    # ===========================================
    st.markdown("---")
    st.subheader("📝 Recent Activities & Comments")
    st.caption("Latest updates and remarks from states")

    # Filter states with comments
    comments_df = filtered_df[filtered_df['Comment or Remark'].notna() &
                              (filtered_df['Comment or Remark'].astype(str).str.strip() != '')]

    if len(comments_df) > 0:
        for _, row in comments_df.iterrows():
            comment = str(row['Comment or Remark']).strip()
            if comment and comment.lower() != 'nan':
                with st.expander(f"📍 {row['State']} - Readiness: {row['Readiness Score']:.0f}%"):
                    st.write(comment)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.caption(f"SLA: {'✅' if row['SLA'] == 'Yes' else '❌'}")
                    with col2:
                        st.caption(f"AWPB: {'✅' if row['AWPB'] == 'Yes' else '❌'}")
                    with col3:
                        st.caption(f"Disbursement: ₦{row['Disbursement (Million)']}M")
    else:
        st.info("No comments or activities recorded for selected states.")

    # Data Table
    st.markdown("---")
    st.subheader("📋 Detailed State Data")

    # Select columns to display
    display_columns = ['State', 'Entity Type', 'Readiness Score', 'SLA', 'Disbursement (Million)', 'Start Up Staff',
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
