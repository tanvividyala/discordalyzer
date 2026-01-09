import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Discord Conversation Analyzer",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Discord Conversation Analyzer")
st.markdown("Analyze your Discord conversation history with interactive visualizations")

# File upload section
st.sidebar.header("📁 Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload your Discord conversation CSV", type=['csv'])

if uploaded_file is not None:
    # Read the CSV
    data = pd.read_csv(uploaded_file)
    
    st.sidebar.success("✅ File uploaded successfully!")
    
    # Column selection
    st.sidebar.header("🔧 Configure Columns")
    st.sidebar.markdown("Select which columns correspond to each data type:")
    
    columns = data.columns.tolist()
    
    date_col = st.sidebar.selectbox("Date Column", columns, index=0)
    author_col = st.sidebar.selectbox("Author Column", columns, index=1 if len(columns) > 1 else 0)
    content_col = st.sidebar.selectbox("Content/Message Column", columns, index=2 if len(columns) > 2 else 0)
    
    # Get unique authors
    unique_authors = data[author_col].unique()
    
    if len(unique_authors) >= 2:
        user1 = st.sidebar.selectbox("User 1", unique_authors, index=0)
        user2 = st.sidebar.selectbox("User 2", unique_authors, index=1)
    else:
        st.error("⚠️ Need at least 2 unique authors in the conversation!")
        st.stop()
    
    # Process the data
    try:
        data[date_col] = pd.to_datetime(data[date_col], utc=True)
    except:
        st.error("⚠️ Could not parse the date column. Please ensure it contains valid dates.")
        st.stop()
    
    # Display basic stats
    st.header("📊 Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Messages", len(data))
    with col2:
        st.metric(f"{user1}'s Messages", len(data[data[author_col] == user1]))
    with col3:
        st.metric(f"{user2}'s Messages", len(data[data[author_col] == user2]))
    with col4:
        date_range = (data[date_col].max() - data[date_col].min()).days
        st.metric("Days of Conversation", date_range)
    
    st.markdown("---")
    
    # Section 1: Messages Over Time
    st.header("📈 Messages Sent Over Time")
    
    # Frequency selector
    freq_options = {
        "Daily": "D",
        "Weekly": "W",
        "Monthly": "M",
        "Yearly": "Y"
    }
    frequency = st.selectbox("Select time frequency", list(freq_options.keys()), index=2)
    
    # Calculate monthly totals
    monthly_totals = data.groupby(pd.Grouper(key=date_col, freq=freq_options[frequency])).size()
    
    # Create Plotly line chart
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=monthly_totals.index,
        y=monthly_totals.values,
        mode='lines+markers',
        name='Messages',
        line=dict(color='#7289DA', width=3),
        marker=dict(size=6)
    ))
    
    fig1.update_layout(
        title=f'Messages Sent Over Time ({frequency})',
        xaxis_title='Date',
        yaxis_title='Number of Messages',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    st.markdown("---")
    
    # Section 2: Proportion of messages by each person
    st.header("🥧 Message Distribution Over Time")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### Controls")
        prop_freq_options = {
            "Daily": "D",
            "Weekly": "W",
            "Monthly": "M"
        }
        prop_frequency = st.selectbox("Time frequency for proportion", list(prop_freq_options.keys()), index=1)
        
        # Date range slider
        min_date = data[date_col].min().date()
        max_date = data[date_col].max().date()
        
        st.markdown("### Date Range")
        date_range = st.slider(
            "Select date range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD"
        )
    
    with col2:
        # Filter data by date range
        filtered_data = data[
            (data[date_col].dt.date >= date_range[0]) & 
            (data[date_col].dt.date <= date_range[1])
        ]
        
        # Calculate proportions
        period_totals = filtered_data.groupby(pd.Grouper(key=date_col, freq=prop_freq_options[prop_frequency])).size()
        sent_by_user1 = filtered_data[filtered_data[author_col] == user1].groupby(pd.Grouper(key=date_col, freq=prop_freq_options[prop_frequency])).size()
        sent_by_user2 = filtered_data[filtered_data[author_col] == user2].groupby(pd.Grouper(key=date_col, freq=prop_freq_options[prop_frequency])).size()
        
        # Replace zeros with NaN to avoid division issues
        period_totals = period_totals.replace(0, float('nan'))
        
        user1_proportion = sent_by_user1 / period_totals
        user2_proportion = sent_by_user2 / period_totals
        
        # Calculate overall proportions for the selected date range
        total_user1 = len(filtered_data[filtered_data[author_col] == user1])
        total_user2 = len(filtered_data[filtered_data[author_col] == user2])
        
        # Create pie chart for overall proportion
        fig2 = go.Figure(data=[go.Pie(
            labels=[user1, user2],
            values=[total_user1, total_user2],
            hole=0.3,
            marker=dict(colors=['#7289DA', '#43B581']),
            textinfo='label+percent',
            textfont_size=14
        )])
        
        fig2.update_layout(
            title=f'Overall Message Distribution ({date_range[0]} to {date_range[1]})',
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Create stacked area chart for proportions over time
        fig3 = go.Figure()
        
        fig3.add_trace(go.Scatter(
            x=user1_proportion.index,
            y=user1_proportion.values * 100,
            mode='lines',
            name=user1,
            line=dict(width=0.5, color='#7289DA'),
            stackgroup='one',
            fillcolor='rgba(114, 137, 218, 0.7)'
        ))
        
        fig3.add_trace(go.Scatter(
            x=user2_proportion.index,
            y=user2_proportion.values * 100,
            mode='lines',
            name=user2,
            line=dict(width=0.5, color='#43B581'),
            stackgroup='one',
            fillcolor='rgba(67, 181, 129, 0.7)'
        ))
        
        fig3.update_layout(
            title=f'Message Proportion Over Time ({prop_frequency})',
            xaxis_title='Date',
            yaxis_title='Percentage of Messages (%)',
            hovermode='x unified',
            template='plotly_white',
            height=400,
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig3, use_container_width=True)
    
    # Show sample data
    with st.expander("📋 View Sample Data"):
        st.dataframe(data.head(20))

else:
    # Landing page when no file is uploaded
    st.info("👈 Please upload a CSV file to get started!")
    
    st.markdown("""
    ### How to use this app:
    
    1. **Upload your CSV file** using the sidebar on the left
    2. **Configure columns** by selecting which columns contain dates, authors, and messages
    3. **Select the two users** you want to analyze
    4. **Explore the visualizations** and use the interactive controls
    
    ### Expected CSV format:
    
    Your CSV should contain at least these columns:
    - A date/timestamp column
    - An author/username column
    - A message content column
    
    ### Features:
    
    - 📈 **Messages over time**: View conversation activity with adjustable time frequencies
    - 🥧 **Message distribution**: See who sent more messages with pie charts and proportion graphs
    - 🎚️ **Interactive controls**: Slider to filter date ranges and dropdowns to adjust visualizations
    
    ### Example CSV structure:
    ```
    Date,Author,Content
    2020-01-01 10:30:00,user1,Hello!
    2020-01-01 10:31:00,user2,Hi there!
    ```
    """)