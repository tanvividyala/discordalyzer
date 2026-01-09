import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="Conversation Analyzer",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Conversation Analyzer")
st.markdown("A tool to analyze your conversation history and visualize trends over time. Works with any CSV or JSON export from messaging platforms!")

# Sidebar setup
st.sidebar.header("📁 Upload Your Data")

# OpenAI API Key input at the top
st.sidebar.markdown("### 🤖 OpenAI API Key (Optional)")
api_key_input = st.sidebar.text_input(
    "Enter your API key to unlock conversation summaries",
    type="password",
    help="Get one at https://platform.openai.com/api-keys",
    placeholder="sk-..."
)

if api_key_input:
    st.sidebar.success("✅ API key provided - summaries unlocked!")
else:
    st.sidebar.info("💡 Add an API key to enable AI-powered daily summaries")

st.sidebar.markdown("---")

# File upload section
uploaded_file = st.sidebar.file_uploader("Upload your conversation file", type=['csv', 'json'])

if uploaded_file is not None:
    # Read the file based on type
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_type == 'json':
            # Read JSON and convert to DataFrame
            json_data = json.load(uploaded_file)
            
            # Handle different JSON structures
            if isinstance(json_data, list):
                data = pd.DataFrame(json_data)
            elif isinstance(json_data, dict):
                # Try to find the messages array in the JSON
                if 'messages' in json_data:
                    data = pd.DataFrame(json_data['messages'])
                else:
                    data = pd.DataFrame([json_data])
            else:
                st.error("⚠️ Couldn't parse this JSON format. Try converting it to CSV first.")
                st.stop()
            
            st.sidebar.success("✅ JSON file loaded and converted!")
        else:
            # Read CSV
            data = pd.read_csv(uploaded_file)
            st.sidebar.success("✅ CSV file loaded!")
    except Exception as e:
        st.error(f"⚠️ Error reading file: {str(e)}")
        st.stop()
    
    # Column selection
    st.sidebar.header("🔧 Configure Columns")
    st.sidebar.markdown("Tell me which columns have what:")
    
    columns = data.columns.tolist()
    
    date_col = st.sidebar.selectbox("Date/Timestamp Column", columns, index=0)
    author_col = st.sidebar.selectbox("Author/Username Column", columns, index=1 if len(columns) > 1 else 0)
    content_col = st.sidebar.selectbox("Message Content Column", columns, index=2 if len(columns) > 2 else 0)
    
    # Get all unique authors
    unique_authors = sorted(data[author_col].unique())
    num_authors = len(unique_authors)
    
    if num_authors < 2:
        st.error("⚠️ Need at least 2 people in the conversation to analyze!")
        st.stop()
    
    # Let user select which authors to analyze
    st.sidebar.markdown(f"### 👥 Participants ({num_authors} found)")
    
    if num_authors == 2:
        # For 2 people, just assign them
        selected_authors = unique_authors
        st.sidebar.info(f"Analyzing: {selected_authors[0]} & {selected_authors[1]}")
    else:
        # For more than 2, let them choose
        selected_authors = st.sidebar.multiselect(
            "Select people to analyze (pick 2+)",
            unique_authors,
            default=unique_authors[:2] if num_authors >= 2 else unique_authors
        )
        
        if len(selected_authors) < 2:
            st.warning("⚠️ Please select at least 2 people to compare")
            st.stop()
    
    # Filter data to only selected authors
    data = data[data[author_col].isin(selected_authors)]
    
    # Process the data
    try:
        data[date_col] = pd.to_datetime(data[date_col], utc=True)
    except:
        st.error("⚠️ Couldn't parse the dates in your file. Make sure the date column has valid timestamps.")
        st.stop()
    
    # Display basic stats
    st.header("📊 Quick Stats")
    
    # Create dynamic columns based on number of selected authors
    cols = st.columns(len(selected_authors) + 2)
    
    with cols[0]:
        st.metric("Total Messages", len(data))
    
    for idx, author in enumerate(selected_authors):
        with cols[idx + 1]:
            author_count = len(data[data[author_col] == author])
            st.metric(f"{author}'s Messages", author_count)
    
    with cols[-1]:
        date_range = (data[date_col].max() - data[date_col].min()).days
        st.metric("Days of Conversation", date_range)
    
    st.markdown("---")
    
    # Section 1: Messages Over Time
    st.header("📈 Messages Over Time")
    
    # Frequency selector
    freq_options = {
        "Daily": "D",
        "Weekly": "W",
        "Monthly": "M",
        "Yearly": "Y"
    }
    frequency = st.selectbox("Pick a time frequency", list(freq_options.keys()), index=2)
    
    # Calculate totals
    time_totals = data.groupby(pd.Grouper(key=date_col, freq=freq_options[frequency])).size()
    
    # Create Plotly line chart
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=time_totals.index,
        y=time_totals.values,
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
    
    # Section 2: Message Distribution
    st.header("🥧 Who's Talking More?")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### Controls")
        prop_freq_options = {
            "Daily": "D",
            "Weekly": "W",
            "Monthly": "M"
        }
        prop_frequency = st.selectbox("Time frequency", list(prop_freq_options.keys()), index=1, key="prop_freq")
        
        # Date range slider
        min_date = data[date_col].min().date()
        max_date = data[date_col].max().date()
        
        st.markdown("### Date Range")
        date_range = st.slider(
            "Filter by date",
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
        
        # Color palette for multiple users
        color_palette = ['#7289DA', '#43B581', '#FAA61A', '#F47FFF', '#00B0F4', '#FF4444']
        
        # Calculate overall proportions
        author_counts = [len(filtered_data[filtered_data[author_col] == author]) for author in selected_authors]
        
        # Create pie chart
        fig2 = go.Figure(data=[go.Pie(
            labels=selected_authors,
            values=author_counts,
            hole=0.3,
            marker=dict(colors=color_palette[:len(selected_authors)]),
            textinfo='label+percent',
            textfont_size=14
        )])
        
        fig2.update_layout(
            title=f'Message Distribution ({date_range[0]} to {date_range[1]})',
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Create stacked area chart for proportions over time
        fig3 = go.Figure()
        
        period_totals = period_totals.replace(0, float('nan'))
        
        for idx, author in enumerate(selected_authors):
            author_counts_time = filtered_data[filtered_data[author_col] == author].groupby(
                pd.Grouper(key=date_col, freq=prop_freq_options[prop_frequency])
            ).size()
            author_proportion = author_counts_time / period_totals
            
            fig3.add_trace(go.Scatter(
                x=author_proportion.index,
                y=author_proportion.values * 100,
                mode='lines',
                name=author,
                line=dict(width=0.5, color=color_palette[idx % len(color_palette)]),
                stackgroup='one',
                fillcolor=f'rgba{tuple(list(int(color_palette[idx % len(color_palette)].lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.7])}'.replace("'", "")
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
    
    st.markdown("---")
    
    # Section 3: Sentiment Analysis
    st.header("😊 Conversation Vibes")
    st.markdown("See how positive or negative your conversations have been over time")
    
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        vader_available = True
    except ImportError:
        vader_available = False
        st.warning("⚠️ vaderSentiment not installed. Run `pip install vaderSentiment` to unlock sentiment analysis.")
    
    if vader_available:
        # Initialize VADER
        analyzer = SentimentIntensityAnalyzer()
        
        # Calculate sentiment for all messages
        with st.spinner("🔍 Analyzing sentiment... This might take a moment for large conversations."):
            def get_sentiment(text):
                if pd.isna(text):
                    return 0
                scores = analyzer.polarity_scores(str(text))
                return scores['compound']  # Returns score from -1 (negative) to 1 (positive)
            
            # Only calculate if not already done
            if 'sentiment' not in data.columns:
                data['sentiment'] = data[content_col].apply(get_sentiment)
        
        tab1, tab2 = st.tabs(["📅 Sentiment Calendar", "📈 Sentiment Trends"])
        
        with tab1:
            st.markdown("### Your conversation vibes, visualized as a calendar")
            st.markdown("Green = positive vibes, Red = negative vibes, Yellow = neutral")
            
            # Calculate daily average sentiment
            daily_sentiment = data.groupby(data[date_col].dt.date)['sentiment'].mean()
            
            # Create a DataFrame for the heatmap
            sentiment_df = pd.DataFrame({
                'date': daily_sentiment.index,
                'sentiment': daily_sentiment.values
            })
            
            # Add year, month, day columns for better organization
            sentiment_df['year'] = pd.to_datetime(sentiment_df['date']).dt.year
            sentiment_df['month'] = pd.to_datetime(sentiment_df['date']).dt.month
            sentiment_df['day'] = pd.to_datetime(sentiment_df['date']).dt.day
            sentiment_df['weekday'] = pd.to_datetime(sentiment_df['date']).dt.dayofweek
            sentiment_df['week'] = pd.to_datetime(sentiment_df['date']).dt.isocalendar().week
            
            # Year selector
            available_years = sorted(sentiment_df['year'].unique(), reverse=True)
            selected_year = st.selectbox("Select year", available_years, key="sentiment_year")
            
            # Filter to selected year
            year_data = sentiment_df[sentiment_df['year'] == selected_year].copy()
            
            if len(year_data) > 0:
                # Create calendar-style heatmap with actual dates
                # Organize by week and weekday
                calendar_data = year_data.pivot_table(
                    values='sentiment',
                    index='weekday',
                    columns='week',
                    aggfunc='mean'
                )
                
                # Create a hover text matrix with actual dates
                hover_text = []
                for weekday in range(7):
                    hover_row = []
                    for week in calendar_data.columns:
                        # Find the actual date for this week/weekday combination
                        matching_dates = year_data[(year_data['week'] == week) & (year_data['weekday'] == weekday)]
                        if len(matching_dates) > 0:
                            date_str = matching_dates.iloc[0]['date'].strftime('%B %d, %Y')
                            sentiment_val = matching_dates.iloc[0]['sentiment']
                            hover_row.append(f"{date_str}<br>Sentiment: {sentiment_val:.2f}")
                        else:
                            hover_row.append("")
                    hover_text.append(hover_row)
                
                # Create custom colorscale (red -> yellow -> green)
                fig_calendar = go.Figure(data=go.Heatmap(
                    z=calendar_data.values,
                    x=calendar_data.columns,
                    y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                    text=hover_text,
                    hovertemplate='%{text}<extra></extra>',
                    colorscale=[
                        [0, '#FF4444'],      # Negative - Red
                        [0.5, '#FFD700'],    # Neutral - Gold
                        [1, '#43B581']       # Positive - Green
                    ],
                    zmid=0,
                    colorbar=dict(
                        title="Sentiment",
                        tickvals=[-1, -0.5, 0, 0.5, 1],
                        ticktext=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
                    )
                ))
                
                fig_calendar.update_layout(
                    title=f'Sentiment Calendar - {selected_year}',
                    xaxis_title='Week of Year',
                    yaxis_title='Day of Week',
                    height=400,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_calendar, use_container_width=True)
                
                # Show some stats
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_sentiment = year_data['sentiment'].mean()
                    st.metric("Average Sentiment", f"{avg_sentiment:.2f}")
                
                with col2:
                    positive_days = len(year_data[year_data['sentiment'] > 0.05])
                    st.metric("Positive Days", positive_days)
                
                with col3:
                    negative_days = len(year_data[year_data['sentiment'] < -0.05])
                    st.metric("Negative Days", negative_days)
                
                with col4:
                    best_day = year_data.loc[year_data['sentiment'].idxmax(), 'date']
                    st.metric("Best Day", best_day.strftime('%b %d'))
                
                # Show most positive and negative days
                with st.expander("🔍 See specific days"):
                    col_pos, col_neg = st.columns(2)
                    
                    with col_pos:
                        st.markdown("**Most Positive Days:**")
                        top_positive = year_data.nlargest(5, 'sentiment')[['date', 'sentiment']]
                        for _, row in top_positive.iterrows():
                            st.write(f"• {row['date'].strftime('%B %d')}: {row['sentiment']:.2f}")
                    
                    with col_neg:
                        st.markdown("**Most Negative Days:**")
                        top_negative = year_data.nsmallest(5, 'sentiment')[['date', 'sentiment']]
                        for _, row in top_negative.iterrows():
                            st.write(f"• {row['date'].strftime('%B %d')}: {row['sentiment']:.2f}")
            else:
                st.info(f"No data available for {selected_year}")
        
        with tab2:
            st.markdown("### Track sentiment over time")
            st.markdown("See how the overall vibe of your conversations has changed")
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.markdown("#### Controls")
                
                # Frequency selector
                sentiment_freq_options = {
                    "Daily": "D",
                    "Weekly": "W",
                    "Monthly": "M"
                }
                sentiment_frequency = st.selectbox(
                    "Time frequency",
                    list(sentiment_freq_options.keys()),
                    index=1,
                    key="sentiment_freq"
                )
                
                # Show by person or overall
                sentiment_view = st.radio(
                    "View",
                    ["Overall", "By Person"],
                    key="sentiment_view"
                )
                
                # Date range
                st.markdown("#### Date Range")
                min_date_sent = data[date_col].min().date()
                max_date_sent = data[date_col].max().date()
                
                date_range_sent = st.slider(
                    "Filter dates",
                    min_value=min_date_sent,
                    max_value=max_date_sent,
                    value=(min_date_sent, max_date_sent),
                    format="YYYY-MM-DD",
                    key="sentiment_date_range"
                )
            
            with col2:
                # Filter data by date range
                filtered_sent = data[
                    (data[date_col].dt.date >= date_range_sent[0]) & 
                    (data[date_col].dt.date <= date_range_sent[1])
                ]
                
                fig_sentiment_trend = go.Figure()
                
                if sentiment_view == "Overall":
                    # Calculate overall sentiment over time
                    time_sentiment = filtered_sent.groupby(
                        pd.Grouper(key=date_col, freq=sentiment_freq_options[sentiment_frequency])
                    )['sentiment'].mean()
                    
                    fig_sentiment_trend.add_trace(go.Scatter(
                        x=time_sentiment.index,
                        y=time_sentiment.values,
                        mode='lines+markers',
                        name='Sentiment',
                        line=dict(color='#7289DA', width=3),
                        marker=dict(size=6),
                        fill='tozeroy',
                        fillcolor='rgba(114, 137, 218, 0.2)'
                    ))
                    
                    # Add zero line
                    fig_sentiment_trend.add_hline(
                        y=0,
                        line_dash="dash",
                        line_color="gray",
                        annotation_text="Neutral"
                    )
                    
                else:  # By Person
                    # Calculate sentiment by person over time
                    for idx, author in enumerate(selected_authors):
                        author_data = filtered_sent[filtered_sent[author_col] == author]
                        author_sentiment = author_data.groupby(
                            pd.Grouper(key=date_col, freq=sentiment_freq_options[sentiment_frequency])
                        )['sentiment'].mean()
                        
                        fig_sentiment_trend.add_trace(go.Scatter(
                            x=author_sentiment.index,
                            y=author_sentiment.values,
                            mode='lines+markers',
                            name=author,
                            line=dict(color=color_palette[idx % len(color_palette)], width=2),
                            marker=dict(size=6)
                        ))
                    
                    # Add zero line
                    fig_sentiment_trend.add_hline(
                        y=0,
                        line_dash="dash",
                        line_color="gray",
                        annotation_text="Neutral"
                    )
                
                fig_sentiment_trend.update_layout(
                    title=f'Sentiment Over Time ({sentiment_frequency})',
                    xaxis_title='Date',
                    yaxis_title='Sentiment Score',
                    hovermode='x unified',
                    template='plotly_white',
                    height=500,
                    yaxis=dict(range=[-1, 1])
                )
                
                st.plotly_chart(fig_sentiment_trend, use_container_width=True)
                
                # Show statistics
                if sentiment_view == "Overall":
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    
                    positive_pct = (filtered_sent['sentiment'] > 0.05).sum() / len(filtered_sent) * 100
                    negative_pct = (filtered_sent['sentiment'] < -0.05).sum() / len(filtered_sent) * 100
                    neutral_pct = 100 - positive_pct - negative_pct
                    
                    with col_stat1:
                        st.metric("Positive Messages", f"{positive_pct:.1f}%")
                    with col_stat2:
                        st.metric("Neutral Messages", f"{neutral_pct:.1f}%")
                    with col_stat3:
                        st.metric("Negative Messages", f"{negative_pct:.1f}%")
                else:
                    stat_cols = st.columns(len(selected_authors))
                    for idx, author in enumerate(selected_authors):
                        author_data = filtered_sent[filtered_sent[author_col] == author]
                        avg_sentiment = author_data['sentiment'].mean()
                        
                        with stat_cols[idx]:
                            st.metric(f"{author}'s Avg", f"{avg_sentiment:.2f}")
        
        st.markdown("""
        **About sentiment scores:**
        - Scores range from -1 (very negative) to +1 (very positive)
        - 0 is neutral
        - Uses VADER sentiment analysis, which is great at handling casual text and emojis
        - A score > 0.05 is considered positive, < -0.05 is negative
        """)
    
    st.markdown("---")
    
    # Section 4: Word Analysis
    st.header("📝 Word Analysis")
    
    tab1, tab2 = st.tabs(["🔍 Track a Word", "🏆 Top Words"])
    
    with tab1:
        st.markdown("### See how word usage changes over time")
        st.markdown("Ever wonder when you started saying 'lol' less or 'lmao' more? Now you can find out.")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            word_to_count = st.text_input("Word or phrase to track", value="lol", key="word_trend")
            
            word_freq_options = {
                "Daily": "D",
                "Weekly": "W",
                "Monthly": "M"
            }
            word_frequency = st.selectbox("Time frequency", list(word_freq_options.keys()), index=2, key="word_freq")
            
            case_sensitive = st.checkbox("Case sensitive", value=False)
        
        with col2:
            if word_to_count:
                # Count keyword occurrences
                data['Keyword_Count'] = data[content_col].str.contains(
                    word_to_count, 
                    case=case_sensitive, 
                    na=False,
                    regex=False
                )
                
                # Calculate totals
                monthly_totals = data.groupby(pd.Grouper(key=date_col, freq=word_freq_options[word_frequency])).size()
                monthly_totals = monthly_totals.replace(0, float('nan'))
                
                # Create plot
                fig_word = go.Figure()
                
                for idx, author in enumerate(selected_authors):
                    author_keyword_counts = data[data[author_col] == author].groupby(
                        pd.Grouper(key=date_col, freq=word_freq_options[word_frequency])
                    )['Keyword_Count'].sum()
                    
                    author_proportions = author_keyword_counts / monthly_totals
                    
                    fig_word.add_trace(go.Scatter(
                        x=author_proportions.index,
                        y=author_proportions.values,
                        mode='lines+markers',
                        name=author,
                        line=dict(color=color_palette[idx % len(color_palette)], width=2),
                        marker=dict(size=6)
                    ))
                
                fig_word.update_layout(
                    title=f'{word_frequency} Proportion of Messages With "{word_to_count}"',
                    xaxis_title='Date',
                    yaxis_title='Proportion',
                    hovermode='x unified',
                    template='plotly_white',
                    height=500
                )
                
                st.plotly_chart(fig_word, use_container_width=True)
                
                # Show statistics
                stat_cols = st.columns(len(selected_authors) + 1)
                
                total_count = 0
                for idx, author in enumerate(selected_authors):
                    author_keyword_counts = data[data[author_col] == author].groupby(
                        pd.Grouper(key=date_col, freq=word_freq_options[word_frequency])
                    )['Keyword_Count'].sum()
                    author_total = author_keyword_counts.sum()
                    total_count += author_total
                    
                    with stat_cols[idx]:
                        st.metric(f"{author}'s uses", int(author_total))
                
                with stat_cols[-1]:
                    st.metric("Total uses", int(total_count))
    
    with tab2:
        st.markdown("### Most used words in your conversations")
        st.markdown("Find out what you and your friends talk about most (with common words filtered out).")
        
        import string
        from collections import Counter
        try:
            import nltk
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize
            
            # Download required NLTK data
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
            try:
                nltk.data.find('tokenizers/punkt_tab')
            except LookupError:
                try:
                    nltk.download('punkt_tab', quiet=True)
                except:
                    pass
            
            nltk_available = True
        except ImportError:
            nltk_available = False
            st.warning("⚠️ NLTK not installed. Run `pip install nltk` to use word analysis.")
        
        if nltk_available:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("#### Filters")
                min_date_words = data[date_col].min().date()
                max_date_words = data[date_col].max().date()
                
                date_range_words = st.date_input(
                    "Date range",
                    value=(min_date_words, max_date_words),
                    min_value=min_date_words,
                    max_value=max_date_words,
                    key="word_date_range"
                )
                
                if len(date_range_words) == 2:
                    start_date_words, end_date_words = date_range_words
                else:
                    start_date_words = date_range_words[0]
                    end_date_words = max_date_words
                
                # User filter
                user_filter_options = ["Everyone"] + selected_authors
                user_filter = st.radio(
                    "Whose words?",
                    user_filter_options,
                    key="user_filter"
                )
                
                n_words = st.slider("How many words?", min_value=5, max_value=50, value=10, step=5)
                
                st.markdown("#### Custom Stop Words")
                custom_stop_words_input = st.text_area(
                    "Words to exclude (comma-separated)",
                    value="um, uh, like",
                    help="Add filler words or inside jokes to exclude"
                )
                custom_stop_words = set([word.strip() for word in custom_stop_words_input.split(',')])
            
            with col2:
                filtered_data_words = data[
                    (data[date_col].dt.date >= start_date_words) & 
                    (data[date_col].dt.date <= end_date_words)
                ]
                
                if user_filter != "Everyone":
                    filtered_data_words = filtered_data_words[filtered_data_words[author_col] == user_filter]
                
                stop_words = set(stopwords.words('english'))
                custom_stop_words.update({'', "'", ' ', "'"})
                
                def preprocess_text(text):
                    if pd.isna(text):
                        return ""
                    text = str(text).lower()
                    text = text.translate(str.maketrans('', '', string.punctuation))
                    words = word_tokenize(text)
                    words = [word for word in words if word not in stop_words and word not in custom_stop_words and len(word) > 1]
                    return words
                
                all_words = []
                for text in filtered_data_words[content_col].dropna():
                    all_words.extend(preprocess_text(text))
                
                word_counts = Counter(all_words)
                top_words = word_counts.most_common(n_words)
                
                if top_words:
                    words_list = [word for word, count in top_words]
                    counts_list = [count for word, count in top_words]
                    
                    fig_top_words = go.Figure(data=[
                        go.Bar(
                            x=counts_list,
                            y=words_list,
                            orientation='h',
                            marker=dict(
                                color=counts_list,
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title="Count")
                            ),
                            text=counts_list,
                            textposition='auto',
                        )
                    ])
                    
                    fig_top_words.update_layout(
                        title=f'Top {n_words} Words',
                        xaxis_title='Count',
                        yaxis_title='Word',
                        height=max(400, n_words * 30),
                        template='plotly_white',
                        yaxis=dict(autorange="reversed")
                    )
                    
                    st.plotly_chart(fig_top_words, use_container_width=True)
                    
                    with st.expander("📊 See the full list"):
                        word_df = pd.DataFrame(top_words, columns=['Word', 'Count'])
                        word_df.index = word_df.index + 1
                        word_df.index.name = 'Rank'
                        st.dataframe(word_df, use_container_width=True)
                else:
                    st.info("No words found in this date range.")
    
    # Section 5: OpenAI Summaries (only if API key provided)
    if api_key_input:
        st.markdown("---")
        st.header("🤖 Daily Summaries")
        st.markdown("Get an AI-generated summary of any day's conversation. Great for remembering what you talked about months or years ago.")
        
        try:
            from openai import OpenAI
            openai_available = True
        except ImportError:
            openai_available = False
            st.error("⚠️ OpenAI library not installed. Run `pip install openai` to use this feature.")
        
        if openai_available:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("#### Settings")
                
                available_dates = sorted(data[date_col].dt.date.unique(), reverse=True)
                
                selected_date = st.date_input(
                    "Pick a day",
                    value=available_dates[0] if len(available_dates) > 0 else datetime.now().date(),
                    min_value=min(available_dates) if len(available_dates) > 0 else None,
                    max_value=max(available_dates) if len(available_dates) > 0 else None,
                    key="summary_date"
                )
                
                model_choice = st.selectbox(
                    "Model",
                    ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                    help="gpt-4o-mini is cheapest and usually good enough"
                )
                
                summary_style = st.radio(
                    "Style",
                    ["Detailed", "Brief", "Highlights Only"]
                )
                
                generate_summary = st.button("✨ Summarize This Day", type="primary", use_container_width=True)
            
            with col2:
                if generate_summary:
                    start_time = pd.Timestamp(selected_date).tz_localize('UTC')
                    end_time = start_time + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                    
                    filter_data = data.loc[(data[date_col] >= start_time) & (data[date_col] <= end_time)]
                    
                    if len(filter_data) == 0:
                        st.warning(f"No messages on {selected_date}")
                    else:
                        st.info(f"Found {len(filter_data)} messages on {selected_date}")
                        
                        conversation_messages = []
                        for _, row in filter_data.iterrows():
                            time_str = row[date_col].strftime("%H:%M")
                            conversation_messages.append(f"[{time_str}] {row[author_col]}: {row[content_col]}")
                        
                        conversation_text = "\n".join(conversation_messages)
                        
                        # Build prompt based on style
                        participant_names = " and ".join(selected_authors)
                        
                        if summary_style == "Detailed":
                            prompt = f"""This is a day's conversation between {participant_names} on {selected_date}. 

Give me a detailed summary in 3-5 paragraphs covering:
- Main topics and themes
- Any important events, plans, or decisions
- Specific names of people, places, media, or things mentioned
- Overall vibe and mood
- Any funny or memorable moments

Conversation:

{conversation_text}"""
                        elif summary_style == "Brief":
                            prompt = f"""Summarize this conversation between {participant_names} from {selected_date} in 1-2 short paragraphs. Just hit the main points.

{conversation_text}"""
                        else:
                            prompt = f"""What were the key highlights from this conversation between {participant_names} on {selected_date}? Give me a bullet list.

{conversation_text}"""
                        
                        estimated_tokens = len(conversation_text.split()) * 1.3
                        if estimated_tokens > 120000:
                            st.warning(f"⚠️ This is a long conversation (~{int(estimated_tokens)} tokens). Might be pricey or fail.")
                        
                        with st.spinner("🤖 Reading through your messages..."):
                            try:
                                client = OpenAI(api_key=api_key_input)
                                
                                response = client.chat.completions.create(
                                    model=model_choice,
                                    messages=[
                                        {"role": "system", "content": "You're summarizing a conversation between friends. Be specific, capture the vibe, and keep it natural."},
                                        {"role": "user", "content": prompt}
                                    ],
                                    temperature=0.7,
                                    max_tokens=1000
                                )
                                
                                summary = response.choices[0].message.content
                                
                                st.markdown("### 📝 Summary")
                                st.markdown(f"**Date:** {selected_date} | **Messages:** {len(filter_data)} | **Model:** {model_choice}")
                                st.markdown("---")
                                
                                st.markdown(
                                    f"""<div style='background-color: rgba(114, 137, 218, 0.1); padding: 20px; border-radius: 10px; border-left: 4px solid #7289DA;'>
                                    {summary}
                                    </div>""",
                                    unsafe_allow_html=True
                                )
                                
                                if hasattr(response, 'usage'):
                                    with st.expander("📊 Token usage"):
                                        col_usage1, col_usage2, col_usage3 = st.columns(3)
                                        with col_usage1:
                                            st.metric("Input", response.usage.prompt_tokens)
                                        with col_usage2:
                                            st.metric("Output", response.usage.completion_tokens)
                                        with col_usage3:
                                            st.metric("Total", response.usage.total_tokens)
                                
                                summary_download = f"""Conversation Summary
Date: {selected_date}
Participants: {participant_names}
Messages: {len(filter_data)}
Model: {model_choice}

---

{summary}

---

Generated by Conversation Analyzer
"""
                                st.download_button(
                                    label="💾 Download",
                                    data=summary_download,
                                    file_name=f"summary_{selected_date}.txt",
                                    mime="text/plain",
                                    use_container_width=True
                                )
                                
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
                                if "api_key" in str(e).lower():
                                    st.info("💡 Check that your API key is valid and has credits")
                                elif "rate_limit" in str(e).lower():
                                    st.info("💡 Rate limited - wait a bit and try again")
                else:
                    st.info("👈 Pick a date and hit the button to generate a summary!")
                    
                    st.markdown("""
                    **How it works:**
                    
                    1. Pick any day from your conversation history
                    2. Choose how detailed you want it
                    3. Click the button and wait for the AI to do its thing
                    
                    **A few notes:**
                    - Your API key isn't stored anywhere
                    - Costs are usually just a few cents per summary
                    - gpt-4o-mini is cheap and works great
                    - Really long days (1000+ messages) will cost more
                    """)
    
    # Sample data viewer
    with st.expander("📋 Peek at your data"):
        st.dataframe(data.head(20))

else:
    st.info("👈 Upload a CSV or JSON file to get started")
    
    st.markdown("""
    ## What this does
    
    This tool helps you analyze conversation history from any messaging platform. I originally built it to look at 6 years 
    of Discord messages with my best friend, but it works with any CSV or JSON export.
    
    ## Features
    
    - 📈 **Message trends** - See how active your conversation has been over time
    - 🥧 **Who talks more** - Compare message counts between people (no judgment)
    - 😊 **Sentiment analysis** - Visualize conversation vibes with calendars and trend charts
    - 🔍 **Word tracking** - Find out when you started/stopped using certain words
    - 🏆 **Top words** - See what you talk about most
    - 🤖 **AI summaries** - Get daily summaries of your conversations (requires OpenAI API key)
    
    ## How to use it
    
    1. Export your messages as CSV or JSON
    2. Upload the file using the sidebar
    3. Tell the app which columns have dates, usernames, and messages
    4. Explore the visualizations
    5. (Optional) Add an OpenAI API key to unlock AI-powered summaries
    
    ## File format
    
    Your file should have at least:
    - A date/timestamp column
    - An author/username column  
    - A message content column
    
    Example CSV:
    ```
    Date,Author,Content
    2020-01-01 10:30:00,alice,hey!
    2020-01-01 10:31:00,bob,what's up
    ```
    
    ## Tech used
    
    Built with pandas, plotly, NLTK, scikit-learn, and OpenAI's API. Made as a personal project to learn data science tools.
    """)