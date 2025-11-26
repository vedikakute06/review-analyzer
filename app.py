import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
# Import the class and categories from the sibling file
from review_analyzer import ReviewAnalyzer, ZERO_SHOT_CATEGORIES, LABEL_MAPPING


# --- CACHING FUNCTION FOR MODEL LOADING ---
@st.cache_resource
def load_and_cache_models():
    """Loads and initializes the ReviewAnalyzer models."""
    analyzer = ReviewAnalyzer()
    analyzer.load_models()
    return analyzer


# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Balanced Review Analyzer Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- INITIALIZATION ---
if 'df_analyzed' not in st.session_state:
    st.session_state.df_analyzed = None
if 'stats' not in st.session_state:
    st.session_state.stats = None

# Using the mapping from review_analyzer for consistency and ensuring all keys are present
SENTIMENT_COLORS = {
    'POSITIVE': '#2ecc71',
    'NEGATIVE': '#e74c3c',
    'NEUTRAL': '#95a5a6'
}

# Use a default chart theme since the option is removed
DEFAULT_CHART_THEME = "plotly_white"

# --- SIDEBAR (Settings) ---
with st.sidebar:
    st.title("Analysis Settings")

    sample_size = st.number_input(
        "Sample Size (0 = All)",
        min_value=0,
        max_value=10000,
        value=500,
        step=100,
        help="Limit number of reviews to analyze (0 for all)"
    )

    # --- Use all categories ---
    st.markdown("### Classification Topics")
    # Set selected_categories to ALL available categories
    selected_categories = ZERO_SHOT_CATEGORIES

    st.success("Using ALL predefined categories for classification.")
    st.caption(f"Topics: {', '.join(selected_categories)}")

    st.markdown("---")
    st.markdown("Models: BERTweet (Sentiment), BART-MNLI (Classification), BART-CNN (Summarization)")

# --- MODEL LOADING LOGIC ---
try:
    with st.spinner("Initializing/Loading NLP Models..."):
        analyzer = load_and_cache_models()
    st.sidebar.success("Models are ready!")
except Exception as e:
    st.sidebar.error(f"Model Loading Error: {e}")
    analyzer = None

# --- TITLE & TABS ---
st.title("Customer Review Analyzer")
st.markdown("---")

# Only two tabs remain
tab1, tab2 = st.tabs([
    "1. Upload & Analyze",
    "2. Dashboard"
])

# ==============================================================================
# TAB 1: Upload & Analyze
# ==============================================================================
with tab1:
    st.header("Upload Your Review Data")

    # Upload and Info side-by-side
    col_up, col_info = st.columns([2, 1])

    with col_up:
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file containing customer reviews"
        )

    with col_info:
        st.markdown("**CSV Requirements:**")
        st.markdown("* Must have a review text column (e.g., `review_text`).")
        st.markdown("* Optional: A rating column (e.g., `rating`, `stars`).")

    if uploaded_file is not None and analyzer is not None:
        st.success("File uploaded. Check sidebar settings before running.")

        if st.button("Run AI Analysis", type="primary"):
            if not selected_categories:
                st.error("Error: Classification categories are empty.")
            else:
                with st.spinner("Analyzing reviews... This may take a few moments..."):
                    try:
                        # 1. Load and Prepare Data
                        sample = sample_size if sample_size > 0 else None
                        df = analyzer.load_and_prepare_data(
                            uploaded_file,
                            sample_size=sample
                        )
                        st.info(f"Processing {len(df)} reviews...")

                        # 2. Analyze
                        df_analyzed = analyzer.analyze_dataframe(df, categories=selected_categories)
                        st.session_state.df_analyzed = df_analyzed

                        # 3. Generate Stats
                        stats = analyzer.generate_summary_stats(df_analyzed)
                        st.session_state.stats = stats

                        st.success(f"Analysis complete! {len(df_analyzed):,} reviews analyzed.")
                        st.balloons()

                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
    elif analyzer is None:
        st.warning("Models failed to load. Check the sidebar for error details.")

# ==============================================================================
# TAB 2: Dashboard
# ==============================================================================
with tab2:
    if st.session_state.df_analyzed is not None and st.session_state.stats is not None:
        df = st.session_state.df_analyzed
        stats = st.session_state.stats

        st.header("Analytics Dashboard")
        st.markdown("---")

        # 2.1 Key Metrics Row
        st.markdown("### Key Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("Total Reviews", f"{stats['total_reviews']:,}")
        col2.metric("Positive %", f"{stats['positive_pct']:.1f}%")
        col3.metric("Negative %", f"{stats['negative_pct']:.1f}%")

        if 'avg_rating' in stats:
            col4.metric("Avg Rating", f"{stats['avg_rating']:.2f}")
            col5.metric("Avg Confidence", f"{stats['avg_sentiment_confidence']:.1%}")
        else:
            col4.metric("Neutral %", f"{stats['neutral_pct']:.1f}%")
            col5.metric("Avg Confidence", f"{stats['avg_sentiment_confidence']:.1%}")

        st.markdown("---")

        # 2.2 Sentiment Breakdown Chart (New)
        st.markdown("### Sentiment Breakdown")
        sentiment_counts = df['sentiment'].value_counts().reindex(
            list(SENTIMENT_COLORS.keys()), fill_value=0
        )

        fig_sent = px.pie(
            names=sentiment_counts.index,
            values=sentiment_counts.values,
            color=sentiment_counts.index,
            color_discrete_map=SENTIMENT_COLORS,
            title='Overall Sentiment Distribution',
            template=DEFAULT_CHART_THEME
        )
        fig_sent.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#000000', width=1)))
        fig_sent.update_layout(height=400)
        st.plotly_chart(fig_sent, use_container_width=True)

        st.markdown("---")

        # 2.3 Top Categories (Original 2.2)
        st.markdown("### Top Categories")
        category_counts = df['category'].value_counts().head(10)

        fig_bar = px.bar(
            x=category_counts.values,
            y=category_counts.index,
            orientation='h',
            labels={'x': 'Count', 'y': 'Category'},
            template=DEFAULT_CHART_THEME,
            color_discrete_sequence=['#1f77b4'],
        )

        fig_bar.update_layout(height=400, showlegend=False, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")

        # 2.4 Rating Analysis (if available)
        if 'rating' in df.columns:
            st.markdown("### Rating Analysis")

            col_rating, col_rating_sent = st.columns(2)

            with col_rating:
                rating_counts = df['rating'].value_counts().sort_index()
                fig_rating = px.bar(
                    x=rating_counts.index, y=rating_counts.values, labels={'x': 'Rating', 'y': 'Count'},
                    template=DEFAULT_CHART_THEME, color=rating_counts.values, color_continuous_scale='RdYlGn'
                )
                fig_rating.update_layout(height=350)
                st.plotly_chart(fig_rating, use_container_width=True)

            with col_rating_sent:
                sentiment_rating = df.groupby(['rating', 'sentiment']).size().reset_index(name='count')
                fig_rating_sentiment = px.bar(
                    sentiment_rating, x='rating', y='count', color='sentiment', barmode='stack',
                    template=DEFAULT_CHART_THEME, color_discrete_map=SENTIMENT_COLORS,
                    labels={'rating': 'Rating', 'count': 'Count'}
                )
                fig_rating_sentiment.update_layout(height=350)
                st.plotly_chart(fig_rating_sentiment, use_container_width=True)

        st.markdown("---")

        # 2.5 Deep NLP Insights: Summarization and Keywords (New Section)
        st.header("Insights")

        # --- Summarization ---
        st.subheader("Automated Summary of Key Reviews")
        # Ensure we are using the globally available analyzer instance
        try:
            summary_text = analyzer.summarize_reviews(df['review_text'].tolist())
            st.success(summary_text)
        except Exception as e:
            st.warning(f"Summarization Error: {e}")

        # --- Keywords ---
        st.subheader("Top Keywords by Sentiment")

        col_kw_pos, col_kw_neg = st.columns(2)

        with col_kw_pos:
            st.markdown("#### Positive Themes")
            # Using the static method directly via the class
            top_pos_kw = ReviewAnalyzer.get_top_keywords(df, sentiment='POSITIVE', top_n=8)
            if top_pos_kw:
                st.table(pd.DataFrame(top_pos_kw, columns=['Keyword', 'Count']))
            else:
                st.info("Not enough positive reviews to extract keywords.")

        with col_kw_neg:
            st.markdown("#### Negative Themes")
            top_neg_kw = ReviewAnalyzer.get_top_keywords(df, sentiment='NEGATIVE', top_n=8)
            if top_neg_kw:
                st.table(pd.DataFrame(top_neg_kw, columns=['Keyword', 'Count']))
            else:
                st.info("Not enough negative reviews to extract keywords.")

        st.markdown("---")

        # 2.6 Download Button (Original 2.4, now 2.6)
        st.markdown("### Download Results")
        if st.session_state.df_analyzed is not None:
            full_csv = st.session_state.df_analyzed.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download All Analyzed Results (CSV)",
                data=full_csv,
                file_name=f"all_analyzed_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )


    else:
        st.info("Please upload and analyze reviews in the 'Upload & Analyze' tab first.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #7f8c8d;'>
        <p>AI-Driven Review Analytics | Minimal Dashboard UI</p>
    </div>
""", unsafe_allow_html=True)