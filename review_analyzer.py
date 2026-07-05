from transformers import pipeline
import torch
import pandas as pd
from collections import Counter
import re
import spacy
from typing import List, Dict, Any, Optional

# Load spaCy model once
try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_STOPWORDS = nlp.Defaults.stop_words
except OSError:
    # Handle case where the model might not be downloaded
    # To fix, run: python -m spacy download en_core_web_sm
    print("SpaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    nlp = None
    SPACY_STOPWORDS = set()

# Categories for zero-shot classification
ZERO_SHOT_CATEGORIES = [
    "Product Quality (Defect or Condition)",
    "Shipping and Delivery Experience",
    "Customer Service and Support",
    "Value for Money and Pricing",
    "Product Features and Functionality",
    "Durability and Longevity"
]

# --- FIX: LABEL MAPPING FOR BERTWEET MODEL OUTPUT ---
LABEL_MAPPING = {
    'POS': 'POSITIVE',
    'NEG': 'NEGATIVE',
    'NEU': 'NEUTRAL',
    # Include full labels for robustness
    'POSITIVE': 'POSITIVE',
    'NEGATIVE': 'NEGATIVE',
    'NEUTRAL': 'NEUTRAL',
}
# ---------------------------------------------------
TEXT_COLUMN_ALIASES = [
    'review_text', 'text', 'review', 'content', 'reviewText', 'Text', 'Review', 'reviews',
    'customer_review', 'tweet_text', 'full_text', 'comment', 'comments', 'message', 'feedback'
]


class ReviewAnalyzer:
    """Main NLP Analysis Handler for Reviews"""
    BATCH_SIZE = 32
    MIN_WORDS_FOR_SUMMARY_INPUT = 40

    def __init__(self):
        """Initialize class attributes"""
        # Determine device
        self.device = 0 if torch.cuda.is_available() else -1
        self.sentiment_analyzer = None
        self.classifier = None
        self.summarizer = None
        self.is_loaded = False

    # ------------------------
    # Load HuggingFace Models
    # ------------------------
    def load_models(self):
        """Initializes and loads all required HuggingFace pipelines."""
        if self.is_loaded:
            return self

        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="finiteautomata/bertweet-base-sentiment-analysis",
                device=self.device
            )

            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=self.device
            )

            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=self.device
            )
            self.is_loaded = True
            return self
        except Exception as e:
            # Propagate error up to Streamlit for display
            raise Exception(
                f"Failed to load models. Check your internet connection or package installation. Error: {e}")

    # ------------------------
    # Sentiment Analysis
    # ------------------------
    def analyze_sentiment_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        if not self.sentiment_analyzer:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        # BERTweet model limit is 128 tokens, we truncate to be safe.
        processed_texts = [text[:256] for text in texts]  # Increased limit slightly for better context

        results = self.sentiment_analyzer(
            processed_texts,
            batch_size=self.BATCH_SIZE,
            truncation=True
        )

        return [{
            'sentiment': r['label'],
            'confidence': round(r['score'], 3)
        } for r in results]

    # ------------------------
    # Zero-Shot Category Classification
    # ------------------------
    def classify_category_batch(self, texts: List[str], categories: List[str] = ZERO_SHOT_CATEGORIES) -> List[
        Dict[str, Any]]:
        if not self.classifier:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        processed_texts = [text[:512] for text in texts]  # BART-MNLI can handle more

        results = self.classifier(
            processed_texts,
            candidate_labels=categories,
            multi_label=False,
            batch_size=self.BATCH_SIZE,
            truncation=True
        )

        return [{
            'category': r['labels'][0],
            'confidence': round(r['scores'][0], 3)
        } for r in results]

    # ------------------------
    # Summarization
    # ------------------------
    def summarize_reviews(self, reviews_list: List[str], max_length: int = 130, min_length: int = 30) -> str:
        if not self.summarizer:
            return "Summarization model not loaded."

        try:
            long_reviews = [
                review for review in reviews_list
                if len(review.split()) >= self.MIN_WORDS_FOR_SUMMARY_INPUT
            ]

            if not long_reviews:
                return "No sufficiently long reviews found (min 40 words) to generate a meaningful summary."

            # Combine a maximum of 50 long reviews for context
            combined_text = " ".join(long_reviews[:50])
            combined_text = combined_text[:3000]  # Limit total text size for BART-CNN (max 1024 tokens)

            if len(combined_text.split()) < 50:
                return "Combined text from long reviews is still too short (fewer than 50 words) to generate a meaningful summary."

            summary = self.summarizer(
                combined_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True
            )
            return summary[0]['summary_text']

        except Exception as e:
            return f"Summarization failed: {str(e)}. Try reducing the number of reviews to summarize."

    # ------------------------
    # Apply All Analysis to DataFrame
    # ------------------------
    def analyze_dataframe(self, df: pd.DataFrame, categories: List[str] = ZERO_SHOT_CATEGORIES) -> pd.DataFrame:
        if 'review_text' not in df.columns:
            raise KeyError("DataFrame must contain a 'review_text' column.")
        if not self.is_loaded:
            raise RuntimeError("Models not loaded.")

        reviews = df['review_text'].tolist()

        # Sentiment
        sentiment_results = self.analyze_sentiment_batch(reviews)
        # --- FIX: Apply mapping to convert 'POS' to 'POSITIVE', etc. ---
        df['sentiment'] = [LABEL_MAPPING.get(r['sentiment'], r['sentiment']) for r in sentiment_results]
        df['sentiment_confidence'] = [r['confidence'] for r in sentiment_results]

        # Categories
        classification_results = self.classify_category_batch(reviews, categories=categories)
        df['category'] = [r['category'] for r in classification_results]
        df['category_confidence'] = [r['confidence'] for r in classification_results]

        return df

    # ------------------------
    # STATIC HELPERS
    # ------------------------
    @staticmethod
    def clean_text(text: Any) -> str:
        """Standard text cleaning for reviews."""
        text = str(text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # Allows for alphanumeric, spaces, and essential punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        text = ' '.join(text.split())
        return text.strip()

    @staticmethod
    def load_and_prepare_data(file: Any, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Loads and prepares the CSV file, identifies text/rating columns."""
        df = pd.read_csv(file)

        # Common names for review text and rating columns
        text_cols = TEXT_COLUMN_ALIASES
        rating_cols = ['rating', 'star_rating', 'stars', 'score', 'Rating', 'Star']

        # Find and rename columns (case-insensitive search for flexibility)
        found_text_col = next((col for col in df.columns if col.lower() in [c.lower() for c in text_cols]), None)
        if not found_text_col:
            raise ValueError(f"No text column found. Expected one of: {text_cols}")

        found_rating_col = next((col for col in df.columns if col.lower() in [c.lower() for c in rating_cols]), None)

        df = df.rename(columns={found_text_col: 'review_text'})
        if found_rating_col:
            df = df.rename(columns={found_rating_col: 'rating'})

        # Drop rows where review_text is missing
        df = df.dropna(subset=['review_text'])
        df['review_text'] = df['review_text'].astype(str).apply(ReviewAnalyzer.clean_text)

        # Filter out very short reviews (e.g., just punctuation or a single word)
        df = df[df['review_text'].str.len() > 10]
        if df.empty:
            raise ValueError("No review rows remain after removing blank or very short text.")

        # Apply sampling
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

        return df

    @staticmethod
    def get_top_keywords(df: pd.DataFrame, sentiment: Optional[str] = None, top_n: int = 10) -> List[tuple]:
        """Extracts and counts top non-stopword keywords from reviews."""
        if nlp is None:
            return []  # Cannot proceed without SpaCy model

        if sentiment:
            texts = df[df['sentiment'] == sentiment]['review_text']
        else:
            texts = df['review_text']

        words = []
        for text in texts:
            # Process text with SpaCy
            doc = nlp(text.lower())
            tokens = [
                token.text for token in doc
                if not token.is_punct and not token.is_space
                   and token.text not in SPACY_STOPWORDS
                   and len(token.text) > 3
            ]
            words.extend(tokens)

        return Counter(words).most_common(top_n)

    @staticmethod
    def generate_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates key statistical summaries for the dashboard."""
        total = len(df)

        sentiment_counts = df['sentiment'].value_counts(normalize=True).reindex(
            ['POSITIVE', 'NEGATIVE', 'NEUTRAL'], fill_value=0
        ) * 100

        category_counts = df['category'].value_counts()

        stats = {
            'total_reviews': total,
            'positive_count': df['sentiment'].value_counts().get('POSITIVE', 0),
            'negative_count': df['sentiment'].value_counts().get('NEGATIVE', 0),
            'neutral_count': df['sentiment'].value_counts().get('NEUTRAL', 0),
            'positive_pct': round(sentiment_counts['POSITIVE'], 2),
            'negative_pct': round(sentiment_counts['NEGATIVE'], 2),
            'neutral_pct': round(sentiment_counts['NEUTRAL'], 2),
            'top_category': category_counts.index[0] if not category_counts.empty else 'N/A',
            'top_category_count': category_counts.values[0] if not category_counts.empty else 0,
            'avg_sentiment_confidence': round(df['sentiment_confidence'].mean(), 3),
            'avg_category_confidence': round(df['category_confidence'].mean(), 3),
        }

        if 'rating' in df.columns:
            # Ensure rating is treated as numeric
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
            stats['avg_rating'] = round(df['rating'].mean(), 2)
            stats['rating_std'] = round(df['rating'].std(), 2)

        return stats
