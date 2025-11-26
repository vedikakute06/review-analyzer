# Customer Review Analyzer Dashboard


An AI-powered customer review analysis tool that uses state-of-the-art NLP models to analyze sentiment, categorize feedback, and extract actionable insights from customer reviews.

**Link**: https://review-analyzer-ai.streamlit.app/

## Features

### AI-Powered Analysis
- **Sentiment Analysis**: Classify reviews as Positive, Negative, or Neutral using BERTweet
- **Category Classification**: Automatically categorize reviews into topics (Product Quality, Shipping, Customer Service, etc.)
- **Review Summarization**: Generate concise summaries from multiple reviews using BART-CNN
- **Keyword Extraction**: Identify key terms from positive and negative reviews

### Interactive Dashboard
- **Real-time Analytics**: Visualize sentiment distribution, category breakdowns, and trends
- **Advanced Filtering**: Filter reviews by sentiment, category, rating, and keywords
- **Custom Visualizations**: Interactive Plotly charts with multiple themes
- **Confidence Scores**: View model confidence for each prediction

### Deep Dive Analysis
- **Keyword Analysis**: Discover most common words in positive/negative reviews
- **Sentiment by Category**: See how sentiment varies across different topics
- **Rating Correlation**: Analyze relationship between ratings and sentiment
- **Review Explorer**: Search and browse individual reviews with full metadata

### Export Capabilities
- Download filtered results as CSV
- Export complete analyzed dataset
- Timestamped file naming for version control

## Quick Start

### Prerequisites

- Python 3.8 or higher
- 4GB+ RAM recommended
- GPU optional (will use CPU if not available)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/review-analyzer.git
cd review-analyzer
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download spaCy language model**
```bash
python -m spacy download en_core_web_sm
```

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## Project Structure

```
review-analyzer/
│
├── app.py                      # Main Streamlit application
├── review_analyzer.py          # Core NLP analysis engine
├── requirements.txt            # Python dependencies
├── README.md                   # This file

```

## Input Data Format

Your CSV file should contain review text in one of these column names:
- `review_text`, `text`, `review`, `content`, `reviewText`, `Text`, `Review`, `reviews`, or `customer_review`

Optional rating column (automatically detected):
- `rating`, `star_rating`, `stars`, `score`, `Rating`, or `Star`

### Example CSV Format

```csv
review_text,rating
"Great product! Exceeded my expectations.",5
"Shipping was delayed by 3 days.",2
"Good value for money. Would recommend.",4
"Poor quality. Broke after one week.",1
"Decent product but overpriced.",3
```

## Usage Guide

### 1. Upload & Analyze Tab

1. **Upload your CSV file**
   - Click "Browse files" and select your review data
   - Supported format: CSV with text and optional rating columns

2. **Configure settings** (in sidebar)
   - Set sample size (0 = analyze all reviews)
   - Choose visualization theme

3. **Load models**
   - Click "Load & Initialize Models"
   - Wait for models to download and initialize (first run takes longer)

4. **Analyze reviews**
   - Click "Analyze Reviews"
   - Processing time depends on dataset size (~1-2 reviews/second on CPU)

### 2. Dashboard Tab

View high-level analytics:
- **Key Metrics**: Total reviews, sentiment breakdown, confidence scores
- **Sentiment Distribution**: Visual breakdown of positive/negative/neutral
- **Category Analysis**: Most common review topics
- **Confidence Distribution**: Model prediction reliability
- **Rating Analysis**: If ratings are available

### 3. Deep Dive Tab

Explore detailed insights:
- **Filter reviews** by sentiment, category, and rating
- **View top keywords** for positive and negative reviews
- **Generate AI summaries** for specific review segments
- Analyze sentiment patterns across categories

### 4. Review Explorer Tab

Browse individual reviews:
- **Search** reviews using keywords
- **Filter** by multiple criteria
- **Sort** by confidence, rating, or other fields
- **Export** filtered results to CSV

## Configuration

### Model Settings

The application uses three transformer models:

| Task | Model | Purpose |
|------|-------|---------|
| Sentiment Analysis | `finiteautomata/bertweet-base-sentiment-analysis` | Classify positive/negative/neutral |
| Category Classification | `facebook/bart-large-mnli` | Zero-shot topic categorization |
| Summarization | `facebook/bart-large-cnn` | Generate review summaries |

### Categories

Default review categories (customizable in `review_analyzer.py`):
- Product Quality (Defect or Condition)
- Shipping and Delivery Experience
- Customer Service and Support
- Value for Money and Pricing
- Product Features and Functionality
- Durability and Longevity

### Performance Tuning

Adjust these constants in `ReviewAnalyzer` class:

```python
BATCH_SIZE = 32  # Increase for faster processing (requires more RAM)
SENTIMENT_MAX_LENGTH = 128  # Maximum tokens for sentiment analysis
CLASSIFICATION_MAX_LENGTH = 512  # Maximum tokens for classification
```

## Performance

### Processing Speed

| Hardware | Reviews/Second | 100 Reviews | 1000 Reviews |
|----------|---------------|-------------|--------------|
| CPU (i7) | ~1-2 | ~1 min | ~10 min |
| GPU (RTX 3060) | ~10-15 | ~10 sec | ~2 min |

### Memory Usage

- **Minimum**: 4GB RAM
- **Recommended**: 8GB RAM
- **With GPU**: 6GB VRAM for optimal performance

## Troubleshooting

### Common Issues

**Models not loading**
```bash
# Clear transformers cache and reinstall
rm -rf ~/.cache/huggingface/
pip install --upgrade transformers
```

**Out of memory errors**
- Reduce `BATCH_SIZE` in `review_analyzer.py`
- Use sample size to analyze fewer reviews
- Close other applications

**spaCy model not found**
```bash
python -m spacy download en_core_web_sm
```

**Slow performance**
- Enable GPU if available
- Increase `BATCH_SIZE` for faster processing
- Use sample size for large datasets

## Dependencies

```
streamlit>=1.28.0
pandas>=1.5.0
plotly>=5.17.0
transformers>=4.30.0
torch>=2.0.0
spacy>=3.6.0
numpy>=1.24.0
```

See `requirements.txt` for complete list with version pinning.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black .
flake8 .
```

## Acknowledgments

- **Hugging Face** for transformer models and the Transformers library
- **Streamlit** for the amazing web framework
- **Plotly** for interactive visualizations
- **spaCy** for NLP preprocessing

## Roadmaps
### Version History

#### v1.0.0 (Current)
- Initial release
- Sentiment analysis, categorization, summarization
- Interactive Streamlit dashboard
- Export capabilities

## Disclaimer

This tool uses AI models for analysis. While generally accurate, predictions should be validated for critical business decisions. Model performance may vary based on:
- Review language and quality
- Domain-specific terminology
- Dataset size and diversity

## Star History

If you find this project useful, please consider giving it a star!

---

**Made with Python, Streamlit, and Transformers**

![Footer](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Footer](https://img.shields.io/badge/Powered%20by-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
