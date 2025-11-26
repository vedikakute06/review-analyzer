# 📈 AI-Driven Customer Review Analyzer Dashboard

This is a powerful, interactive dashboard built with **Streamlit** that utilizes advanced **Hugging Face NLP models** to analyze customer feedback from CSV files. It quickly processes thousands of reviews, providing actionable insights into sentiment, key topics, and overall themes.

## ✨ Features

* **Batch Sentiment Analysis:** Uses **BERTweet** to classify reviews as Positive, Negative, or Neutral.
* **Zero-Shot Topic Classification:** Uses **BART-MNLI** to automatically categorize reviews into predefined business topics (e.g., "Product Quality," "Customer Service," "Shipping").
* **Automated Summarization:** Uses **BART-CNN** to generate a concise summary of the overall review corpus, focusing on long, high-context reviews.
* **Interactive Dashboard:** Displays key metrics, sentiment distribution charts, top keywords by sentiment, and rating analysis (if rating data is available).
* **Scalable:** Supports processing large datasets via customizable sampling.

## 🚀 How to Run Locally

### Prerequisites

1.  **Python 3.8+**
2.  **Git** (for cloning)
3.  **Hugging Face Transformer Models** (installed via requirements.txt)

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
    cd YOUR_REPO_NAME
    ```
    *(Replace `YOUR_USERNAME/YOUR_REPO_NAME` with your actual repository path)*

2.  **Create and Activate Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # .\venv\Scripts\activate   # On Windows
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The `torch` and `spacy` libraries can take time to install.*

4.  **Download SpaCy Model:**
    ```bash
    python -m spacy download en_core_web_sm
    ```

### Running the App

Execute the following command in your terminal from the project root:

```bash
streamlit run app.py
