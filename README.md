# Flask NLP & Trading API

## Description
This is a Flask-based API that provides various functionalities for **Natural Language Processing (NLP)** and **Stock Market Analysis**. 

### **NLP Features:**
- **Text Classification**
- **Named Entity Recognition (NER)**
- **Question Answering**
- **Summarization**
- **Translation**
- **Text Generation**
- **Wikipedia Document Search**
- **Feature Extraction**
- **Sentiment Analysis**
- **POS Tagging**
- **Noun Chunking**

### **Stock Market Analysis Features:**
- **Plot Daily Stock Price Equity**
- **Technical Indicators for Equities**
- **Stock Volume vs. Moving Averages**
- **Daily Volume Change**
- **Volume Surges vs. Price**
- **Relative Strength Index (RSI)**
- **MACD Analysis**
- **Volume Prediction**
- **Prophet-Based Volume Prediction for Multiple Stocks**

---

## Installation

### **1. Clone the Repository**
```bash
git clone https://github.com/abdelouahedlabrigui/ai_repo.git
cd YOUR-REPO
```

### **2. Create a Virtual Environment (Optional but Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scriptsctivate
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## Usage

### **Run the Flask App**
```bash
python app.py
```
By default, the server runs on **http://0.0.0.0:5005/**.

### **Example API Calls**

#### **NLP Endpoints:**

- **Text Classification:**
  ```http
  GET /llms/text_classification?title=ExampleTitle&text=ExampleText&sector=Finance
  ```

- **Named Entity Recognition (NER):**
  ```http
  GET /llms/ner_tagger?title=ExampleTitle&text=ExampleText&sector=Healthcare
  ```

- **Question Answering:**
  ```http
  GET /llms/question_answering?title=ExampleTitle&text=ExampleText&question=What is AI?&sector=Education
  ```

- **Summarization:**
  ```http
  GET /llms/summarization?title=ExampleTitle&text=LongTextToSummarize&sector=Science
  ```

- **Translation:**
  ```http
  GET /llms/translation?title=ExampleTitle&text=Hello, how are you?&sector=Languages
  ```

- **Sentiment Analysis:**
  ```http
  GET /nlp/generate_sentiments?title=ExampleTitle&searchString=ExampleSearch&text=The product is amazing!
  ```

#### **Stock Market Trading Endpoints:**

- **Plot Daily Stock Price Equity**
  ```http
  GET /trading/plot_daily_price_equity?ticker=AAPL
  ```

- **Technical Indicators for Equities**
  ```http
  GET /trading/plot_daily_price_equity_technical_indicators?ticker=TSLA
  ```

- **Stock Volume vs. Moving Averages**
  ```http
  GET /trading/plot_stock_volume_vs_moving_average?tickers=AAPL,GOOGL&start_date=2023-01-01&end_date=2023-12-31&filename=report.pdf
  ```

- **Daily Volume Change**
  ```http
  GET /trading/plot_daily_volume_change?tickers=AMZN&start_date=2023-01-01&end_date=2023-12-31&filename=volume_report.pdf
  ```

- **Volume Surges vs. Price**
  ```http
  GET /trading/plot_volume_surges_vs_price?tickers=MSFT&start_date=2023-01-01&end_date=2023-12-31&filename=volume_price.pdf
  ```

- **Relative Strength Index (RSI)**
  ```http
  GET /trading/plot_rsi?tickers=NFLX&start_date=2023-01-01&end_date=2023-12-31&filename=rsi_report.pdf
  ```

- **MACD Analysis**
  ```http
  GET /trading/plot_macd?tickers=GOOG&start_date=2023-01-01&end_date=2023-12-31&filename=macd_report.pdf
  ```

- **Volume Prediction**
  ```http
  GET /trading/plot_volume_prediction?tickers=FB&start_date=2023-01-01&end_date=2023-12-31&filename=prediction_report.pdf
  ```

- **Prophet-Based Volume Prediction for Multiple Stocks**
  ```http
  GET /trading/plot_prophet_volume_prediction_for_multiple_stocks?tickers=AAPL,GOOGL&start_date=2023-01-01&end_date=2023-12-31&filename=prophet_report.pdf
  ```

---

## Project Structure

```
📂 YOUR-REPO/
│-- 📄 app.py  # Main Flask API
│-- 📂 Repo/   # Contains NLP model scripts
│-- 📂 volume_based_features/  # Stock market volume analysis scripts
│-- 📂 pdfs_documents/  # PDF Report generation scripts
│-- 📂 algorithmic_trading/  # Algorithmic trading analysis scripts
│-- 📄 requirements.txt  # List of dependencies
│-- 📄 README.md  # Project documentation
```

---

## Dependencies
- **Flask** (Web API framework)
- **Flask-CORS** (Handle cross-origin requests)
- **pandas** (Data processing)
- **Spacy** (NLP processing)
- **Transformers** (Hugging Face library for NLP)
- **urllib.parse** (To handle URL parameters)
- **Matplotlib & Seaborn** (For data visualization)
- **Prophet** (For time-series forecasting)

Install dependencies using:
```bash
pip install -r requirements.txt
```
