# TASE Stock AI Agent

An intelligent stock recommendation system for the Israeli Stock Exchange (Tel Aviv Stock Exchange - TASE), powered by Google Gemini AI.

## Features

- **Multi-Factor Analysis**: Combines technical, fundamental, and sentiment analysis
- **Real-Time Data**: Uses TASE DataWise API and Yahoo Finance for live market data
- **AI-Powered Recommendations**: Google Gemini provides intelligent reasoning for buy/sell decisions
- **Swing Trade Focus**: Optimized for 1-week to 1-month holding periods
- **Rule-Based Scoring**: Transparent scoring system with configurable weights

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/stcks-ai-agent.git
cd stcks-ai-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

Create a `.env` file with the following variables:

```env
GEMINI_API_KEY=your_gemini_api_key_here
TASE_API_KEY=your_tase_api_key_here
```

### Getting API Keys

1. **Gemini API Key**: Get it free from [Google AI Studio](https://aistudio.google.com/apikey)
2. **TASE API Key**: Register at [TASE DataHub](https://www.tase.co.il/en/content/products_lobby/datahub)

## Usage

### Scan All Stocks

Scan all TASE stocks and get top recommendations:

```bash
python -m src.main scan
```

### Analyze Specific Stock

Get detailed analysis for a specific stock:

```bash
python -m src.main analyze TEVA.TA
```

### Portfolio Suggestions

Get diversified portfolio recommendations:

```bash
python -m src.main portfolio --budget 50000 --risk medium
```

### Real-Time Monitoring

Monitor stocks in real-time:

```bash
python -m src.main watch --interval 5m
```

## Analysis Components

### Technical Analysis (40% weight)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- SMA/EMA crossovers (20, 50, 200 day)
- Volume patterns
- Bullish/Bearish divergences

### Fundamental Analysis (35% weight)
- P/E ratio vs sector average
- Earnings growth (QoQ)
- Trading volume/liquidity
- TA-35/TA-125 index membership

### Sentiment Analysis (25% weight)
- News sentiment from Globes and Calcalist
- Sector momentum
- Market sentiment indicators

## Scoring System

Each stock receives a score from 0-100:
- **80-100**: Strong Buy
- **60-79**: Buy
- **40-59**: Hold
- **20-39**: Sell
- **0-19**: Strong Sell

## Market Hours

TASE trading hours: Sunday-Thursday, 09:00-17:30 IST

## License

MIT License
