# news/news_analyzer.py
"""
Real-time news analysis system using OpenAI GPT for market sentiment.
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import aiohttp
import feedparser
import yfinance as yf
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import re
import time
from dotenv import load_dotenv

from config.settings import WATCHLIST, DATA_DIR
from data.database import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class NewsItem:
    """Represents a news article."""
    title: str
    summary: str
    url: str
    published: datetime
    source: str
    relevant_symbols: List[str] = None
    sentiment_score: float = 0.0
    impact_analysis: str = ""


@dataclass
class StockImpact:
    """Predicted impact on a stock from news."""
    symbol: str
    impact_score: float  # -1 to +1 (bearish to bullish)
    confidence: float  # 0 to 1
    reasoning: str
    news_items: List[NewsItem]
    analysis_date: datetime


class NewsAnalyzer:
    """Analyzes news using GPT to predict market impacts."""

    def __init__(self, openai_api_key: str):
        """Initialize news analyzer with OpenAI."""
        self.client = OpenAI(api_key=openai_api_key)
        self.db = DatabaseManager()
        self.symbols = WATCHLIST

        # Company name mapping for better matching
        self.company_names = self._build_company_mapping()

        # News sources
        self.rss_feeds = [
            # Financial news
            "https://feeds.finance.yahoo.com/rss/2.0/headline",
            "https://feeds.bloomberg.com/markets/news.rss",
            "https://feeds.reuters.com/reuters/businessNews",
            "https://www.cnbc.com/id/10001147/device/rss/rss.html",
            "https://feeds.marketwatch.com/marketwatch/topstories/",

            # Tech news
            "https://techcrunch.com/feed/",
            "https://www.theverge.com/rss/index.xml",

            # General news that might impact markets
            "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
            "https://feeds.wsj.com/rss/RSSWorldNews.xml",
        ]

    def _build_company_mapping(self) -> Dict[str, str]:
        """Build symbol to company name mapping."""
        mapping = {}

        # Extended mappings for all symbols in your watchlist
        common_names = {
            # Technology
            "AAPL": "Apple",
            "MSFT": "Microsoft",
            "GOOGL": "Google|Alphabet",
            "AMZN": "Amazon",
            "META": "Meta|Facebook",
            "NVDA": "Nvidia",
            "TSLA": "Tesla",
            "AMD": "AMD|Advanced Micro Devices",
            "INTC": "Intel",
            "CRM": "Salesforce",
            "ORCL": "Oracle",
            "ADBE": "Adobe",
            "NFLX": "Netflix",
            "AVGO": "Broadcom",
            "CSCO": "Cisco",
            "QCOM": "Qualcomm",
            "TXN": "Texas Instruments",
            "IBM": "IBM",
            "NOW": "ServiceNow",
            "UBER": "Uber",

            # Financial
            "JPM": "JPMorgan|JP Morgan|Chase",
            "BAC": "Bank of America",
            "WFC": "Wells Fargo",
            "GS": "Goldman Sachs",
            "MS": "Morgan Stanley",
            "C": "Citigroup|Citi",
            "USB": "US Bank|U.S. Bancorp",
            "PNC": "PNC Bank|PNC Financial",
            "AXP": "American Express|Amex",
            "BLK": "BlackRock",
            "SCHW": "Charles Schwab|Schwab",
            "COF": "Capital One",
            "SPGI": "S&P Global",
            "CME": "CME Group",
            "ICE": "Intercontinental Exchange",
            "V": "Visa",
            "MA": "Mastercard",
            "PYPL": "PayPal",
            "SQ": "Square|Block",
            "COIN": "Coinbase",

            # Healthcare
            "JNJ": "Johnson & Johnson|J&J",
            "UNH": "UnitedHealth|United Health",
            "PFE": "Pfizer",
            "ABBV": "AbbVie",
            "TMO": "Thermo Fisher",
            "ABT": "Abbott",
            "CVS": "CVS Health|CVS",
            "MRK": "Merck",
            "DHR": "Danaher",
            "AMGN": "Amgen",
            "GILD": "Gilead",
            "ISRG": "Intuitive Surgical",
            "VRTX": "Vertex",
            "REGN": "Regeneron",
            "ZTS": "Zoetis",
            "BIIB": "Biogen",
            "ILMN": "Illumina",
            "IDXX": "IDEXX",
            "ALGN": "Align Technology",
            "DXCM": "DexCom",

            # Consumer Discretionary
            "HD": "Home Depot",
            "NKE": "Nike",
            "MCD": "McDonald's",
            "SBUX": "Starbucks",
            "TGT": "Target",
            "LOW": "Lowe's",
            "DIS": "Disney",
            "CMCSA": "Comcast",
            "CHTR": "Charter Communications",
            "ROKU": "Roku",
            "F": "Ford",
            "GM": "General Motors|GM",
            "RIVN": "Rivian",
            "LCID": "Lucid|Lucid Motors",
            "CCL": "Carnival",
            "RCL": "Royal Caribbean",
            "WYNN": "Wynn Resorts",
            "MGM": "MGM Resorts",
            "DKNG": "DraftKings",
            "PENN": "Penn Entertainment",

            # Consumer Staples
            "WMT": "Walmart",
            "PG": "Procter & Gamble|P&G",
            "KO": "Coca-Cola|Coke",
            "PEP": "PepsiCo|Pepsi",
            "COST": "Costco",

            # Industrial
            "BA": "Boeing",
            "CAT": "Caterpillar",
            "GE": "General Electric|GE",
            "MMM": "3M",
            "HON": "Honeywell",
            "UPS": "UPS",
            "RTX": "Raytheon",
            "LMT": "Lockheed Martin",
            "NOC": "Northrop Grumman",
            "DE": "Deere|John Deere",
            "EMR": "Emerson Electric",
            "ETN": "Eaton",
            "ITW": "Illinois Tool Works",
            "PH": "Parker Hannifin",
            "GD": "General Dynamics",
            "FDX": "FedEx",
            "NSC": "Norfolk Southern",
            "UNP": "Union Pacific",
            "CSX": "CSX",
            "DAL": "Delta Airlines|Delta",

            # Energy
            "CVX": "Chevron",
            "XOM": "Exxon|ExxonMobil",
            "COP": "ConocoPhillips",
            "SLB": "Schlumberger",

            # Utilities
            "NEE": "NextEra Energy",
            "DUK": "Duke Energy",
            "SO": "Southern Company",
            "D": "Dominion Energy",
            "AEP": "American Electric Power",
            "EXC": "Exelon",
            "XEL": "Xcel Energy",
            "ED": "Consolidated Edison|ConEd",
            "WEC": "WEC Energy",
            "ES": "Eversource",

            # Real Estate
            "AMT": "American Tower",
            "PLD": "Prologis",
            "CCI": "Crown Castle",
            "EQIX": "Equinix",
            "PSA": "Public Storage",
            "O": "Realty Income",
            "WELL": "Welltower",
            "AVB": "AvalonBay",
            "EQR": "Equity Residential",
            "SPG": "Simon Property",

            # Communication Services
            "VZ": "Verizon",
            "T": "AT&T",
            "TMUS": "T-Mobile",
            "SNAP": "Snapchat|Snap",
            "PINS": "Pinterest",
            "TWTR": "Twitter|X",
            "ZM": "Zoom",
            "DOCU": "DocuSign",

            # Other
            "AAL": "American Airlines",
            "UAL": "United Airlines",
            "OKTA": "Okta",
            "GME": "GameStop",

            # Market ETFs
            "SPY": "S&P 500|SPX|market",
            "QQQ": "Nasdaq 100|NASDAQ|tech",
            "DIA": "Dow Jones|Dow",
            "IWM": "Russell 2000|small cap",
            "VTI": "Total Market",
            "XLF": "Financial sector|banks",
            "XLK": "Technology sector|tech",
            "XLE": "Energy sector|oil",
            "XLV": "Healthcare sector|pharma",
            "XLI": "Industrial sector",
            "XLY": "Consumer Discretionary",
            "XLP": "Consumer Staples",
            "XLU": "Utilities sector",
            "XLRE": "Real Estate sector|REIT",
            "XLB": "Materials sector",
            "XLC": "Communication Services",

            # Leveraged ETFs
            "TQQQ": "3x Nasdaq Bull|tech bull|triple Q",
            "SQQQ": "3x Nasdaq Bear|tech bear|short nasdaq",
            "TECL": "3x Tech Bull|technology bull",
            "TECS": "3x Tech Bear|technology bear",
            "SOXL": "3x Semiconductor Bull|chip bull|semiconductor",
            "SOXS": "3x Semiconductor Bear|chip bear|semiconductor",
            "UPRO": "3x S&P 500 Bull|SPY bull",
            "SPXS": "3x S&P 500 Bear|SPY bear",
            "SPXL": "3x S&P 500 Bull|market bull",
            "SPXU": "3x S&P 500 Bear|market bear",
            "FAS": "3x Financial Bull|bank bull",
            "FAZ": "3x Financial Bear|bank bear",
            "ERX": "3x Energy Bull|oil bull",
            "ERY": "3x Energy Bear|oil bear",
            "GUSH": "3x Oil Gas Bull|energy bull",
            "DRIP": "3x Oil Gas Bear|energy bear",
            "DRN": "3x Real Estate Bull|REIT bull",
            "DRV": "3x Real Estate Bear|REIT bear",
            "LABU": "3x Biotech Bull|pharma bull",
            "LABD": "3x Biotech Bear|pharma bear",
            "NUGT": "3x Gold Miners Bull|gold bull",
            "DUST": "3x Gold Miners Bear|gold bear",
            "JNUG": "3x Junior Gold Bull",
            "JDST": "3x Junior Gold Bear",
            "TNA": "3x Russell Bull|small cap bull",
            "TZA": "3x Russell Bear|small cap bear",
            "UVXY": "1.5x VIX|volatility|fear index",
            "SVXY": "-0.5x VIX|inverse volatility",
            "YINN": "3x China Bull|China market",
            "YANG": "3x China Bear|China short",
            "EDC": "3x Emerging Markets Bull",
            "EDZ": "3x Emerging Markets Bear",
        }

        # Initialize mapping with common names
        mapping = common_names.copy()

        # For any symbols not in common_names, try to fetch from yfinance
        logger.info("Building company name mappings...")
        missing_symbols = [s for s in self.symbols if s not in common_names]

        if missing_symbols:
            logger.info(f"Fetching names for {len(missing_symbols)} additional symbols...")
            batch_size = 10
            for i in range(0, len(missing_symbols), batch_size):
                batch = missing_symbols[i:i + batch_size]
                for symbol in batch:
                    if symbol not in mapping:
                        try:
                            ticker = yf.Ticker(symbol)
                            info = ticker.info
                            if 'longName' in info and info['longName']:
                                mapping[symbol] = info['longName']
                            elif 'shortName' in info and info['shortName']:
                                mapping[symbol] = info['shortName']
                            else:
                                mapping[symbol] = symbol
                        except Exception as e:
                            logger.debug(f"Could not fetch name for {symbol}: {e}")
                            mapping[symbol] = symbol
                time.sleep(0.1)  # Rate limiting

        logger.info(f"Company mapping complete. Total symbols: {len(mapping)}")
        return mapping

    def verify_symbol_coverage(self):
        """Verify that all symbols from WATCHLIST are included."""
        missing_symbols = []
        for symbol in self.symbols:
            if symbol not in self.company_names:
                missing_symbols.append(symbol)

        if missing_symbols:
            logger.warning(f"Missing company mappings for {len(missing_symbols)} symbols: {missing_symbols}")
        else:
            logger.info(f"All {len(self.symbols)} symbols have company mappings")

        return len(self.symbols), len(missing_symbols)

    async def fetch_news(self, hours_back: int = 24) -> List[NewsItem]:
        """Fetch recent news from multiple sources."""
        all_news = []
        cutoff_time = datetime.now() - timedelta(hours=hours_back)

        # Fetch from RSS feeds
        for feed_url in self.rss_feeds:
            try:
                feed = feedparser.parse(feed_url)
                source = feed.feed.get('title', 'Unknown')

                for entry in feed.entries[:20]:  # Limit per source
                    # Parse publication date
                    published = datetime.now()
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        try:
                            # Convert time.struct_time to datetime
                            published = datetime.fromtimestamp(
                                time.mktime(entry.published_parsed)
                            )
                        except Exception:
                            # Fallback to current time if parsing fails
                            published = datetime.now()

                    if published < cutoff_time:
                        continue

                    news_item = NewsItem(
                        title=entry.get('title', ''),
                        summary=entry.get('summary', '')[:500],  # Limit length
                        url=entry.get('link', ''),
                        published=published,
                        source=source
                    )
                    all_news.append(news_item)

            except Exception as e:
                logger.error(f"Error fetching from {feed_url}: {e}")

        # Remove duplicates by title
        seen_titles = set()
        unique_news = []
        for item in all_news:
            if item.title not in seen_titles:
                seen_titles.add(item.title)
                unique_news.append(item)

        logger.info(f"Fetched {len(unique_news)} unique news items")
        return sorted(unique_news, key=lambda x: x.published, reverse=True)

    def _extract_relevant_symbols(self, text: str) -> List[str]:
        """Extract stock symbols mentioned in text."""
        relevant_symbols = []
        text_lower = text.lower()

        for symbol, names in self.company_names.items():
            # Check symbol directly (with word boundaries)
            if re.search(r'\b' + symbol + r'\b', text, re.IGNORECASE):
                relevant_symbols.append(symbol)
                continue

            # Check company names
            for name in names.split('|'):
                if name.lower() in text_lower:
                    relevant_symbols.append(symbol)
                    break

        return list(set(relevant_symbols))

    async def analyze_news_with_gpt(self, news_items: List[NewsItem]) -> List[StockImpact]:
        """Use GPT to analyze news impact on stocks."""

        # Group news by relevance to symbols
        symbol_news = defaultdict(list)

        # First pass: identify relevant symbols
        for item in news_items:
            item.relevant_symbols = self._extract_relevant_symbols(
                item.title + " " + item.summary
            )
            for symbol in item.relevant_symbols:
                symbol_news[symbol].append(item)

        # Also analyze general market news
        market_news = [item for item in news_items if not item.relevant_symbols]

        # Analyze impact for each symbol with relevant news
        impacts = []

        # Process in batches to avoid overwhelming the API
        symbols_to_analyze = list(symbol_news.keys())
        batch_size = 10

        for i in range(0, len(symbols_to_analyze), batch_size):
            batch = symbols_to_analyze[i:i + batch_size]

            for symbol in batch:
                news_list = symbol_news[symbol]
                if not news_list:
                    continue

                # Prepare news summary for GPT
                news_summary = "\n".join([
                    f"- {item.title}: {item.summary[:200]}..."
                    for item in news_list[:10]  # Limit to 10 most recent
                ])

                try:
                    # Call GPT for analysis with retry logic
                    for attempt in range(3):
                        try:
                            response = self.client.chat.completions.create(
                                model="gpt-4o-mini",  # Using 3.5 for better JSON compliance
                                messages=[
                                    {
                                        "role": "system",
                                        "content": """You are a financial analyst expert. Analyze news and predict stock impact.
                                        IMPORTANT: Respond with ONLY valid JSON, no other text before or after."""
                                    },
                                    {
                                        "role": "user",
                                        "content": f"""
                                        Stock: {symbol} ({self.company_names.get(symbol, symbol)})

                                        Recent news:
                                        {news_summary}

                                        Analyze impact on {symbol} stock price (1-5 days).

                                        Return ONLY this JSON:
                                        {{
                                            "impact_score": 0.0,
                                            "confidence": 0.5,
                                            "reasoning": "Brief explanation",
                                            "key_factors": ["factor1", "factor2"]
                                        }}

                                        impact_score: -1.0 to 1.0 (-1=very bearish, 0=neutral, 1=very bullish)
                                        confidence: 0.0 to 1.0
                                        """
                                    }
                                ],
                                temperature=0.3,
                                max_tokens=300,
                                response_format={"type": "json_object"}  # Force JSON response
                            )

                            # Get response content
                            response_text = response.choices[0].message.content.strip()

                            # Parse JSON
                            analysis = json.loads(response_text)
                            break  # Success, exit retry loop

                        except json.JSONDecodeError as e:
                            if attempt == 2:  # Last attempt
                                logger.error(f"Failed to parse JSON for {symbol} after 3 attempts: {e}")
                                logger.error(f"Response was: {response_text[:200]}")
                                analysis = {
                                    "impact_score": 0.0,
                                    "confidence": 0.3,
                                    "reasoning": "Unable to analyze due to parsing error",
                                    "key_factors": ["technical_error"]
                                }
                            else:
                                await asyncio.sleep(1)  # Wait before retry

                    # Validate and sanitize the response
                    impact_score = float(analysis.get('impact_score', 0.0))
                    impact_score = max(-1.0, min(1.0, impact_score))  # Clamp to [-1, 1]

                    confidence = float(analysis.get('confidence', 0.5))
                    confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]

                    reasoning = str(analysis.get('reasoning', 'No reasoning provided'))[:500]

                    impact = StockImpact(
                        symbol=symbol,
                        impact_score=impact_score,
                        confidence=confidence,
                        reasoning=reasoning,
                        news_items=news_list,
                        analysis_date=datetime.now()
                    )
                    impacts.append(impact)

                    logger.info(f"Analyzed {symbol}: score={impact.impact_score:.2f}, conf={impact.confidence:.2f}")

                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")

            # Rate limiting between batches
            if i + batch_size < len(symbols_to_analyze):
                await asyncio.sleep(2)

        # Analyze general market sentiment
        if market_news:
            try:
                market_impact = await self._analyze_market_sentiment(market_news)
                impacts.extend(market_impact)
            except Exception as e:
                logger.error(f"Error in market sentiment analysis: {e}")

        return impacts

    async def _analyze_market_sentiment(self, news_items: List[NewsItem]) -> List[StockImpact]:
        """Analyze general market news for broad impact."""

        news_summary = "\n".join([
            f"- {item.title}"
            for item in news_items[:20]
        ])

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a market analyst. Analyze news for market impact.
                        IMPORTANT: Respond with ONLY valid JSON."""
                    },
                    {
                        "role": "user",
                        "content": f"""
                        Recent market news:
                        {news_summary}

                        Analyze market impact by sector.

                        Return ONLY this JSON:
                        {{
                            "market_sentiment": 0.0,
                            "affected_sectors": {{
                                "tech": 0.0,
                                "financial": 0.0,
                                "energy": 0.0,
                                "consumer": 0.0
                            }},
                            "reasoning": "Brief explanation"
                        }}

                        All values: -1.0 to 1.0
                        """
                    }
                ],
                temperature=0.3,
                max_tokens=300,
                response_format={"type": "json_object"}
            )

            analysis = json.loads(response.choices[0].message.content)
            impacts = []

            # Map sectors to relevant ETFs and stocks
            sector_mapping = {
                "tech": ["QQQ", "TQQQ", "SQQQ", "XLK", "AAPL", "MSFT", "NVDA", "GOOGL", "META", "TECL", "TECS"],
                "financial": ["XLF", "FAS", "FAZ", "JPM", "BAC", "GS", "MS", "C", "WFC"],
                "energy": ["XLE", "ERX", "ERY", "XOM", "CVX", "COP", "SLB", "GUSH", "DRIP"],
                "consumer": ["XLY", "XLP", "AMZN", "WMT", "HD", "TGT", "MCD", "NKE", "SBUX"]
            }

            # Create impacts for affected sectors
            affected_sectors = analysis.get('affected_sectors', {})
            for sector, impact_score in affected_sectors.items():
                if abs(impact_score) > 0.1:  # Significant impact
                    for symbol in sector_mapping.get(sector, []):
                        if symbol in self.symbols:
                            impacts.append(StockImpact(
                                symbol=symbol,
                                impact_score=impact_score * 0.7,  # Reduce for indirect impact
                                confidence=0.5,  # Lower confidence for broad impacts
                                reasoning=f"Market sentiment: {analysis.get('reasoning', 'Market impact')}",
                                news_items=news_items[:5],
                                analysis_date=datetime.now()
                            ))

            return impacts

        except Exception as e:
            logger.error(f"Error analyzing market sentiment: {e}")
            return []

    def save_predictions(self, impacts: List[StockImpact], filename: str = None):
        """Save news-based predictions to CSV."""
        if not impacts:
            logger.warning("No impacts to save")
            return None

        if filename is None:
            filename = f"news_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        # Convert to DataFrame
        data = []
        for impact in impacts:
            data.append({
                'symbol': impact.symbol,
                'impact_score': impact.impact_score,
                'confidence': impact.confidence,
                'predicted_direction': 'LONG' if impact.impact_score > 0 else 'SHORT',
                'predicted_return': impact.impact_score * 0.05,  # Convert to approx return
                'reasoning': impact.reasoning[:200],  # Truncate
                'news_count': len(impact.news_items),
                'analysis_date': impact.analysis_date,
                'key_headlines': '; '.join([n.title for n in impact.news_items[:3]])
            })

        df = pd.DataFrame(data)
        df = df.sort_values('confidence', ascending=False)

        # Save to CSV
        filepath = DATA_DIR / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} news predictions to {filepath}")

        # Also save detailed analysis
        detailed_file = filepath.with_suffix('.json')
        detailed_data = []
        for impact in impacts:
            detailed_data.append({
                'symbol': impact.symbol,
                'impact_score': impact.impact_score,
                'confidence': impact.confidence,
                'reasoning': impact.reasoning,
                'analysis_date': impact.analysis_date.isoformat(),
                'news_items': [
                    {
                        'title': n.title,
                        'summary': n.summary,
                        'url': n.url,
                        'published': n.published.isoformat(),
                        'source': n.source
                    }
                    for n in impact.news_items
                ]
            })

        with open(detailed_file, 'w') as f:
            json.dump(detailed_data, f, indent=2)

        return df

    async def run_analysis(self, hours_back: int = 24):
        """Run complete news analysis pipeline."""
        logger.info("Starting news analysis...")

        # Verify symbol coverage
        total_symbols, missing = self.verify_symbol_coverage()
        logger.info(f"Analyzing news for {total_symbols} symbols from WATCHLIST")

        # Fetch recent news
        news_items = await self.fetch_news(hours_back)

        if not news_items:
            logger.warning("No recent news found")
            return None

        logger.info(f"Analyzing {len(news_items)} news items...")

        # Analyze with GPT
        impacts = await self.analyze_news_with_gpt(news_items)

        # Save predictions
        df = self.save_predictions(impacts)

        # Print summary
        if df is not None and not df.empty:
            print("\n=== NEWS ANALYSIS SUMMARY ===")
            print(f"Total symbols analyzed: {total_symbols}")
            print(f"News items processed: {len(news_items)}")
            print(f"Total predictions generated: {len(df)}")
            print(f"Bullish signals: {len(df[df['impact_score'] > 0])}")
            print(f"Bearish signals: {len(df[df['impact_score'] < 0])}")
            print(f"High confidence (>0.7): {len(df[df['confidence'] > 0.7])}")

            # Show symbol coverage
            analyzed_symbols = df['symbol'].unique()
            print(f"Symbols with news impact: {len(analyzed_symbols)}")

            print("\nTop 10 News-Based Predictions:")
            display_cols = ['symbol', 'impact_score', 'confidence', 'predicted_direction', 'reasoning']
            print(df.head(10)[display_cols].to_string(index=False))

        return df


# Integration with live trading system
class NewsTrader:
    """Integrates news analysis with live trading."""

    def __init__(self, analyzer: NewsAnalyzer, trading_system):
        self.analyzer = analyzer
        self.trading_system = trading_system
        self.last_analysis = None

    async def monitor_news_loop(self, interval_minutes: int = 30):
        """Continuously monitor news during market hours."""
        while True:
            try:
                # Check if market is open
                market_open = self._is_market_open()

                if market_open:
                    logger.info("Running news analysis...")

                    # Analyze recent news
                    df = await self.analyzer.run_analysis(hours_back=2)

                    if df is not None and not df.empty:
                        # Filter high confidence signals
                        high_conf = df[df['confidence'] > 0.6]

                        # Integrate with trading system
                        for _, row in high_conf.iterrows():
                            # Add to trading signals
                            self._add_news_signal(row)

                    self.last_analysis = datetime.now()
                else:
                    logger.info("Market closed, skipping news analysis")

                # Wait for next cycle
                await asyncio.sleep(interval_minutes * 60)

            except Exception as e:
                logger.error(f"Error in news monitoring: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error

    def _is_market_open(self) -> bool:
        """Check if US market is open."""
        now = datetime.now()

        # Simple check - enhance with holiday calendar
        if now.weekday() >= 5:  # Weekend
            return False

        market_open = now.time() >= pd.Timestamp("09:30").time()
        market_close = now.time() <= pd.Timestamp("16:00").time()

        return market_open and market_close

    def _add_news_signal(self, news_signal):
        """Add news-based signal to trading system."""
        logger.info(f"News signal: {news_signal['symbol']} "
                    f"score={news_signal['impact_score']:.2f} "
                    f"conf={news_signal['confidence']:.2f}")


# Standalone script to run news analysis
async def main():
    """Run news analysis standalone."""
    # Load environment variables
    load_dotenv()

    # Get OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")

    # Check if it's still the placeholder
    if not openai_key or openai_key == "your-openai-api-key-here":
        print("ERROR: Please set your actual OpenAI API key!")
        print("Either:")
        print("1. Add to .env file: OPENAI_API_KEY=sk-proj-...")
        print("2. Set environment variable: export OPENAI_API_KEY=sk-proj-...")
        return

    # Create analyzer
    analyzer = NewsAnalyzer(openai_key)

    # Run analysis
    await analyzer.run_analysis(hours_back=24)


if __name__ == "__main__":
    asyncio.run(main())