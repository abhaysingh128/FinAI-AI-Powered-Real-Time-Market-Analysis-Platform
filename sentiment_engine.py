from GoogleNews import GoogleNews
from transformers import pipeline
import pandas as pd
from datetime import datetime
import yfinance as yf

class SentimentEngine:
    def __init__(self):
        # Load FinBERT pipeline
        model_path = "ProsusAI/finbert"
        local_path = "./saved_models/finbert_finai"
        
        # Check if local model exists and has config.json to prevent pipeline crash
        import os
        if os.path.exists(local_path) and "config.json" in os.listdir(local_path):
            print(f"[SentimentEngine] Loading local model from {local_path}...")
            model_path = local_path
        else:
            print(f"[SentimentEngine] Local model not found. Using default {model_path}...")

        try:
            # top_k=None forces all scores to be returned
            self.pipe = pipeline("text-classification", model=model_path, top_k=None)
        except Exception as e:
            print(f"Error loading FinBERT: {e}")
            self.pipe = None
    
    # ... (skipping to analyze_sentiment)

    def analyze_sentiment(self, news_list):
        # ...
        for news in news_list:
            try:
                # Predict
                # Pipeline with top_k=None returns [[{'label': 'A', 'score': X}, ...]] for single text
                raw_out = self.pipe(news['title'])
                
                # Robust extraction
                if isinstance(raw_out, list) and len(raw_out) > 0:
                    if isinstance(raw_out[0], list):
                        # [[{...}, {...}]]
                        preds = raw_out[0]
                    elif isinstance(raw_out[0], dict):
                         # [{...}, {...}] - could be multi-label or flattened batch
                         preds = raw_out
                    else:
                        preds = []
                else:
                    preds = []

                scores = {p['label']: p['score'] for p in preds if isinstance(p, dict)}
                
                # FinBERT labels: 'positive', 'negative', 'neutral'
                compound = 0
                label = 'Neutral'
                
                # If keys are missing (capitalization?), normalize
                # Common labels: 'positive', 'negative', 'neutral' OR 'Positive', 'Negative', 'Neutral'
                # ProsusAI/finbert uses lowercase 'positive', 'negative', 'neutral'.
                # Let's handle generic case
                
                s_pos = scores.get('positive', scores.get('Positive', 0))
                s_neg = scores.get('negative', scores.get('Negative', 0))
                s_neu = scores.get('neutral', scores.get('Neutral', 0))
                
                # If Neutral is dominant (>0.85), keep it 0.
                if s_neu > 0.85:
                    compound = 0
                    stats['neutral'] += 1
                    label = 'Neutral'
                else:
                    # Amplify the difference
                    compound = s_pos - s_neg
                    
                    if compound > 0.05:
                        stats['positive'] += 1
                        label = 'Positive'
                    elif compound < -0.05:
                        stats['negative'] += 1
                        label = 'Negative'
                    else:
                        stats['neutral'] += 1
                        label = 'Neutral'
                
                news['sentiment_score'] = compound
                news['sentiment_label'] = label
                total_score += compound
                count += 1
                
            except Exception as e:
                print(f"[SentimentEngine] Error analyzing item '{news.get('title', 'N/A')}': {e}")
                news['sentiment_score'] = 0
                news['sentiment_label'] = 'Neutral'
                stats['neutral'] += 1

    def fetch_news(self, ticker):
        """Fetch news for the ticker using yfinance (primary) and GoogleNews (fallback)."""
        news_list = []
        
        print(f"[SentimentEngine] Fetching news for {ticker}...")

        def normalize_yf_news(raw_news):
            normalized = []
            for item in raw_news:
                # Handle nested structure
                title = item.get('title')
                link = item.get('link')
                pub_time = item.get('providerPublishTime')
                
                if not title and 'content' in item:
                    content = item['content']
                    title = content.get('title')
                    link = content.get('canonicalUrl', {}).get('url') or content.get('clickThroughUrl', {}).get('url')
                    pub_time = content.get('pubDate')
                
                if title:
                    date_str = "Recent"
                    if pub_time:
                        try:
                            if isinstance(pub_time, (int, float)):
                                 date_str = datetime.fromtimestamp(pub_time).strftime('%Y-%m-%d')
                            else:
                                 date_str = str(pub_time)[:10]
                        except:
                            pass
                    normalized.append({
                        'title': title,
                        'link': link if link else '#',
                        'published': date_str
                    })
            return normalized

        # 1. Try yfinance with exact ticker
        try:
            ticker_obj = yf.Ticker(ticker)
            yf_news = ticker_obj.news
            if yf_news:
                news_list = normalize_yf_news(yf_news[:5])
                print(f"[SentimentEngine] Found {len(news_list)} news items via yfinance ({ticker})")
        except Exception as e:
            print(f"[SentimentEngine] YFinance ({ticker}) failed: {e}")

        # 1.5. Try yfinance with base ticker (e.g. INFY instead of INFY.NS)
        # News for the global listing is often relevant for the local listing
        if not news_list and "." in ticker:
            base_ticker = ticker.split(".")[0]
            print(f"[SentimentEngine] Retrying yfinance with base ticker: {base_ticker}")
            try:
                base_obj = yf.Ticker(base_ticker)
                yf_news_base = base_obj.news
                if yf_news_base:
                    news_list = normalize_yf_news(yf_news_base[:5])
                    print(f"[SentimentEngine] Found {len(news_list)} news items via yfinance ({base_ticker})")
            except Exception as e:
                print(f"[SentimentEngine] YFinance ({base_ticker}) failed: {e}")

        # 2. Fallback to GoogleNews if empty
        if not news_list:
            print("[SentimentEngine] Falling back to GoogleNews...")
            try:
                try:
                    name = yf.Ticker(ticker).info.get('shortName', ticker)
                except:
                    name = ticker 
                
                search_term = f"{name} financial news"
                print(f"[SentimentEngine] Searching GoogleNews for: {search_term}")
                
                googlenews = GoogleNews(period='7d')
                googlenews.search(search_term)
                results = googlenews.result()
                
                if results:
                    print(f"[SentimentEngine] GoogleNews found {len(results)} results")
                    for item in results[:5]:
                        title = item.get('title', '')
                        if title: # Skip empty titles
                            news_list.append({
                                'title': title,
                                'link': item.get('link', '#'),
                                'published': item.get('date', 'Recent')
                            })
                else:
                    print("[SentimentEngine] GoogleNews found no results.")
                    
            except Exception as e:
                print(f"[SentimentEngine] GoogleNews failed: {e}")

        return news_list

    def analyze_sentiment(self, news_list):
        """Analyze sentiment of news headlines."""
        if not self.pipe:
            return 0, {'positive': 0, 'negative': 0, 'neutral': 0}, news_list 
            
        total_score = 0
        count = 0
        
        stats = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for news in news_list:
            try:
                # Predict
                preds = self.pipe(news['title'])[0]
                scores = {p['label']: p['score'] for p in preds}
                
                # FinBERT labels: 'positive', 'negative', 'neutral'
                compound = 0
                label = 'Neutral'
                
                # If Neutral is dominant (>0.85), keep it 0.
                if scores.get('neutral', 0) > 0.85:
                    compound = 0
                    stats['neutral'] += 1
                    label = 'Neutral'
                else:
                    # Amplify the difference for non-neutral news
                    compound = scores.get('positive', 0) - scores.get('negative', 0)
                    
                    if compound > 0.05:
                        stats['positive'] += 1
                        label = 'Positive'
                    elif compound < -0.05:
                        stats['negative'] += 1
                        label = 'Negative'
                    else:
                        stats['neutral'] += 1
                        label = 'Neutral'
                
                news['sentiment_score'] = compound
                news['sentiment_label'] = label
                total_score += compound
                count += 1
                # print(f"DEBUG: Analyzed '{news['title']}' -> {label} ({compound})")
            except Exception as e:
                print(f"[SentimentEngine] Error analyzing item '{news.get('title', 'N/A')}': {e}")
                news['sentiment_score'] = 0
                news['sentiment_label'] = 'Neutral'
                stats['neutral'] += 1
                
        avg_score = total_score / count if count > 0 else 0
        print(f"[SentimentEngine] Analyzed {count} items. Avg Score: {avg_score:.2f}")
        print(f"[SentimentEngine] Stats: Positive={stats['positive']}, Negative={stats['negative']}, Neutral={stats['neutral']}")
        return avg_score, stats, news_list

    def run(self, ticker):
        """Execute full sentiment pipeline."""
        news = self.fetch_news(ticker)
        if not news:
            return 0, {'positive': 0, 'negative': 0, 'neutral': 0}, []
        
        score, stats, analyzed_news = self.analyze_sentiment(news)
        return score, stats, analyzed_news
