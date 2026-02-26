"""LLM Sentiment Analysis for Market News"""

import numpy as np
from typing import List, Dict, Optional
import pickle
import os
import re


class LLMSentiment:
    """Sentiment analysis using LLM for market news and social media."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        sentiment_threshold: float = 0.6
    ):
        self.model_name = model_name
        self.sentiment_threshold = sentiment_threshold
        self.model = None
        self.tokenizer = None
        self.is_loaded = False

    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text."""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def _load_model(self) -> None:
        """Load the sentiment analysis model."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.is_loaded = True
            print(f"Loaded sentiment model: {self.model_name}")
        except ImportError:
            print("Transformers not installed. Using rule-based sentiment.")
            self.is_loaded = False
        except Exception as e:
            print(f"Error loading model: {e}")
            self.is_loaded = False

    def _rule_based_sentiment(self, text: str) -> float:
        """Rule-based sentiment as fallback."""
        positive_words = [
            'bullish', 'buy', 'long', 'up', 'gain', 'profit', 'growth', 'positive',
            'rise', 'rally', 'surge', 'breakout', 'support', 'strong', 'upgrade',
            'outperform', 'overweight', 'optimistic', 'happy', 'excited'
        ]
        negative_words = [
            'bearish', 'sell', 'short', 'down', 'loss', 'decline', 'negative',
            'fall', 'drop', 'crash', 'breakdown', 'resistance', 'weak', 'downgrade',
            'underperform', 'overweight', 'pessimistic', 'worried', 'fear'
        ]

        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        if pos_count + neg_count == 0:
            return 0.0

        return (pos_count - neg_count) / (pos_count + neg_count)

    def analyze(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of a single text."""
        if not self.is_loaded:
            self._load_model()

        cleaned_text = self._clean_text(text)

        if self.is_loaded and self.model is not None:
            try:
                import torch

                inputs = self.tokenizer(cleaned_text, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = self.model(**inputs)

                probs = torch.softmax(outputs.logits, dim=1)
                # probs[0] = negative, probs[1] = positive
                sentiment_score = probs[0][1].item() - probs[0][0].item()

                return {
                    'sentiment': sentiment_score,
                    'positive_prob': probs[0][1].item(),
                    'negative_prob': probs[0][0].item()
                }
            except Exception as e:
                print(f"Error in sentiment analysis: {e}")
                sentiment_score = self._rule_based_sentiment(cleaned_text)
                return {
                    'sentiment': sentiment_score,
                    'positive_prob': max(0, sentiment_score),
                    'negative_prob': max(0, -sentiment_score)
                }
        else:
            sentiment_score = self._rule_based_sentiment(cleaned_text)
            return {
                'sentiment': sentiment_score,
                'positive_prob': max(0, sentiment_score),
                'negative_prob': max(0, -sentiment_score)
            }

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Analyze sentiment of multiple texts."""
        return [self.analyze(text) for text in texts]

    def aggregate_sentiment(self, texts: List[str]) -> Dict[str, float]:
        """Aggregate sentiment from multiple texts."""
        if not texts:
            return {'sentiment': 0.0, 'positive_prob': 0.5, 'negative_prob': 0.5, 'count': 0}

        results = self.analyze_batch(texts)

        avg_sentiment = np.mean([r['sentiment'] for r in results])
        avg_positive = np.mean([r['positive_prob'] for r in results])
        avg_negative = np.mean([r['negative_prob'] for r in results])

        # Count bullish vs bearish signals
        bullish_signals = sum(1 for r in results if r['sentiment'] > self.sentiment_threshold)
        bearish_signals = sum(1 for r in results if r['sentiment'] < -self.sentiment_threshold)

        return {
            'sentiment': avg_sentiment,
            'positive_prob': avg_positive,
            'negative_prob': avg_negative,
            'count': len(texts),
            'bullish_signals': bullish_signals,
            'bearish_signals': bearish_signals
        }

    def get_market_sentiment(
        self,
        news: List[str],
        social_sentiment: Optional[float] = None
    ) -> Dict[str, any]:
        """Get overall market sentiment from news and social media."""
        news_sentiment = self.aggregate_sentiment(news) if news else {
            'sentiment': 0.0, 'positive_prob': 0.5, 'negative_prob': 0.5
        }

        # Combine with social sentiment if available
        if social_sentiment is not None:
            combined_sentiment = (news_sentiment['sentiment'] + social_sentiment) / 2
        else:
            combined_sentiment = news_sentiment['sentiment']

        # Determine overall signal
        if combined_sentiment > self.sentiment_threshold:
            signal = 'BULLISH'
        elif combined_sentiment < -self.sentiment_threshold:
            signal = 'BEARISH'
        else:
            signal = 'NEUTRAL'

        return {
            'signal': signal,
            'sentiment': combined_sentiment,
            'news_sentiment': news_sentiment,
            'social_sentiment': social_sentiment,
            'confidence': abs(combined_sentiment)
        }

    def save(self, path: str) -> None:
        """Save model configuration."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model_name': self.model_name,
                'sentiment_threshold': self.sentiment_threshold,
                'is_loaded': self.is_loaded
            }, f)

    def load(self, path: str) -> None:
        """Load model configuration."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model_name = data['model_name']
            self.sentiment_threshold = data['sentiment_threshold']
            self.is_loaded = data.get('is_loaded', False)
