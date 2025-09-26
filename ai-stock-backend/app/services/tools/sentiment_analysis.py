import time
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from app.models.schemas import ToolResult
from app.services.tools.multi_source_data import multi_source_tool
from app.core.config import settings
from app.services.tools.technical_analysis import technical_analysis_tool

class SentimentAnalysisTool:
    """
    Comprehensive market sentiment analysis tool for investment decision making.
    Analyzes sentiment from news, social media, analyst ratings, and market indicators.
    """

    def __init__(self):
        self.name = "sentiment_analysis"
        self.cache = {}
        self.cache_ttl = 1800  #30 minutes cache (sentiment changes frequently)

        #Sentiment analysis categories and their weights
        self.analysis_categories = {
            'news_sentiment': 0.30, #Financial news sentiment
            'social_sentiment': 0.25, #Social media sentiment
            'analyst_sentiment': 0.20, #Analyst sentiment and recommendations
            'market_sentiment': 0.25 #Market-based sentiment indicators
        }

        #Sentiment keywords for basic text analysis
        self.positive_keywords = {
            'strong': 2.0, 'excellent': 2.0, 'outstanding': 2.0, 'bullish': 2.0,
            'upgrade': 1.8, 'beat': 1.8, 'exceeded': 1.8, 'growth': 1.5,
            'positive': 1.5, 'good': 1.2, 'buy': 1.8, 'strong buy': 2.0,
            'outperform': 1.6, 'overweight': 1.4, 'momentum': 1.3,
            'breakthrough': 1.7, 'surge': 1.6, 'rally': 1.5, 'gains': 1.3
        }

        self.negative_keywords = {
            'weak': -2.0, 'poor': -2.0, 'terrible': -2.0, 'bearish': -2.0,
            'downgrade': -1.8, 'miss': -1.8, 'disappointed': -1.8, 'decline': -1.5,
            'negative': -1.5, 'bad': -1.2, 'sell': -1.8, 'strong sell': -2.0,
            'underperform': -1.6, 'underweight': -1.4, 'concern': -1.3,
            'crisis': -1.9, 'crash': -1.8, 'plunge': -1.6, 'losses': -1.3
        }

    def _get_comprehensive_sentiment_data(self, ticker: str) -> Dict[str, Any]:
        """Get comprehensive sentiment data"""
        print(f"🔍 Fetching sentiment data for analysis: {ticker}")
        
        sentiment_data = {
            'ticker': ticker.upper(),
            'collection_timestamp': datetime.now().isoformat(),
        }

        try:
            #Get basic stock data
            basic_result = multi_source_tool.get_stock_data(ticker)
            if basic_result.success:
                #Exctract what is needed for sentiment analysis
                market_data = basic_result.data
                sentiment_data['market_data'] = {
                    'current_price': market_data.get('current_price'),
                    '1_day_change': market_data.get('performance', {}).get('1_day_change'),
                    'volume_ratio': market_data.get('volume', {}).get('ratio', 1.0),
                    'company_name': market_data.get('company_name'),
                    'sector': market_data.get('sector')
                }
                print(f"✅ Minimal market context collected")

            #External sentiment sources
            reddit_sentiment = self._get_reddit_sentiment(ticker)
            if reddit_sentiment:
                sentiment_data['reddit_sentiment'] = reddit_sentiment
                print(f"✅ Reddit sentiment collected")

            
            company_name = market_data.get('company_name')

            news_sentiment = self._get_news_sentiment(ticker, company_name)
            if news_sentiment:
                sentiment_data['news_sentiment'] = news_sentiment
                print(f"✅ News sentiment collected")

            twitter_sentiment = self._get_twitter_sentiment(ticker, company_name)
            if twitter_sentiment:
                sentiment_data['twitter_sentiment'] = twitter_sentiment
                print(f"✅ Twitter sentiment collected")
            
            print(f"📊 Sentiment analysis focused on external sources")
        
        except Exception as e:
            print(f"⚠️ Warning: Limited sentiment data available: {str(e)}")

        return sentiment_data

    def _get_reddit_sentiment(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get sentiment indcators from Reddit using public API"""
        try:
            #Reddit reuires a user agent
            headers = {
                'User-Agent': 'AIStockSentiment/1.0 (Stock Analysis Tool)'
            }

            #Targer investing subreddits
            subreddits = ['investing', 'stocks', 'SecurityAnalysis', 'ValueInvesting', 'StockMarket', 'DayTrading']

            sentiment_indicators = {
                'reddit_mentions': 0,
                'positive_mentions': 0,
                'negative_mentions': 0,
                'neutral_mentions': 0,
                'sentiment_score': 0.5,
                'confidence': 0.1,
                'posts_analyzed': [],
                'subreddits_checked': subreddits,
                'total_upvotes': 0,
                'avg_sentiment_per_subreddit': {}
            }

            for subreddit in subreddits:
                try:
                    #Reddit public JSON API endpoint
                    url = f'https://www.reddit.com/r/{subreddit}/search.json'
                    params = {
                        'q': f'"{ticker}" OR "${ticker}" OR "{ticker.lower()}"',
                        'sort': 'new',
                        'limit': 25,
                        't': 'week', #Recent 1 week
                        'type': 'link'
                    }

                    print(f"🔍 Searching r/{subreddit} for {ticker} mentions...")
                    response = requests.get(url, headers=headers, params=params, timeout=15)

                    if response.status_code == 200:
                        data = response.json()
                        posts = data.get('data', {}).get('children', [])

                        subreddit_mentions = 0
                        subreddit_sentiment_total = 0

                        for post in posts:
                            post_data = post.get('data', {})
                            title = post_data.get('title', '').lower()
                            selftext = post_data.get('selftext', '').lower()
                            combined_text = f"{title} {selftext}"

                            #Check if ticker is actually mentioned
                            ticker_patterns = [
                                ticker.lower(),
                                f"${ticker.lower()}",
                                f" {ticker.lower()} ",
                                f"({ticker.lower()})"
                            ]

                            if any(pattern in combined_text for pattern in ticker_patterns):
                                sentiment_indicators['reddit_mentions'] += 1
                                subreddit_mentions += 1

                                #Analyze sentiment of the post using keywords
                                post_sentiment = self._analyze_text_sentiment(combined_text)
                                subreddit_sentiment_total += post_sentiment

                                #Get post metrics
                                post_score = post_data.get('score', 0)
                                num_comments = post_data.get('num_comments', 0)
                                sentiment_indicators['total_upvotes'] += max(0, post_score)

                                #Categorize sentiment
                                if post_sentiment > 0.6:
                                    sentiment_indicators['positive_mentions'] += 1
                                    sentiment_category = 'positive'
                                elif post_sentiment < 0.4:
                                    sentiment_indicators['negative_mentions'] += 1
                                    sentiment_category = 'negative'
                                else:
                                    sentiment_indicators['neutral_mentions'] += 1
                                    sentiment_category = 'neutral'

                            #Store post info for detailed analysis
                            sentiment_indicators['posts_analyzed'].append({
                                'subreddit': subreddit,
                                'title': post_data.get('title', '')[:100], #First 100 chars
                                'score': post_score,
                                'num_comments': num_comments,
                                'sentiment': sentiment_category,
                                'sentiment_score': post_sentiment,
                                'created_utc': post_data.get('created_utc', 0),
                                'url': f"https://www.reddit.com{post_data.get('permalink', '')}"
                            })

                        #Calculate average sentiment for this subreddit
                        if subreddit_mentions > 0:
                            avg_sentiment = subreddit_sentiment_total / subreddit_mentions
                            sentiment_indicators['avg_sentiment_per_subreddit'][subreddit] = {
                                'mentions': subreddit_mentions,
                                'avg_sentiment': round(avg_sentiment, 3)
                            }
                            print(f"   r/{subreddit}: {subreddit_mentions} mentions, avg sentiment: {avg_sentiment:.3f}")
                        
                    elif response.status_code == 429:
                        print(f"⚠️ Rate limited on r/{subreddit}, skipping...")
                        time.sleep(2)  # Longer wait for rate limiting
                        continue
                    
                    else:
                        print(f"⚠️ HTTP {response.status_code} error for r/{subreddit}")


                    # Rate limiting - be respectful to Reddit
                    time.sleep(1.2)  # Slightly longer delay

                except requests.exceptions.Timeout:
                    print(f"⚠️ Timeout fetching from r/{subreddit}")
                    continue
                except Exception as e:
                    print(f"⚠️ Error fetching from r/{subreddit}: {str(e)}")
                    continue
            
            #Calculate overall sentiment metrics
            total_mentions = sentiment_indicators['reddit_mentions']
            if total_mentions > 0:
                positive_ratio = sentiment_indicators['positive_mentions'] / total_mentions
                negative_ratio = sentiment_indicators['negative_mentions'] / total_mentions
                neutral_ratio = sentiment_indicators['neutral_mentions'] / total_mentions

                #Calculate wighted sentiment score
                sentiment_indicators['sentiment_score'] = 0.5 + (positive_ratio - negative_ratio)*0.5

                #Confidence calculation
                base_confidence = min(total_mentions/19, 0.8) #Max 0.8 confidence, need 15+ mentions for max

                #Boost confidence if sentiment is consistent acrooss subreddits
                subreddit_consistency = self._calculate_subreddit_consistency(
                    sentiment_indicators['avg_sentiment_per_subreddit']
                )
                confidence_boost = subreddit_consistency * 0.2

                #Factor in post engagement (upvotes and comments)
                avg_engagement = sentiment_indicators['total_upvotes'] / total_mentions if total_mentions > 0 else 0
                engagement_boost = min(avg_engagement/50, 0.1) #Max 0.1 boost

                final_confidence = min(base_confidence + confidence_boost + engagement_boost, 0.9)
                sentiment_indicators['confidence'] = round(final_confidence, 3)

                #Add summary stats
                sentiment_indicators['summary'] = {
                    'positive_ratio': round(positive_ratio, 3),
                    'negative_ratio': round(negative_ratio, 3),
                    'neutral_ratio': round(neutral_ratio, 3),
                    'avg_upvotes_per_post': round(avg_engagement, 1),
                    'subreddits_with_mentions': len(sentiment_indicators['avg_sentiment_per_subreddit'])
                }

                print(f"✅ Reddit Analysis Complete:")
                print(f"   Total mentions: {total_mentions}")
                print(f"   Sentiment breakdown - Positive: {sentiment_indicators['positive_mentions']}, "
                    f"Negative: {sentiment_indicators['negative_mentions']}, "
                    f"Neutral: {sentiment_indicators['neutral_mentions']}")
                print(f"   Overall sentiment: {sentiment_indicators['sentiment_score']:.3f}")
                print(f"   Confidence: {sentiment_indicators['confidence']:.3f}")

                return sentiment_indicators
            else:
                print(f"ℹ️ No Reddit mentions found for {ticker}")
                return None

        except Exception as e:
            print(f"❌ Reddit sentiment analysis failed: {str(e)}")
            return None

    def _analyze_text_sentiment(self, text: str) -> float:
        """Keyword-based sentiment analysis for Reddit text"""
        if not text:
            return 0.5

        text_lower = text.lower()
        sentiment_score = 0.5 #Neutral baseline
        total_weight = 0

        #Check positive keywords
        for keyword, weight in self.positive_keywords.items():
            if keyword in text_lower:
                #Scale impact based on keyword strength
                impact = (weight-1) * 0.08
                sentiment_score += impact
                total_weight += abs(impact)

        #Check negative keywords
        for keyword, weight in self.negative_keywords.items():
            if keyword in text_lower:
                #Scale impact based on keyword strength
                impact = weight * 0.08
                sentiment_score += impact
                total_weight += abs(impact)

        #Additional Reddit-specific keywords
        reddit_positive = ['moon', 'rocket', '🚀', 'bullish', 'calls', 'long', 'hodl', 'diamond hands']
        reddit_negative = ['puts', 'short', 'crash', 'dump', 'paper hands', 'bagholding']

        for word in reddit_positive:
            if word in text_lower:
                sentiment_score += 0.05
                total_weight += 0.05

        for word in reddit_negative:
            if word in text_lower:
                sentiment_score -= 0.05
                total_weight += 0.05
        
        #Normalize and ensure reasonable bounds
        if total_weight > 0:
            #Apply some smoothing to avoid extreme values
            sentiment_score = max(0.05, min(sentiment_score, 0.95))
        
        return sentiment_score

    def _calculate_subreddit_consistency(self, subreddit_sentiments: Dict[str, Any]) -> float:
        """Calculate how consistent sentiment is across different subreddits"""
        if len(subreddit_sentiments) < 2:
            return 0.5 #Not enough data for consistency

        sentiment_values = [
            data['avg_sentiment'] for data in subreddit_sentiments.values()
        ]

        #Calculate standard deviation
        mean_sentiment = sum(sentiment_values) / len(sentiment_values)
        variance = sum((x - mean_sentiment)**2 for x in sentiment_values) / len(sentiment_values)
        std_dev = variance**0.5

        #Convert to consistency score (lower is more consistent)
        #Maax std_dev for sentiment score is 0.5, so we normalize
        consistency = max(0, 1 - std_dev/0.5)
        return consistency
    
    def _get_news_sentiment(self, ticker: str, company_name: str) -> Optional[Dict[str, Any]]:
        """Get sentiment indicators from financial news using NewsAPI"""
        try:
            if not settings.news_api_key:
                print("⚠️ NewsAPI key is not configured, skipping news sentiment analysis")
                return None
            
            #NewsAPI endpoint
            base_url = 'https://newsapi.org/v2/everything'
            headers = {
                'X-Api-Key': settings.news_api_key,
                'User-Agent': 'AIStockSentiment/1.0'
            }

            #Build search query - combine ticker and company name for better results
            search_terms = [ticker]
            if company_name:
                search_terms.append(company_name)
            
            query = ' OR '.join([f'"{term}"' for term in search_terms])

            #NewsAPI parameters
            params = {
                'q': query,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 50, #Max articles per request
                'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'), #Last 7 days
                'domains': 'reuters.com,bloomberg.com,cnbc.com,marketwatch.com,yahoo.com,fool.com,seekingalpha.com,benzinga.com,investorplace.com,finance.yahoo.com' #Financial news sources
                }
            
            print(f"🔍 Searching financial news for {ticker}({company_name or 'N/A'})...")

            sentiment_indicators = {
                'news_mentions': 0,
                'positive_mentions': 0,
                'negative_mentions': 0,
                'neutral_mentions': 0,
                'sentiment_score': 0.5,
                'confidence': 0.1,
                'articles_analyzed': [],
                'news_sources': [],
                'total_articles_found': 0,
                'avg_sentiment_per_source': {},
                'headline_sentiment_distribution': {
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0
                }
            }

            response = requests.get(base_url, headers=headers, params=params, timeout=20)

            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                sentiment_indicators['total_articles_found'] = len(articles)

                source_sentiment_totals = {}
                source_article_counts = {}

                for article in articles:
                    #Extract article data
                    title = article.get('title', '').lower()
                    description = article.get('description', '').lower()
                    source_name = article.get('source', {}).get('name', 'Unknown')
                    published_at = article.get('publishedAt', '')
                    url = article.get('url', '')

                    #Combine title and description for sentiment analysis
                    combined_text = f"{title} {description}"

                    #More precise ticker matching for news articles
                    ticker_patterns = [
                        f" {ticker.lower()} ",
                        f"({ticker.upper()})",
                        f" {ticker.upper()} ",
                        f"${ticker.upper()}",
                        f"${ticker.lower()}"
                    ]

                    if company_name:
                        ticker_patterns.extend([
                            company_name.lower(),
                            f" {company_name.lower()} ",
                        ])

                    #Check if article is actually about our ticker
                    if any(pattern in combined_text for pattern in ticker_patterns):
                        sentiment_indicators['news_mentions'] += 1

                        #Analyze sentiment using keyword system
                        article_sentiment = self._analyze_news_text_sentiment(combined_text, title)

                        #Track source-based sentiment
                        if source_name not in source_sentiment_totals:
                            source_sentiment_totals[source_name] = 0
                            source_article_counts[source_name] = 0
                        source_sentiment_totals[source_name] += article_sentiment
                        source_article_counts[source_name] += 1

                        #Categorize sentiment
                        if article_sentiment > 0.6:
                            sentiment_indicators['positive_mentions'] += 1
                            sentiment_category = 'positive'
                            sentiment_indicators['headline_sentiment_distribution']['positive'] += 1
                        elif article_sentiment < 0.4:
                            sentiment_indicators['negative_mentions'] += 1
                            sentiment_category = 'negative'
                            sentiment_indicators['headline_sentiment_distribution']['negative'] += 1
                        else:
                            sentiment_indicators['neutral_mentions'] += 1
                            sentiment_category = 'neutral'
                            sentiment_indicators['headline_sentiment_distribution']['neutral'] += 1

                        #Store article info
                        sentiment_indicators['articles_analyzed'].append({
                            'title': article.get('title', '')[:120],  # First 120 chars
                            'source': source_name,
                            'published_at': published_at,
                            'sentiment': sentiment_category,
                            'sentiment_score': round(article_sentiment, 3),
                            'url': url
                        })

                        #Track unique news sources
                        if source_name not in sentiment_indicators['news_sources']:
                            sentiment_indicators['news_sources'].append(source_name)
                    
                #Calculate overall sentiment metrics
                total_mentions = sentiment_indicators['news_mentions']
                if total_mentions > 0:
                    positive_ratio = sentiment_indicators['positive_mentions'] / total_mentions
                    negative_ratio = sentiment_indicators['negative_mentions'] / total_mentions
                    neutral_ratio = sentiment_indicators['neutral_mentions'] / total_mentions

                    #Calculate weighted sentiment score
                    sentiment_indicators['sentiment_score'] = 0.5 + (positive_ratio - negative_ratio)*0.5

                    #Calculate per source averages
                    for source in source_sentiment_totals:
                        avg_sentiment = source_sentiment_totals[source] / source_article_counts[source]
                        sentiment_indicators['avg_sentiment_per_source'][source] = {
                            'mentions': source_article_counts[source],
                            'avg_sentiment': round(avg_sentiment, 3)
                        }

                    #Confidence calculation
                    base_confidence = min(total_mentions/25, 0.8) #Max 0.8 confidence, need 20+ mentions for max

                    #Boost confidence for source diversity
                    source_diversity_boost = min(len(sentiment_indicators['news_sources'])/10, 0.15)

                    #Boost confidence for more reputable sources
                    reputable_sources = ['Reuters', 'Bloomberg', 'CNBC', 'MarketWatch']
                    reputable_mentions = sum(1 for article in sentiment_indicators['articles_analyzed'] 
                                            if any(rep in article['source'] for rep in reputable_sources))
                    reputation_boost = min((reputable_mentions/total_mentions) * 0.1, 0.1)

                    final_confidence = min(base_confidence + source_diversity_boost + reputation_boost, 0.9)
                    sentiment_indicators['confidence'] = round(final_confidence, 3)

                    #Add summary stats
                    sentiment_indicators['summary'] = {
                        'positive_ratio': round(positive_ratio, 3),
                        'negative_ratio': round(negative_ratio, 3),
                        'neutral_ratio': round(neutral_ratio, 3),
                        'unique_sources': len(sentiment_indicators['news_sources']),
                        'avg_articles_per_source': round(total_mentions/len(sentiment_indicators['news_sources']), 1),
                        'reputable_source_coverage': round(reputable_mentions/total_mentions, 3)
                    }

                    print(f"✅ News Analysis Complete:")
                    print(f"   Total relevant articles: {total_mentions}")
                    print(f"   Sentiment breakdown - Positive: {sentiment_indicators['positive_mentions']}, "
                        f"Negative: {sentiment_indicators['negative_mentions']}, "
                        f"Neutral: {sentiment_indicators['neutral_mentions']}")
                    print(f"   Overall sentiment: {sentiment_indicators['sentiment_score']:.3f}")
                    print(f"   Confidence: {sentiment_indicators['confidence']:.3f}")
                    print(f"   Sources covered: {len(sentiment_indicators['news_sources'])}")
                    
                    return sentiment_indicators
                else:
                    print(f"ℹ️ No relevant news articles found for {ticker}({company_name or 'N/A'})")
                    return None

            elif response.status_code == 429:
                print(f"⚠️ NewsAPI rate limit exceeded")
                return None
            elif response.status_code == 401:
                print(f"❌ NewsAPI authentication failed - check your API key")
                return None
            else:
                print(f"⚠️ NewsAPI HTTP {response.status_code} error")
                return None
            
        except requests.exceptions.Timeout:
            print(f"⚠️ NewsAPI request timeout")
            return None
        except Exception as e:
            print(f"❌ News sentiment analysis failed: {str(e)}")
            return None

    def _analyze_news_text_sentiment(self, text: str, title: str) -> float:
        """Keyword-based sentiment analysis for news articles"""
        if not text:
            return 0.5

        text_lower = text.lower()
        title_lower = title.lower() if title else ''

        sentiment_score = 0.5 #Neutral baseline
        total_weight = 0

        #Headlines carry more weight than descriptions
        title_weight_multiplier = 0.18

        #Check positive keywords in both title and text
        for keyword, weight in self.positive_keywords.items():
            title_matches = title_lower.count(keyword)
            text_matches = text_lower.count(keyword) - title_matches #Avoid double counting

            if title_matches > 0:
                impact = (weight-1) * title_weight_multiplier * title_matches
                sentiment_score += impact
                total_weight += abs(impact)
            
            if text_matches > 0:
                impact = (weight-1) * 0.08 * text_matches
                sentiment_score += impact
                total_weight += abs(impact)
        
        #Check negative keywords in both title and text
        for keyword, weight in self.negative_keywords.items():
            title_matches = title_lower.count(keyword)
            text_matches = text_lower.count(keyword) - title_matches

            if title_matches > 0:
                impact = weight * title_weight_multiplier * title_matches
                sentiment_score += impact
                total_weight += abs(impact)

            if text_matches > 0:
                impact = weight * 0.08 * text_matches
                sentiment_score += impact
                total_weight += abs(impact)
        
        #Financial news-specific keywords
        financial_positive = [
            'beats expectations', 'exceeds forecast', 'strong earnings', 'revenue growth',
            'profit increase', 'dividend increase', 'share buyback', 'merger', 'acquisition',
            'partnership', 'expansion', 'new contract', 'patent approval', 'fda approval'
        ]
    
        financial_negative = [
            'misses expectations', 'below forecast', 'earnings decline', 'revenue drop',
            'profit warning', 'dividend cut', 'layoffs', 'bankruptcy', 'investigation',
            'lawsuit', 'recall', 'downgrade', 'guidance lowered', 'loss widens'
        ]
    
        # Check financial patterns
        for phrase in financial_positive:
            if phrase in text_lower:
                weight_boost = 0.08 if phrase in title_lower else 0.05
                sentiment_score += weight_boost
                total_weight += weight_boost
    
        for phrase in financial_negative:
            if phrase in text_lower:
                weight_boost = 0.08 if phrase in title_lower else 0.05
                sentiment_score += weight_boost
                total_weight += weight_boost
    
        # Normalize and ensure reasonable bounds
        if total_weight > 0:
            sentiment_score = max(0.05, min(0.95, sentiment_score))
        
        return sentiment_score

    def _get_twitter_sentiment(self, ticker: str, company_name: str) -> Optional[Dict[str, Any]]:
        """Get sentiment indicators from Twitter using Twitter API v2"""
        try:
            if not settings.twitter_bearer_token:
                print("⚠️ Twitter API key is not configured, skipping Twitter sentiment analysis")
                return None
            
            #Twitter API v2 endpoints
            search_url = "https://api.twitter.com/2/tweets/search/recent"
            headers = {
                'Authorization': f'Bearer {settings.twitter_bearer_token}',
                'User-Agent': 'AIStockSentiment/1.0'
            }

            #Build search query - Twitter has specific syntax
            search_terms = [f"${ticker.upper()}", ticker.upper()]
            if company_name:
                search_terms.append(company_name)

            #Twitter search query with operators
            query = f"({' OR '.join(search_terms)}) lang:en -is:retweet" #Exclude retweets

            #Twitter API v2 parameters
            params = {
                'query': query,
                'max_results': 100,
                'tweet.fields': 'created_at,author_id,public_metrics,context_annotations,lang',
                'expansions': 'author_id',
                'user.fields': 'verified,public_metrics'
            }

            print(f"🔍 Searching Twitter for {ticker}({company_name or 'N/A'})...")

            sentiment_indicators = {
                'twitter_mentions': 0,
                'positive_mentions': 0,
                'negative_mentions': 0,
                'neutral_mentions': 0,
                'sentiment_score': 0.5,
                'confidence': 0.1,
                'tweets_analyzed': [],
                'total_engagement': 0,
                'verified_user_mentions': 0,
                'avg_engagement_per_tweet': 0,
                'sentiment_distribution': {
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0
                },
                'top_engaged_tweets': []
            }

            response = requests.get(search_url, headers=headers, params=params, timeout=20)

            if response.status_code == 200:
                data = response.json()
                tweets = data.get('data', [])
                users = {user['id']: user for user in data.get('includes', {}).get('users', [])}


                if not tweets:
                    print(f"ℹ️ No recent tweets found for {ticker}")
                    return None

                engagement_scores = []

                for tweet in tweets:
                    tweet_text = tweet.get('text', '').lower()
                    tweet_id = tweet.get('id', '')
                    author_id = tweet.get('author_id', '')
                    created_at = tweet.get('created_at', '')

                    #Get user info
                    user_info = users.get(author_id, {})
                    is_verified = user_info.get('verified', False)

                    #Get engagement metrics
                    metrics = tweet.get('public_metrics', {})
                    likes = metrics.get('like_count', 0)
                    retweets = metrics.get('retweet_count', 0)
                    replies = metrics.get('reply_count', 0)
                    quotes = metrics.get('quote_count', 0)

                    #More precise ticker matching for tweets
                    ticker_patterns = [
                        f"${ticker.lower()}",
                        f"${ticker.upper()}",
                        f" {ticker.lower()} ",
                        f" {ticker.upper()} ",
                        f"#{ticker.lower()}",
                        f"#{ticker.upper()}"
                    ]
                    
                    if company_name:
                        ticker_patterns.extend([
                            company_name.lower(),
                            f" {company_name.lower()} "
                        ])

                    #Check if tweet is actually about our ticker
                    if any(pattern in tweet_text for pattern in ticker_patterns):
                        #Calculate engagement score
                        engagement_score = likes + retweets*2 + replies*1.5 + quotes*1.5
                        engagement_scores.append(engagement_score)
                        sentiment_indicators['total_engagement'] += engagement_score

                        sentiment_indicators['twitter_mentions'] += 1

                        if is_verified:
                            sentiment_indicators['verified_user_mentions'] += 1

                        #Analyze sentiment using Twitter-specific analysis
                        tweet_sentiment = self._analyze_twitter_text_sentiment(tweet_text)

                        #Categorize sentiment
                        if tweet_sentiment > 0.6:
                            sentiment_indicators['positive_mentions'] += 1
                            sentiment_category = 'positive'
                            sentiment_indicators['sentiment_distribution']['positive'] += 1
                        elif tweet_sentiment < 0.4:
                            sentiment_indicators['negative_mentions'] += 1
                            sentiment_category = 'negative'
                            sentiment_indicators['sentiment_distribution']['negative'] += 1
                        else:
                            sentiment_indicators['neutral_mentions'] += 1
                            sentiment_category = 'neutral'
                            sentiment_indicators['sentiment_distribution']['neutral'] += 1

                        #Store tweet info
                        tweet_info = {
                            'text': tweet.get('text', '')[:140], #First 140 chars
                            'author_verified': is_verified,
                            'created_at': created_at,
                            'sentiment': sentiment_category,
                            'sentiment_score': round(tweet_sentiment, 3),
                            'engagement_score': engagement_score,
                            'likes': likes,
                            'retweets': retweets,
                            'replies': replies,
                            'url': f"https://x.com/i/{tweet_id}"
                        }

                        sentiment_indicators['tweets_analyzed'].append(tweet_info)

                        #Track tope engaged tweets
                        if len(sentiment_indicators['top_engaged_tweets']) < 5:
                            sentiment_indicators['top_engaged_tweets'].append(tweet_info)
                        else:
                            #Replace lowest engagement tweet if current is higher
                            min_engagement = min(
                                sentiment_indicators['top_engaged_tweets'],
                                key = lambda x: x['engagement_score']
                            )
                            if engagement_score > min_engagement['engagement_score']:
                                sentiment_indicators['top_engaged_tweets'].remove(min_engagement)
                                sentiment_indicators['top_engaged_tweets'].append(tweet_info)

                #Calculate overall sentiment metrics
                total_mentions = sentiment_indicators['twitter_mentions']
                if total_mentions > 0:
                    positive_ratio = sentiment_indicators['positive_mentions'] / total_mentions
                    negative_ratio = sentiment_indicators['negative_mentions'] / total_mentions
                    neutral_ratio = sentiment_indicators['neutral_mentions'] / total_mentions

                    #Calculate weighted sentiment score
                    sentiment_indicators['sentiment_score'] = 0.5 + (positive_ratio - negative_ratio)*0.5

                    #Calculate average engagement
                    sentiment_indicators['avg_engagement_per_tweet'] = round(
                        sentiment_indicators['total_engagement'] / total_mentions, 1
                    )

                    #Calculate confidence
                    base_confidence = min(total_mentions/43, 0.7) #Max 0.8 confidence, need 30+ mentions for max

                    #Boost confidence for verified user mentions
                    verified_ratio  = sentiment_indicators['verified_user_mentions'] / total_mentions
                    verified_boost = verified_ratio * 0.15

                    #Boost confidence for high engagement
                    avg_engagement = sentiment_indicators['avg_engagement_per_tweet']
                    engagement_boost = min(avg_engagement/100, 0.1) #Max 0.1 boost

                    #Recent activity boost (Twitter data is very fresh)
                    recency_boost = 0.05

                    final_confidence = min(base_confidence + verified_boost + engagement_boost + recency_boost, 0.85)
                    sentiment_indicators['confidence'] = round(final_confidence, 3)

                    #Sort top engaged tweets
                    sentiment_indicators['top_engaged_tweets'].sort(
                        key = lambda x: x['engagement_score'], reverse = True
                    )

                    #Add summary stats
                    sentiment_indicators['summary'] = {
                        'positive_ratio': round(positive_ratio, 3),
                        'negative_ratio': round(negative_ratio, 3),
                        'neutral_ratio': round(neutral_ratio, 3),
                        'verified_user_ratio': round(verified_ratio, 3),
                        'avg_engagement_per_tweet': sentiment_indicators['avg_engagement_per_tweet'],
                        'high_engagement_tweets': len(
                                                        [t for t in sentiment_indicators['tweets_analyzed'] 
                                                        if t['engagement_score'] > 10])
                        }

                    print(f"✅ Twitter Analysis Complete:")
                    print(f"   Total relevant tweets: {total_mentions}")
                    print(f"   Sentiment breakdown - Positive: {sentiment_indicators['positive_mentions']}, "
                        f"Negative: {sentiment_indicators['negative_mentions']}, "
                        f"Neutral: {sentiment_indicators['neutral_mentions']}")
                    print(f"   Overall sentiment: {sentiment_indicators['sentiment_score']:.3f}")
                    print(f"   Confidence: {sentiment_indicators['confidence']:.3f}")
                    print(f"   Verified users: {sentiment_indicators['verified_user_mentions']}")
                    print(f"   Avg engagement: {sentiment_indicators['avg_engagement_per_tweet']}")
                        
                    return sentiment_indicators
                    
                else:
                    print(f"ℹ️ No relevant tweets found for {ticker}")
                    return None

            elif response.status_code == 429:
                print(f"⚠️ Twitter API rate limit exceeded")
                return None
            elif response.status_code == 401:
                print(f"❌ Twitter API authentication failed - check your Bearer Token")
                return None
            elif response.status_code == 403:
                print(f"❌ Twitter API access forbidden - check your app permissions")
                return None
            else:
                print(f"⚠️ Twitter API HTTP {response.status_code} error")
                return None

        except requests.exceptions.Timeout:
            print(f"⚠️ Twitter API request timeout")
            return None
        except Exception as e:
            print(f"❌ Twitter sentiment analysis failed: {str(e)}")
            return None

    def _analyze_twitter_text_sentiment(self, text: str) -> float:
        """Sentiment analysis specifically for Twitter content"""
        if not text:
            return 0.5

        sentiment_score = 0.5 #Neutral baseline
        total_weight = 0

        #Check positive keywords
        for keyword, weight in self.positive_keywords.items():
            matches = text.count(keyword)
            if matches > 0:
                impact = (weight - 1.0) * 0.1 * matches  #Slightly higher impact for Twitter
                sentiment_score += impact
                total_weight += abs(impact)
        
        #Check negative keywords
        for keyword, weight in self.negative_keywords.items():
            matches = text.count(keyword)
            if matches > 0:
                impact = weight * 0.1 * matches
                sentiment_score += impact
                total_weight += abs(impact)
        
        #Twitter-specific sentiment indicators
        twitter_positive = [
            '🚀', '📈', '💎', '🌙', 'moon', 'rocket', 'bullish', 'calls', 'long', 
            'hodl', 'diamond hands', 'to the moon', 'stonks', '💪', '🔥', 'lfg',
            'buying the dip', 'btfd', 'yolo', 'tendies'
        ]
        
        twitter_negative = [
            '📉', '💀', '😭', 'rip', 'puts', 'short', 'bearish', 'crash', 'dump',
            'paper hands', 'bagholding', 'fud', 'dead cat bounce', 'rug pull',
            'exit scam', 'worthless', 'going to zero'
        ]
        
        #Check Twitter-specific patterns
        for indicator in twitter_positive:
            if indicator in text:
                sentiment_score += 0.06
                total_weight += 0.06
        
        for indicator in twitter_negative:
            if indicator in text:
                sentiment_score -= 0.06
                total_weight += 0.06
        
        #Emoji sentiment analysis (basic)
        positive_emojis = ['😊', '😄', '🎉', '👍', '💰', '📈', '🚀', '🔥', '💎', '🌙']
        negative_emojis = ['😢', '😭', '😡', '👎', '💸', '📉', '💀', '😵', '🤮']
        
        for emoji in positive_emojis:
            if emoji in text:  #Use original text for emojis
                sentiment_score += 0.03
                total_weight += 0.03
        
        for emoji in negative_emojis:
            if emoji in text:
                sentiment_score -= 0.03
                total_weight += 0.03
        
        #Check for ALL CAPS (often indicates strong emotion)
        caps_words = [word for word in text.split() if word.isupper() and len(word) > 2]
        if caps_words:
            caps_ratio = len(caps_words) / len(text.split())
            if caps_ratio > 0.3:  #More than 30% caps
                #Could be positive or negative excitement - check context
                caps_text = ' '.join(caps_words).lower()
                caps_sentiment = 0
                
                for keyword, weight in self.positive_keywords.items():
                    if keyword in caps_text:
                        caps_sentiment += 0.02
                
                for keyword, weight in self.negative_keywords.items():
                    if keyword in caps_text:
                        caps_sentiment -= 0.02
                
                sentiment_score += caps_sentiment
                total_weight += abs(caps_sentiment)
        
        #Normalize and ensure reasonable bounds
        if total_weight > 0:
            sentiment_score = max(0.05, min(0.95, sentiment_score))
        
        return sentiment_score
    
    def _analyze_analyst_sentiment(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Extract analyst sentiment from existing fundamental data"""
        try:
            #Get fundamental data (already cached if fundamental analysis tool is used beforehand)
            fundamental_data = multi_source_tool.get_fundamental_data(ticker)
            if not fundamental_data.success:
                print(f"⚠️ No fundamental data available for analyst sentiment")
                return None
            
            fund_data = fundamental_data.data

            #Extract analyst data from Yahoo Finance
            recommendation_mean = fund_data.get('recommendation_mean')
            recommendation_key = fund_data.get('recommendation_key')
            num_analysts = fund_data.get('number_of_analyst_opinions')
            target_mean = fund_data.get('target_mean_price')
            current_price = fund_data.get('current_price')

            #Extract Alpha Vantage format if Yahoo data not available
            if not recommendation_mean:
                av_target = fund_data.get('analyst_target_price')
                strong_buy = fund_data.get('analyst_rating_strong_buy', 0)
                buy = fund_data.get('analyst_rating_buy', 0)
                hold = fund_data.get('analyst_rating_hold', 0)
                sell = fund_data.get('analyst_rating_sell', 0)
                strong_sell = fund_data.get('analyst_rating_strong_sell', 0)

                #Calculate weighted recommendation mean
                total_ratings = strong_buy + buy + hold + sell + strong_sell
                if total_ratings > 0:
                    recommendation_mean = (strong_buy*1 + buy*2 + hold*3 + sell*4 + strong_sell*5) / total_ratings
                    num_analysts = total_ratings
                    target_mean = av_target


            #Check if we have any analyst data
            if not recommendation_mean and not target_mean:
                print(f"ℹ️ No analyst data available for {ticker}")
                return None

            print(f"🔍 Analyzing analyst sentiment for {ticker}...")

            #Initialize sentiment indicators
            sentiment_indicators = {
                'analyst_mentions': num_analysts or 0,
                'recommendation_score': 0.5,
                'price_target_sentiment': 0.5,
                'overall_sentiment_score': 0.5,
                'confidence': 0.1,
                'recommendation_breakdown': {},
                'price_analysis': {},
                'sentiment_signals': []
            }

            #Analyze recommendation sentiment(1=Strong Buy ... 5=Strong Sell)
            if recommendation_mean:
                #Convert recommendation mean to sentiment score (invert scale)
                #1.0-1.5 = Very Bullish
                #1.5-2.5 = Bullish
                #2.5-3.5 = Neutral
                #3.5-4.5 = Bearish
                #4.5-5.0 = Very Bearish
                if recommendation_mean <= 1.5:
                    sentiment_indicators['recommendation_score'] = 0.9
                    recommendation_sentiment = 'Very Bullish'
                    sentiment_indicators['sentiment_signals'].append('Strong Analyst Buy Recommendations')
                elif recommendation_mean <= 2.5:
                    sentiment_indicators['recommendation_score'] = 0.7
                    recommendation_sentiment = 'Bullish'
                    sentiment_indicators['sentiment_signals'].append('Positive Analyst Recommendations')
                elif recommendation_mean <= 3.5:
                    sentiment_indicators['recommendation_score'] = 0.5
                    recommendation_sentiment = 'Neutral'
                    sentiment_indicators['sentiment_signals'].append('Mixed Analyst Recommendations')
                elif recommendation_mean <= 4.5:
                    sentiment_indicators['recommendation_score'] = 0.3
                    recommendation_sentiment = 'Bearish'
                    sentiment_indicators['sentiment_signals'].append('Negative Analyst Recommendations')
                else:
                    sentiment_indicators['recommendation_score'] = 0.1
                    recommendation_sentiment = 'Very Bearish'
                    sentiment_indicators['sentiment_signals'].append('Strong Analyst Sell Recommendations')
                

                sentiment_indicators['recommendation_breakdown'] = {
                    'recommendation_mean': round(recommendation_mean, 2),
                    'recommendation_key': recommendation_key,
                    'sentiment_recommendation': recommendation_sentiment,
                    'num_analysts': num_analysts
                }

                #Analyze price target sentiment
                if target_mean and current_price:
                    price_upside = ((target_mean - current_price) / current_price) * 100

                    #Convert price upside to sentiment score
                    if price_upside >= 20:
                        sentiment_indicators['price_target_sentiment'] = 0.9
                        price_sentiment = 'Very Bullish'
                        sentiment_indicators['sentiment_signals'].append(f"High analyst price target upside ({price_upside:.1f}%)")
                    elif price_upside >= 10:
                        sentiment_indicators['price_target_sentiment'] = 0.75
                        price_sentiment = 'Bullish'
                        sentiment_indicators['sentiment_signals'].append(f"Positive analyst price target upside ({price_upside:.1f}%)")
                    elif price_upside >= 0:
                        sentiment_indicators['price_target_sentiment'] = 0.6
                        price_sentiment = 'Slightly Bullish'
                        sentiment_indicators['sentiment_signals'].append(f"Modest analyst price target upside ({price_upside:.1f}%)")
                    elif price_upside >= -10:
                        sentiment_indicators['price_target_sentiment'] = 0.4
                        price_sentiment = 'Slightly Bearish'
                        sentiment_indicators['sentiment_signals'].append(f"Limited analyst price target downside ({price_upside:.1f}%)")
                    else:
                        sentiment_indicators['price_target_sentiment'] = 0.2
                        price_sentiment = 'Bearish'
                        sentiment_indicators['sentiment_signals'].append(f"Significant analyst price target downside ({price_upside:.1f}%)")

                    sentiment_indicators['price_analysis'] = {
                        'current_price': current_price,
                        'target_mean': target_mean,
                        'upside_potential': round(price_upside, 2),
                        'sentiment_interpretation': price_sentiment
                    }

                #Calculate overall analyst sentiment score
                rec_weight = 0.6 #Recommendations weighted more heavily
                price_weight = 0.4

                overall_score = (
                    sentiment_indicators['recommendation_score'] * rec_weight +
                    sentiment_indicators['price_target_sentiment'] * price_weight
                )

                sentiment_indicators['overall_sentiment_score'] = round(overall_score, 3)

                #Calculate confidence base on number of analysts and data availability
                base_confidence = 0.3  #Base confidence for having analyst data

                if num_analysts:
                    #More analysts = higher confidence, cap at 0.4 additional confidence
                    analyst_confidence_boost = min(num_analysts/63, 0.4) #25+ analysts for max boost

                else:
                    analyst_confidence_boost = 0.1 #Some boost even without count

                #Boost confidence if we have both recommendation and price target data
                data_completeness_boost = 0.2 if (recommendation_mean and target_mean) else 0.1
                
                final_confidence = min(base_confidence + analyst_confidence_boost + data_completeness_boost, 0.85)
                sentiment_indicators['confidence'] = round(final_confidence, 3)

                #Add summary stats
                sentiment_indicators['summary'] = {
                    'has_recommendations': bool(recommendation_mean),
                    'has_price_targets': bool(target_mean),
                    'analyst_coverage': num_analysts or 0,
                    'overall_sentiment': self._get_sentiment_label(sentiment_indicators['overall_sentiment_score']),
                    'key_signals_count': len(sentiment_indicators['sentiment_signals'])
                }

                print(f"✅ Analyst Sentiment Analysis Complete:")
                print(f"   Analyst coverage: {num_analysts or 'N/A'} analysts")
                if recommendation_mean:
                    print(f"   Recommendation: {recommendation_sentiment} ({recommendation_mean:.2f})")
                if target_mean and current_price:
                    print(f"   Price target upside: {price_upside:.1f}%")
                print(f"   Overall sentiment: {sentiment_indicators['overall_sentiment_score']:.3f}")
                print(f"   Confidence: {sentiment_indicators['confidence']:.3f}")

                return sentiment_indicators
            
        except Exception as e:
            print(f"❌ Analyst sentiment analysis failed: {str(e)}")
            return None

    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to human-readable label"""
        if score >= 0.8:
            return 'Very Positive'
        elif score >= 0.6:
            return 'Positive'
        elif score >= 0.4:
            return 'Neutral'
        elif score >= 0.2:
            return 'Negative'
        else:
            return 'Very Negative'
    
    def _analyze_market_sentiment(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Analyze market-based sentiment indicators using existing technical analysis"""
        try:
            print(f"🔍 Analyzing market sentiment for {ticker}...")

            #Get technical analysis results(leverages existing caching by expecting technical analysis tool to be used beforehand)
            technical_result = technical_analysis_tool.analyze_stock(ticker)
            if not technical_result.success:
                print("⚠️ No technical analysis data available for market sentiment")
                return None

            tech_data = technical_result.data
            indicators = tech_data.get('indicators', {})
            overall_analysis = tech_data.get('overall_analysis', {})
            trend_analysis = tech_data.get('trend_analysis', {})

            #Initialize sentiment indicators
            sentiment_indicators = {
                'trend_sentiment': 0.5,
                'volume_sentiment': 0.5,
                'overall_technical_sentiment': 0.5,
                'individual_technical_sentiment': 0.5,
                'short_interest_sentiment': 0.5,
                'overall_sentiment_score': 0.5,
                'confidence': 0.1,
                'sentiment_signals': [],
                'market_indicators': {}
            }

            #1. Volume Sentiment Analysis
            volume_sentiment = self._extract_volume_sentiment(indicators.get('volume_indicators', {}))
            sentiment_indicators.update(volume_sentiment)
            all_signals = volume_sentiment.get('sentiment_signals', [])

            #2. Technical Indicators Sentiment
            technical_sentiment = self._extract_technical_sentiment(indicators)
            sentiment_indicators.update(technical_sentiment)
            all_signals.extend(technical_sentiment.get('sentiment_signals', []))

            #3. Trend Sentiment
            trend_sentiment = self._extract_trend_sentiment(trend_analysis)
            sentiment_indicators.update(trend_sentiment)
            all_signals.extend(trend_sentiment.get('sentiment_signals', []))

            #4. Overall Technical Signal Sentiment
            overall_sentiment = self._extract_overall_sentiment(overall_analysis)
            sentiment_indicators.update(overall_sentiment)
            all_signals.extend(overall_sentiment.get('sentiment_signals', []))

            #5. Short Interest Sentiment

            #Get fundamental data for short interest
            fundamental_result = multi_source_tool.get_fundamental_data(ticker)
            fund_data = fundamental_result.data if fundamental_result.success else {}

            short_sentiment = self._analyze_short_interest_sentiment(fund_data)
            sentiment_indicators.update(short_sentiment)
            all_signals.extend(short_sentiment.get('sentiment_signals', []))

            #Update with all collected signals
            sentiment_indicators['sentiment_signals'] = all_signals
            
            #Calculate overall market sentiment score
            weights = {
                'volume_sentiment': 0.2,
                'individual_technical_sentiment': 0.3,
                'trend_sentiment': 0.2,
                'overall_technical_sentiment': 0.15,
                'short_interest_sentiment': 0.15
            }

            overall_score = sum(sentiment_indicators[key] * weight for key, weight in weights.items())
            sentiment_indicators['overall_sentiment_score'] = round(overall_score, 3)

            #Calculate confidence base on technical analysis confidence and data availability
            base_confidence = overall_analysis.get('confidence', 0.5)

            #Boost confidence based on data completeness
            data_completeness_boost = 0.1 if fund_data.get('short_ratio') else 0
            signal_boost = min((len(sentiment_indicators['sentiment_signals'])/8) * 0.15, 0.15)
        
            final_confidence = min(base_confidence + data_completeness_boost + signal_boost, 0.9)
            sentiment_indicators['confidence'] = round(final_confidence, 3)

            print(f"✅ Market Sentiment Analysis Complete:")
            print(f"   Technical sentiment: {sentiment_indicators['individual_technical_sentiment']:.3f}")
            print(f"   Trend sentiment: {sentiment_indicators['trend_sentiment']:.3f}")
            print(f"   Overall market sentiment: {sentiment_indicators['overall_sentiment_score']:.3f}")
            print(f"   Technical signal: {overall_analysis.get('signal', 'HOLD')}")
            print(f"   Confidence: {sentiment_indicators['confidence']:.3f}")
            
            return sentiment_indicators
        
        except Exception as e:
            print(f"❌ Market sentiment analysis failed: {str(e)}")
            return None  

    def _extract_volume_sentiment(self, volume_indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Extract sentiment from volume indicators"""
        volume_sentiment = 0.5
        signals = []

        volume_signal = volume_indicators.get('volume_signal', 'NORMAL')
        volume_ratio = volume_indicators.get('volume_ratio', 1.0)

        if volume_signal == 'HIGH VOLUME':
            volume_sentiment = 0.75
            signals.append(f"Very high volume activity ({volume_ratio:.1f}x average)")
        elif volume_signal == 'LOW VOLUME':
            volume_sentiment = 0.4
            signals.append("Low volume activity suggests weak conviction")

        return {
            'volume_sentiment': volume_sentiment,
            'sentiment_signals': signals,
            'market_indicators': {
                'volume_ratio': volume_ratio,
                'volume_signal': volume_signal,
                'current_volume': volume_indicators.get('current_volume'),
                'avg_volume_20': volume_indicators.get('avg_volume_20')
            }
        }

    def _extract_technical_sentiment(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Extract sentiment from price momentum indicators"""
        technical_sentiment = 0.5
        signals = []

        #RSI sentiment
        rsi = indicators.get('rsi', {})
        rsi_signal = rsi.get('signal', 'NEUTRAL')
        if rsi_signal == "OVERSOLD":
            technical_sentiment += 0.15
            signals.append(f"RSI oversold ({rsi.get('current_rsi', 0):.1f}) - potential buying opportunity")
        elif rsi_signal == "OVERBOUGHT":
            technical_sentiment -= 0.15
            signals.append(f"RSI overbought ({rsi.get('current_rsi', 0):.1f}) - potential selling pressure")
        
        #MACD sentiment
        macd = indicators.get('macd', {})
        macd_signal = macd.get('signal', 'NEUTRAL')
        macd_crossover = macd.get('crossover')
        
        if macd_crossover == "BULLISH CROSSOVER":
            technical_sentiment += 0.2
            signals.append("MACD bullish crossover - positive momentum shift")
        elif macd_crossover == "BEARISH CROSSOVER":
            technical_sentiment -= 0.2
            signals.append("MACD bearish crossover - negative momentum shift")
        elif macd_signal == "BULLISH":
            technical_sentiment += 0.1
            signals.append("MACD above signal line - bullish momentum")
        elif macd_signal == "BEARISH":
            technical_sentiment -= 0.1
            signals.append("MACD below signal line - bearish momentum")
        
        #Bollinger Bands sentiment
        bollinger = indicators.get('bollinger', {})
        bollinger_signal = bollinger.get('signal', 'NEUTRAL')
        if bollinger_signal == "OVERSOLD":
            technical_sentiment += 0.1
            signals.append("Price below lower Bollinger Band - oversold condition")
        elif bollinger_signal == "OVERBOUGHT":
            technical_sentiment -= 0.1
            signals.append("Price above upper Bollinger Band - overbought condition")
        
        #Stochastic sentiment
        stochastic = indicators.get('stochastic', {})
        stoch_signal = stochastic.get('signal', 'NEUTRAL')
        if stoch_signal == "OVERSOLD":
            technical_sentiment += 0.08
            signals.append("Stochastic oversold - potential reversal")
        elif stoch_signal == "OVERBOUGHT":
            technical_sentiment -= 0.08
            signals.append("Stochastic overbought - potential reversal")
        
        technical_sentiment = max(0.05, min(0.95, technical_sentiment))
        
        return {
            'individual_technical_sentiment': round(technical_sentiment, 3),
            'sentiment_signals': signals,
            'market_indicators': {
                'rsi_signal': rsi_signal,
                'macd_signal': macd_signal,
                'macd_crossover': macd_crossover,
                'bollinger_signal': bollinger_signal,
                'stochastic_signal': stoch_signal
            }
        }

    def _extract_trend_sentiment(self, trend_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract sentiment from trend analysis"""
        trend_sentiment = 0.5
        signals = []

        #Short-term trend
        short_term = trend_analysis.get('short_term', {})
        short_trend = short_term.get('trend', 'SIDEWAYS')
        short_strength = short_term.get('strength', 0)
        
        if short_trend == "STRONG UPTREND":
            trend_sentiment += 0.2 * short_strength
            signals.append(f"Strong short-term uptrend (strength: {short_strength:.2f})")
        elif short_trend == "UPTREND":
            trend_sentiment += 0.1 * short_strength
            signals.append(f"Short-term uptrend (strength: {short_strength:.2f})")
        elif short_trend == "STRONG DOWNTREND":
            trend_sentiment -= 0.2 * short_strength
            signals.append(f"Strong short-term downtrend (strength: {short_strength:.2f})")
        elif short_trend == "DOWNTREND":
            trend_sentiment -= 0.1 * short_strength
            signals.append(f"Short-term downtrend (strength: {short_strength:.2f})")
        
        #Medium-term trend
        medium_term = trend_analysis.get('medium_term', {})
        medium_trend = medium_term.get('trend', 'SIDEWAYS')
        medium_strength = medium_term.get('strength', 0)
        
        if medium_trend == "STRONG UPTREND":
            trend_sentiment += 0.15 * medium_strength
            signals.append(f"Strong medium-term uptrend")
        elif medium_trend == "UPTREND":
            trend_sentiment += 0.08 * medium_strength
            signals.append(f"Medium-term uptrend")
        elif medium_trend == "STRONG DOWNTREND":
            trend_sentiment -= 0.15 * medium_strength
            signals.append(f"Strong medium-term downtrend")
        elif medium_trend == "DOWNTREND":
            trend_sentiment -= 0.08 * medium_strength
            signals.append(f"Medium-term downtrend")
        
        trend_sentiment = max(0.05, min(0.95, trend_sentiment))
        
        return {
            'trend_sentiment': round(trend_sentiment, 3),
            'sentiment_signals': signals,
            'market_indicators': {
                'short_term_trend': short_trend,
                'short_term_strength': short_strength,
                'medium_term_trend': medium_trend,
                'medium_term_strength': medium_strength
            }
        }

    def _extract_overall_sentiment(self, overall_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract sentiment from overall technical indicators"""
        momentum_sentiment = 0.5
        signals = []
        
        technical_signal = overall_analysis.get('signal', 'HOLD')
        confidence = overall_analysis.get('confidence', 0.5)
        bullish_indicators = overall_analysis.get('bullish_indicators', 0)
        bearish_indicators = overall_analysis.get('bearish_indicators', 0)

        #Convert technical signal to sentiment
        if technical_signal == "BUY":
            momentum_sentiment = 0.5 + (confidence * 0.4)  # 0.5 to 0.9
            signals.append(f"Technical analysis suggests BUY (confidence: {confidence:.2f})")
        elif technical_signal == "SELL":
            momentum_sentiment = 0.5 - (confidence * 0.4)  # 0.1 to 0.5
            signals.append(f"Technical analysis suggests SELL (confidence: {confidence:.2f})")
        else:  #HOLD
            if bullish_indicators > bearish_indicators:
                momentum_sentiment = 0.55
                signals.append("Technical indicators lean slightly bullish")
            elif bearish_indicators > bullish_indicators:
                momentum_sentiment = 0.45
                signals.append("Technical indicators lean slightly bearish")
            else:
                signals.append("Technical indicators are neutral")
        
        return {
            'overall_technical_sentiment': round(momentum_sentiment, 3),
            'sentiment_signals': signals,
            'market_indicators': {
                'technical_signal': technical_signal,
                'technical_confidence': confidence,
                'bullish_indicators': bullish_indicators,
                'bearish_indicators': bearish_indicators
            }
        }

    def _analyze_short_interest_sentiment(self, fund_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze short interest for sentiment indicators"""
        try:
            short_ratio = fund_data.get('short_ratio')
            short_percent_float = fund_data.get('short_percent_of_float') or fund_data.get('short_percent_float')
            
            short_sentiment = 0.5
            signals = []
            
            if short_ratio:
                if short_ratio > 10:
                    short_sentiment += 0.1
                    signals.append(f"High short interest (days to cover: {short_ratio:.1f})")
                elif short_ratio > 5:
                    short_sentiment += 0.05
                    signals.append(f"Elevated short interest (days to cover: {short_ratio:.1f})")
                elif short_ratio < 2:
                    short_sentiment -= 0.02
                    signals.append(f"Low short interest (days to cover: {short_ratio:.1f})")
            
            if short_percent_float:
                short_pct = short_percent_float * 100 if short_percent_float < 1 else short_percent_float
                
                if short_pct > 20:
                    short_sentiment += 0.1
                    signals.append(f"Very high short interest ({short_pct:.1f}% of float)")
                elif short_pct > 10:
                    short_sentiment += 0.05
                    signals.append(f"High short interest ({short_pct:.1f}% of float)")
                elif short_pct < 3:
                    short_sentiment -= 0.02
                    signals.append(f"Low short interest ({short_pct:.1f}% of float)")
            
            if not short_ratio and not short_percent_float:
                signals.append("Short interest data not available")
            
            short_sentiment = max(0.05, min(0.95, short_sentiment))
            
            return {
                'short_interest_sentiment': round(short_sentiment, 3),
                'sentiment_signals': signals,
                'market_indicators': {
                    'short_ratio': short_ratio,
                    'short_percent_float': round(short_percent_float * 100, 1) if short_percent_float else None
                }
            }
            
        except Exception as e:
            print(f"⚠️ Short interest sentiment analysis error: {str(e)}")
            return {'short_interest_sentiment': 0.5, 'sentiment_signals': [], 'market_indicators': {}} 

    def analyze_stock(self, ticker: str) -> ToolResult:
        """
        Main method to perform comprehensive sentiment analysis on a stock.
        Returns ToolResult with sentiment analysis data.
        """
        start_time = time.time()
        
        try:
            ticker = ticker.upper().strip()
            print(f"🎯 Starting sentiment analysis for {ticker}")
            
            #Check cache first
            cache_key = f"sentiment_{ticker}"
            if cache_key in self.cache:
                cache_data = self.cache[cache_key]
                if time.time() - cache_data['timestamp'] < self.cache_ttl:
                    print(f"📋 Using cached sentiment data for {ticker}")
                    execution_time = time.time() - start_time
                    return ToolResult(
                        tool_name=self.name,
                        success=True,
                        data=cache_data['data'],
                        execution_time_seconds=execution_time
                    )
            
            #Get comprehensive sentiment data from all sources
            sentiment_data = self._get_comprehensive_sentiment_data(ticker)
            
            #Analyze each sentiment category
            analysis_results = {}
            
            #1. News Sentiment Analysis
            if 'news_sentiment' in sentiment_data:
                news_analysis = self._analyze_news_sentiment_category(sentiment_data['news_sentiment'])
                analysis_results['news_sentiment'] = news_analysis
                print(f"✅ News sentiment analyzed")
            
            #2. Social Sentiment Analysis (Reddit + Twitter)
            social_analysis = self._analyze_social_sentiment_category(sentiment_data)
            if social_analysis:
                analysis_results['social_sentiment'] = social_analysis
                print(f"✅ Social sentiment analyzed")
            
            #3. Analyst Sentiment Analysis
            analyst_analysis = self._analyze_analyst_sentiment(ticker)
            if analyst_analysis:
                analysis_results['analyst_sentiment'] = analyst_analysis
                print(f"✅ Analyst sentiment analyzed")
            
            # 4. Market Sentiment Analysis
            market_analysis = self._analyze_market_sentiment(ticker)
            if market_analysis:
                analysis_results['market_sentiment'] = market_analysis
                print(f"✅ Market sentiment analyzed")
            
            #Generate overall sentiment analysis
            overall_analysis = self._generate_overall_sentiment_analysis(analysis_results)
            
            #Prepare final result
            final_result = {
                'ticker': ticker,
                'analysis_timestamp': datetime.now().isoformat(),
                'sentiment_categories': analysis_results,
                'overall_analysis': overall_analysis,
                'data_sources': self._get_data_sources(sentiment_data),
                'analysis_summary': self._generate_sentiment_summary(overall_analysis)
            }
            
            # Cache the result
            self.cache[cache_key] = {
                'data': final_result,
                'timestamp': time.time()
            }
            
            execution_time = time.time() - start_time
            print(f"🎯 Sentiment analysis completed for {ticker} in {execution_time:.2f}s")
            
            return ToolResult(
                tool_name=self.name,
                success=True,
                data=final_result,
                execution_time_seconds=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Sentiment analysis failed for {ticker}: {str(e)}"
            print(f"❌ {error_msg}")
            
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                error_message=error_msg,
                execution_time_seconds=execution_time
            )

    def _analyze_news_sentiment_category(self, news_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process news sentiment data into standardized format"""
        return {
            'score': news_data.get('sentiment_score', 0.5),
            'confidence': news_data.get('confidence', 0.1),
            'signals': [f"News: {signal}" for signal in news_data.get('sentiment_signals', [])],
            'rating': self._get_sentiment_label(news_data.get('sentiment_score', 0.5)),
            'data_quality': {
                'articles_analyzed': news_data.get('news_mentions', 0),
                'sources_covered': len(news_data.get('news_sources', [])),
                'coverage_period': '7 days'
            }
        }

    def _analyze_social_sentiment_category(self, sentiment_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Combine and analyze social media sentiment (Reddit + Twitter)"""
        reddit_data = sentiment_data.get('reddit_sentiment')
        twitter_data = sentiment_data.get('twitter_sentiment')
        
        if not reddit_data and not twitter_data:
            return None
        
        #Combine social sentiment scores with weighting
        reddit_weight = 0.4
        twitter_weight = 0.6  #Twitter gets higher weight due to real-time nature
        
        combined_score = 0.5
        combined_confidence = 0.1
        signals = []
        
        if reddit_data and twitter_data:
            combined_score = (reddit_data['sentiment_score'] * reddit_weight + 
                            twitter_data['sentiment_score'] * twitter_weight)
            combined_confidence = (reddit_data['confidence'] * reddit_weight + 
                                 twitter_data['confidence'] * twitter_weight)
            signals.extend([f"Reddit: {s}" for s in reddit_data.get('sentiment_signals', [])])
            signals.extend([f"Twitter: {s}" for s in twitter_data.get('sentiment_signals', [])])
        elif reddit_data:
            combined_score = reddit_data['sentiment_score']
            combined_confidence = reddit_data['confidence'] * 0.8  # Reduce confidence for single source
            signals.extend([f"Reddit: {s}" for s in reddit_data.get('sentiment_signals', [])])
        elif twitter_data:
            combined_score = twitter_data['sentiment_score']
            combined_confidence = twitter_data['confidence'] * 0.8
            signals.extend([f"Twitter: {s}" for s in twitter_data.get('sentiment_signals', [])])
        
        return {
            'score': round(combined_score, 3),
            'confidence': round(combined_confidence, 3),
            'signals': signals[:10],  #Limit to top 10 signals
            'rating': self._get_sentiment_label(combined_score),
            'data_quality': {
                'reddit_mentions': reddit_data.get('reddit_mentions', 0) if reddit_data else 0,
                'twitter_mentions': twitter_data.get('twitter_mentions', 0) if twitter_data else 0,
                'platforms_analyzed': len([p for p in ['reddit', 'twitter'] if p + '_sentiment' in sentiment_data])
            }
        }

    def _generate_overall_sentiment_analysis(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive overall sentiment analysis"""
        
        #Calculate weighted overall sentiment score
        total_score = 0
        total_weight = 0
        all_signals = []
        strengths = []
        weaknesses = []
        
        for category, weight in self.analysis_categories.items():
            if category in analysis_results:
                analysis = analysis_results[category]
                score = analysis.get('score', 0.5)
                confidence = analysis.get('confidence', 0.1)
                rating = analysis.get('rating', 'Neutral')
                
                #Weight by confidence
                effective_weight = weight * confidence
                total_score += score * effective_weight
                total_weight += effective_weight
                
                #Collect signals
                if 'signals' in analysis:
                    all_signals.extend(analysis['signals'])
                
                #Categorize as strength or weakness
                if score >= 0.7:
                    strengths.append(f"{category.replace('_', ' ').title()}: {rating}")
                elif score < 0.4:
                    weaknesses.append(f"{category.replace('_', ' ').title()}: {rating}")
        
        #Calculate final sentiment score
        final_score = total_score / total_weight if total_weight > 0 else 0.5
        
        #Generate recommendation
        if final_score >= 0.75:
            recommendation = "VERY POSITIVE"
            signal = "BUY"
        elif final_score >= 0.6:
            recommendation = "POSITIVE"
            signal = "BUY"
        elif final_score >= 0.4:
            recommendation = "NEUTRAL"
            signal = "HOLD"
        elif final_score >= 0.25:
            recommendation = "NEGATIVE"
            signal = "SELL"
        else:
            recommendation = "VERY NEGATIVE"
            signal = "SELL"
        
        #Calculate overall confidence
        category_confidences = [analysis_results[cat].get('confidence', 0.1) 
                              for cat in self.analysis_categories.keys() 
                              if cat in analysis_results]
        
        overall_confidence = sum(category_confidences) / len(category_confidences) if category_confidences else 0.1
        
        #Boost confidence based on data source diversity
        data_sources_count = len(analysis_results)
        confidence_boost = min((data_sources_count / 4) * 0.2, 0.2)  #Max 0.2 boost for all 4 categories
        final_confidence = min(overall_confidence + confidence_boost, 0.95)
        
        return {
            'sentiment_score': round(final_score, 3),
            'sentiment_rating': recommendation,
            'signal': signal,
            'confidence': round(final_confidence, 3),
            'strengths': strengths,
            'weaknesses': weaknesses,
            'key_signals': all_signals[:15],  #Top 15 signals
            'categories_analyzed': list(analysis_results.keys()),
            'sentiment_breakdown': {
                category: {
                    'score': analysis_results[category].get('score', 0.5),
                    'weight': self.analysis_categories[category],
                    'rating': analysis_results[category].get('rating', 'Neutral')
                }
                for category in analysis_results.keys()
            }
        }

    def _generate_sentiment_summary(self, overall_analysis: Dict[str, Any]) -> str:
        """Generate human-readable sentiment summary"""
        sentiment_rating = overall_analysis.get('sentiment_rating', 'NEUTRAL')
        confidence = overall_analysis.get('confidence', 0.5)
        signal = overall_analysis.get('signal', 'HOLD')
        
        summary_parts = []
        
        #Main sentiment
        summary_parts.append(f"Overall sentiment is {sentiment_rating.lower()}")
        
        #Signal
        summary_parts.append(f"suggesting a {signal} signal")
        
        #Confidence
        if confidence >= 0.7:
            confidence_desc = "high confidence"
        elif confidence >= 0.5:
            confidence_desc = "moderate confidence"
        else:
            confidence_desc = "low confidence"
        
        summary_parts.append(f"with {confidence_desc} ({confidence:.1%})")
        
        #Key strengths/weaknesses
        strengths = overall_analysis.get('strengths', [])
        weaknesses = overall_analysis.get('weaknesses', [])
        
        if strengths:
            summary_parts.append(f"Key strengths: {', '.join(strengths[:2])}")
        
        if weaknesses:
            summary_parts.append(f"Key concerns: {', '.join(weaknesses[:2])}")
        
        return ". ".join(summary_parts) + "."

    def _get_data_sources(self, sentiment_data: Dict[str, Any]) -> List[str]:
        """Extract list of data sources used in analysis"""
        sources = []
        
        if 'reddit_sentiment' in sentiment_data:
            sources.append('Reddit')
        if 'news_sentiment' in sentiment_data:
            sources.append('Financial News')
        if 'twitter_sentiment' in sentiment_data:
            sources.append('Twitter')
        
        #Always include these as they use existing data
        sources.extend(['Analyst Ratings', 'Technical Indicators'])
        
        return sources

# Create global instance
sentiment_analysis_tool = SentimentAnalysisTool()