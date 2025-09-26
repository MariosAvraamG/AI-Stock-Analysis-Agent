import time
from typing import Dict, Any, List
from datetime import datetime
from app.models.schemas import ToolResult
from app.services.tools.multi_source_data import multi_source_tool


class FundamentalAnalysisTool:
    """
    Comprehensive fundamental analysis tool for investment decision making.
    Analyzes financial health, valuation, growth, profitability, and competitive position.
    """
    
    def __init__(self):
        self.name = "fundamental_analysis"
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour cache (fundamental data changes less frequently)
        
        # Fundamental analysis categories and their weights
        self.analysis_categories = {
            'valuation': 0.25,      # P/E, P/B, P/S, PEG ratios
            'financial_health': 0.25,  # Debt ratios, current ratio, ROE, ROA
            'growth': 0.20,         # Revenue growth, earnings growth
            'profitability': 0.15,  # Margins
            'dividends': 0.10,  # Yield, payout ratio
            'market_position': 0.05     # Market cap, Beta
        }

    def _get_comprehensive_fundamental_data(self, ticker: str) -> Dict[str, Any]:
        """Get comprehensive fundamental data using multi-source approach"""
        print(f"🔍 Fetching fundamental data for analysis: {ticker}")
        
        # Use multi-source data tool for reliability
        result = multi_source_tool.get_fundamental_data(ticker)
        
        if not result.success:
            raise ValueError(f"Failed to fetch fundamental data: {result.error_message}")
        
        # Extract the data
        fundamental_data = result.data
        if not fundamental_data:
            raise ValueError(f"No fundamental data found for {ticker}")
        
        # Log data source
        source = result.data.get('primary_source', 'unknown')
        print(f"📊 Using fundamental data from {source}")
        
        return fundamental_data
    
    def analyze_stock(self, ticker: str) -> ToolResult:
        """Perform comprehensive fundamental analysis"""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"fa_{ticker.upper()}"
            if cache_key in self.cache:
                cached_result, cached_time = self.cache[cache_key]
                if time.time() - cached_time < self.cache_ttl:
                    print(f"📦 Using cached fundamental analysis for {ticker}")
                    return cached_result
            
            print(f"🔍 Performing fundamental analysis for {ticker}")
            
            # Get comprehensive fundamental data using multi-source approach
            fundamental_data = self._get_comprehensive_fundamental_data(ticker)
            
            # Perform analysis
            analysis_results = {
                'valuation_analysis': self._analyze_valuation(fundamental_data),
                'financial_health_analysis': self._analyze_financial_health(fundamental_data),
                'growth_analysis': self._analyze_growth(fundamental_data),
                'profitability_analysis': self._analyze_profitability(fundamental_data),
                'dividends_analysis': self._analyze_dividends(fundamental_data),
                'market_position_analysis': self._analyze_market_position(fundamental_data)
            }
            
            # Generate overall fundamental score and recommendation
            overall_analysis = self._generate_overall_analysis(analysis_results)
            
            # Prepare result data
            result_data = {
                'ticker': ticker.upper(),
                'company_info': {
                    'name': fundamental_data.get('company_name'),
                    'sector': fundamental_data.get('sector'),
                    'industry': fundamental_data.get('industry'),
                    'market_cap': fundamental_data.get('market_cap'),
                    'employees': fundamental_data.get('employees')
                },
                'fundamental_data': fundamental_data,
                'analysis_results': analysis_results,
                'overall_analysis': overall_analysis,
                'last_updated': datetime.now().isoformat(),
                'data_source': fundamental_data.get('primary_source', 'unknown')
            }
            
            # Create successful result
            result = ToolResult(
                tool_name=self.name,
                success=True,
                data=result_data,
                execution_time_seconds=time.time() - start_time
            )
            
            # Cache successful result
            self.cache[cache_key] = (result, time.time())
            
            print(f"✅ Fundamental analysis completed for {ticker}")
            return result
            
        except Exception as e:
            print(f"❌ Error in fundamental analysis for {ticker}: {str(e)}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                error_message=str(e),
                execution_time_seconds=time.time() - start_time
            )
    
    def _analyze_valuation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze valuation metrics"""
        valuation_score = 0
        max_score = 0
        signals = []

        #P/E Ratio Analysis
        pe_ratio = data.get('pe_ratio')
        if pe_ratio:
            max_score += 20
            if pe_ratio < 15:
                valuation_score += 20
                signals.append("Low P/E ratio indicates potential undervaluation")
            elif pe_ratio < 25:
                valuation_score += 15
                signals.append("Moderate P/E ratio suggests fair valuation")
            elif pe_ratio < 35:
                valuation_score += 10
                signals.append("High P/E ratio - growth expected")
            else:
                signals.append("High P/E ratio - potential overvaluation")
        
        #PEG Ratio Analysis
        peg_ratio = data.get('peg_ratio')
        if peg_ratio:
            max_score += 15
            if peg_ratio < 1.0:
                valuation_score += 15
                signals.append("PEG ratio < 1.0 suggests undervaluation")
            elif peg_ratio < 1.5:
                valuation_score += 12
                signals.append("Reasonable PEG ratio")
            else:
                valuation_score += 5
                signals.append("High PEG ratio - growth may not justify price")
        
        #P/B Ratio Analysis
        pb_ratio = data.get('pb_ratio')
        if pb_ratio:
            max_score += 10
            if pb_ratio < 1.0:
                valuation_score += 10
                signals.append("Trading below book value")
            elif pb_ratio < 3.0:
                valuation_score += 8
                signals.append("Reasonable P/B ratio")
            else:
                valuation_score += 3
                signals.append("High P/B ratio - potential overvaluation")

        #P/S Ratio Analysis
        ps_ratio = data.get('ps_ratio')
        if ps_ratio:
            max_score += 10
            if ps_ratio < 2.0:
                valuation_score += 10
                signals.append("Low P/S ratio - potential undervaluation")
            elif ps_ratio < 5.0:
                valuation_score += 7
                signals.append("Moderate P/S ratio")
            else:
                valuation_score += 3
                signals.append("High P/S ratio - potential overvaluation")
                
        #Calculate final score
        final_score = (valuation_score / max_score) if max_score > 0 else 0.5

        #Determine rating
        if final_score >= 0.8:
            rating = "UNDERVALUED"
        elif final_score >= 0.6:
            rating = "FAIRLY VALUED"
        elif final_score >= 0.4:
            rating = "SLIGHTLY OVERVALUED"
        else:
            rating = "OVERVALUED"
            
        return {
            'score': final_score,
            'rating': rating,
            'metrics': {
                'pe_ratio': pe_ratio,
                'peg_ratio': peg_ratio,
                'pb_ratio': pb_ratio,
                'ps_ratio': ps_ratio,
                'ev_revenue': data.get('ev_revenue'),
                'ev_ebitda': data.get('ev_ebitda')
            },
            'signals': signals
        }

    def _analyze_financial_health(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze financial health metrics"""
        health_score = 0
        max_score = 0
        signals = []

        #Debt-to-Equity Analysis
        debt_to_equity = data.get('debt_to_equity')
        if debt_to_equity:
            max_score += 20
            if debt_to_equity < 0.3:
                health_score += 20
                signals.append("Very Low debt-to-equity ratio - strong balance sheet")
            elif debt_to_equity < 0.6:
                health_score += 15
                signals.append("Moderate debt levels")
            elif debt_to_equity < 1.0:
                health_score += 10
                signals.append("Higher debt levels - monitor closely")
            else:
                health_score += 5
                signals.append("High debt levels - potential concern")
        
        #Current Ratio Analysis
        current_ratio = data.get('current_ratio')
        if current_ratio:
            max_score += 15
            if current_ratio > 2.0:
                health_score += 15
                signals.append("Strong liquidity position")
            elif current_ratio > 1.5:
                health_score += 12
                signals.append("Good liquidity position")
            elif current_ratio > 1.0:
                health_score += 8
                signals.append("Adequate liquidity")
            else:
                health_score += 3
                signals.append("Liquidity concerns")

        #ROE Analysis
        roe = data.get('roe')
        if roe:
            max_score += 15
            if roe > 0.20:
                health_score += 15
                signals.append("Excellent return on equity")
            elif roe > 0.15:
                health_score += 12
                signals.append("Good return on equity")
            elif roe > 0.10:
                health_score += 8
                signals.append("Average return on equity")
            else:
                health_score += 5
                signals.append("Below average return on equity")
        
        #ROA Analysis
        roa = data.get('roa')
        if roa:
            max_score += 10
            if roa > 0.10:
                health_score += 10
                signals.append("Excellent asset utilization")
            elif roa > 0.05:
                health_score += 8
                signals.append("Good asset utilization")
            else:
                health_score += 5
                signals.append("Average asset utilization")

        #Calculate final score
        final_score = (health_score / max_score) if max_score > 0 else 0.5

        #Determine rating
        if final_score >= 0.8:
            rating = "EXCELLENT"
        elif final_score >= 0.6:
            rating = "GOOD"
        elif final_score >= 0.4:
            rating = "AVERAGE"
        else:
            rating = "WEAK"

        return {
            'score': final_score,
            'rating': rating,
            'metrics': {
                'debt_to_equity': debt_to_equity,
                'current_ratio': current_ratio,
                'quick_ratio': data.get('quick_ratio'),
                'roe': roe,
                'roa': roa,
                'free_cash_flow': data.get('free_cash_flow'),
                'operating_cash_flow': data.get('operating_cash_flow')
            },
            'signals': signals
        }

    def _analyze_growth(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze growth metrics"""
        growth_score = 0
        max_score = 0
        signals = []

        #Revenue Growth Analysis
        revenue_growth = data.get('revenue_growth') or data.get('revenue_growth_yoy')
        if revenue_growth:
            max_score += 25
            if revenue_growth > 0.20:
                growth_score += 25
                signals.append("Strong revenue growth")
            elif revenue_growth > 0.10:
                growth_score += 20
                signals.append("Good revenue growth")
            elif revenue_growth > 0.05:
                growth_score += 15
                signals.append("Moderate revenue growth")
            elif revenue_growth > 0:
                growth_score += 10
                signals.append("Slow revenue growth")
            else:
                signals.append("Declining revenue")
            
        #Earnings Growth Analysis
        earnings_growth = data.get('earnings_growth') or data.get('earnings_growth_yoy')
        if earnings_growth:
            max_score += 25
            if earnings_growth > 0.25:
                growth_score += 25
                signals.append("Excellent earnings growth")
            elif earnings_growth > 0.15:
                growth_score += 20
                signals.append("Strong earnings growth")
            elif earnings_growth > 0.10:
                growth_score += 15
                signals.append("Good earnings growth")
            elif earnings_growth > 0:
                growth_score += 10
                signals.append("Modest earnings growth")
            else:
                signals.append("Declining earnings")

        #Calculate final score\
        final_score = (growth_score / max_score) if max_score > 0 else 0.5

        #Determine rating
        if final_score >= 0.8:
            rating = "HIGH GROWTH"
        elif final_score >= 0.6:
            rating = "MODERATE GROWTH"
        elif final_score >= 0.4:
            rating = "SLOW GROWTH"
        else:
            rating = "NO GROWTH"

        return {
            'score': final_score,
            'rating': rating,
            'metrics': {
                'revenue_growth': revenue_growth,
                'earnings_growth': earnings_growth,
                'quarterly_revenue_growth': data.get('quarterly_revenue_growth'),
                'quarterly_earnings_growth': data.get('quarterly_earnings_growth')
            },
            'signals': signals
        }

    def _analyze_profitability(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze profitability metrics"""
        profit_score = 0
        max_score = 0
        signals = []

        #Profit Margin Analysis
        profit_margin = data.get('profit_margin')
        if profit_margin:
            max_score += 25
            if profit_margin > 0.20:
                profit_score += 25
                signals.append("Excellent profit margins")
            elif profit_margin > 0.15:
                profit_score += 20
                signals.append("Very good profit margins")
            elif profit_margin > 0.10:
                profit_score += 15
                signals.append("Good profit margins")
            elif profit_margin > 0.05:
                profit_score += 8
                signals.append("Average profit margins")
            elif profit_margin > 0.02:
                profit_score += 4
                signals.append("Below average profit margins")
            else:
                profit_score += 1
                signals.append("Low profit margins")
        
        #Operating Margin Analysis
        operating_margin = data.get('operating_margin')
        if operating_margin:
            max_score += 20
            if operating_margin > 0.25:
                profit_score += 20
                signals.append("Excellent operational efficiency")
            elif operating_margin > 0.20:
                profit_score += 16
                signals.append("Strong operational efficiency")
            elif operating_margin > 0.15:
                profit_score += 12
                signals.append("Good operational efficiency")
            elif operating_margin > 0.10:
                profit_score += 8
                signals.append("Average operational efficiency")
            elif operating_margin > 0.05:
                profit_score += 4
                signals.append("Below average operational efficiency")
            else:
                profit_score += 1
                signals.append("Poor operational efficiency")
            
        #Gross Margin Analysis
        gross_margin = data.get('gross_margin')
        if gross_margin:
            max_score += 15
            if gross_margin > 0.50:
                profit_score += 15
                signals.append("Excellent gross margins")
            elif gross_margin > 0.40:
                profit_score += 12
                signals.append("High gross margins")
            elif gross_margin > 0.30:
                profit_score += 9
                signals.append("Good gross margins")
            elif gross_margin > 0.20:
                profit_score += 6
                signals.append("Average gross margins")
            elif gross_margin > 0.10:
                profit_score += 3
                signals.append("Below average gross margins")
            else:
                profit_score += 1
                signals.append("Low gross margins")

        # Calculate final score
        final_score = (profit_score / max_score) if max_score > 0 else 0.5
        
        # Determine rating
        if final_score >= 0.80:
            rating = "HIGHLY PROFITABLE"
        elif final_score >= 0.65:
            rating = "PROFITABLE"
        elif final_score >= 0.45:
            rating = "MODERATELY PROFITABLE"
        elif final_score >= 0.25:
            rating = "BELOW AVERAGE PROFITABILITY"
        else:
            rating = "LOW PROFITABILITY"
        
        return {
            'rating': rating,
            'score': final_score,
            'metrics': {
                'profit_margin': profit_margin,
                'operating_margin': operating_margin,
                'gross_margin': gross_margin,
                'roe': data.get('roe'),
                'roa': data.get('roa')
            },
            'signals': signals
        }

    def _analyze_dividends(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dividend metrics with sustainability focus"""
        dividend_score = 0
        max_score = 0
        signals = []
        
        #Dividend Yield Analysis
        dividend_yield = data.get('dividend_yield')
        if dividend_yield and dividend_yield > 0:
            max_score += 25
            if dividend_yield > 0.10:  #Suspiciously high yield (>10%)
                dividend_score += 5
                signals.append("Very high dividend yield - potential red flag")
            elif dividend_yield > 0.08:  #High yield (8-10%)
                dividend_score += 10
                signals.append("High dividend yield - investigate sustainability")
            elif dividend_yield > 0.05:  #Good yield (5-8%)
                dividend_score += 20
                signals.append("Attractive dividend yield")
            elif dividend_yield > 0.03:  #Moderate yield (3-5%)
                dividend_score += 18
                signals.append("Good dividend yield")
            elif dividend_yield > 0.02:  #Low-moderate yield (2-3%)
                dividend_score += 15
                signals.append("Moderate dividend yield")
            else:  #Low yield (<2%)
                dividend_score += 10
                signals.append("Low dividend yield")
        else:
            signals.append("No dividend paid")
            return {
                'rating': "NO DIVIDEND",
                'score': 0,
                'metrics': {
                    'dividend_yield': 0,
                    'dividend_rate': 0,
                    'payout_ratio': None
                },
                'signals': signals
            }
        
        #Dividend Payout Ratio Analysis
        payout_ratio = data.get('payout_ratio')
        if payout_ratio:
            max_score += 35
            if 0.30 <= payout_ratio <= 0.50:  #Ideal sustainable range
                dividend_score += 35
                signals.append("Excellent dividend payout ratio")
            elif 0.50 < payout_ratio <= 0.65:  #Good range
                dividend_score += 30
                signals.append("Good dividend payout ratio")
            elif payout_ratio < 0.30:  #Conservative
                dividend_score += 25
                signals.append("Conservative dividend payout ratio - room for growth")
            elif 0.65 < payout_ratio <= 0.80:  #Concerning
                dividend_score += 15
                signals.append("High dividend payout ratio - monitor sustainability")
            elif 0.80 < payout_ratio <= 0.95:  #Very concerning
                dividend_score += 8
                signals.append("Very high dividend payout ratio - sustainability risk")
            else:  #Unsustainable (>95%)
                dividend_score += 3
                signals.append("Unsustainable dividend payout ratio - dividend cut risk")
        
        #Calculate final score
        final_score = (dividend_score / max_score) if max_score > 0 else 0
        
        # Determine rating
        if final_score >= 0.85:
            rating = "EXCELLENT DIVIDEND"
        elif final_score >= 0.70:
            rating = "GOOD DIVIDEND"
        elif final_score >= 0.50:
            rating = "AVERAGE DIVIDEND"
        elif final_score >= 0.30:
            rating = "BELOW AVERAGE DIVIDEND"
        else:
            rating = "POOR DIVIDEND"
        
        return {
            'rating': rating,
            'score': final_score,
            'metrics': {
                'dividend_yield': dividend_yield,
                'dividend_rate': data.get('dividend_rate') or data.get('dividend_per_share'),
                'payout_ratio': payout_ratio,
                'five_year_avg_dividend_yield': data.get('five_year_avg_dividend_yield')
            },
            'signals': signals
        }

    def _analyze_market_position(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market position and competitive factors"""
        position_score = 0
        max_score = 0
        signals = []
        
        #Market Cap Analysis
        market_cap = data.get('market_cap')
        if market_cap:
            max_score += 25
            if market_cap > 200_000_000_000:  # $200B+
                position_score += 25
                signals.append("Mega-cap stock - highly stable, market leader")
            elif market_cap > 10_000_000_000:  # $10B - $200B
                position_score += 20
                signals.append("Large-cap stock - stable with growth potential")
            elif market_cap > 2_000_000_000:  # $2B - $10B
                position_score += 15
                signals.append("Mid-cap stock - balanced growth and risk")
            elif market_cap > 300_000_000:  # $300M - $2B
                position_score += 10
                signals.append("Small-cap stock - higher growth/risk")
            else:  # <$300M
                position_score += 5
                signals.append("Micro-cap stock - very high risk/reward")

                
        #Beta Analysis (volatility relative to market)
        beta = data.get('beta')
        if beta:
            max_score += 15
            if 0.8 <= beta <= 1.2:
                position_score += 15
                signals.append("Market-level volatility")
            elif beta < 0.8:
                position_score += 12
                signals.append("Lower volatility than market")
            else:
                position_score += 10
                signals.append("Higher volatility than market")
        
        #Calculate final score
        final_score = (position_score / max_score) if max_score > 0 else 0.5
        
        #Determine rating
        if final_score >= 0.8:
            rating = "STRONG MARKET POSITION"
        elif final_score >= 0.6:
            rating = "GOOD MARKET POSITION"
        elif final_score >= 0.4:
            rating = "AVERAGE MARKET POSITION"
        else:
            rating = "WEAK MARKET POSITION"
            
        return {
            'rating': rating,
            'score': final_score,
            'metrics': {
                'market_cap': market_cap,
                'beta': beta,
                'sector': data.get('sector'),
                'industry': data.get('industry')
            },
            'signals': signals
        }

    def _generate_overall_analysis(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall fundamental analysis and investment recommendation"""

        #Calculate weighted score
        total_score = 0
        total_weight = 0
        category_scores = {}

        #Collect all signals
        all_signals = []
        strengths = []
        weaknesses = []

        for category, weight in self.analysis_categories.items():
            analysis_key = f"{category}_analysis"
            if analysis_key in analysis_results:
                analysis = analysis_results[analysis_key]

                #Calculate score and add to category scores
                score = analysis.get('score', 0.5)
                category_scores[category] = score
                total_score += score * weight
                total_weight += weight

                #Collect signals from the full analysis
                if 'signals' in analysis:
                    all_signals.append([f"{category.replace('_', ' ').title()}: {signal}" 
                                      for signal in analysis['signals']])

                #Categorize as strength or weakness based on score
                rating = analysis.get('rating', 'UNKNOWN')
                if score >= 0.7:
                    strengths.append(f"{category.replace('_', ' ').title()}: {rating}")
                elif score < 0.4:
                    weaknesses.append(f"{category.replace('_', ' ').title()}: {rating}")
        
        #Calculate overall score
        overall_score = (total_score / total_weight) if total_weight > 0 else 0.5
        
        #Determine overall rating
        if overall_score >= 0.8:
            recommendation = "STRONG BUY"
            confidence = "HIGH"
        elif overall_score >= 0.65:
            recommendation = "BUY"
            confidence = "MODERATE"
        elif overall_score >= 0.55:
            recommendation = "HOLD"
            confidence = "LOW"
        elif overall_score >= 0.4:
            recommendation = "WEAK HOLD"
            confidence = "LOW"
        else:
            recommendation = "SELL"
            confidence = "MODERATE"

        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'overall_score': round(overall_score, 2),
            'category_scores': category_scores,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'key_signals': all_signals,
            'investment_thesis': self._generate_investment_thesis(
                recommendation, analysis_results, strengths, weaknesses
            )
        }

    def _generate_investment_thesis(self, recommendation: str, analysis_results: Dict[str, Any], strengths: List[str], weaknesses: List[str]) -> str:
        """Generate investment thesis"""

        thesis_parts = []

        #Opening based on recommendation
        if recommendation in ["STRONG BUY", "BUY"]:
            thesis_parts.append("This stock presents a compelling investment opportunity.")
        elif recommendation == "HOLD":
            thesis_parts.append("This stock appears fairly valued with mixed fundamentals.")
        else:
            thesis_parts.append("This stock faces fundamental challenges.")

        #Add Strengths
        if strengths:
           thesis_parts.append(f"Strengths: {', '.join(strengths).lower()}.")
        
        #Add Weaknesses
        if weaknesses:
            thesis_parts.append(f"Weaknesses: {', '.join(weaknesses).lower()}.")
        
        #Add specific insights from analysis
        valuation = analysis_results.get('valuation_analysis', {})
        growth = analysis_results.get('growth_analysis', {})

        if valuation.get('rating') == 'UNDERVALUED':
            thesis_parts.append("The stock appears undervalued based on key metrics.")
        elif valuation.get('rating') == 'OVERVALUED':
            thesis_parts.append("Valuation metrics suggest the stock may be overpriced.")
        
        if growth.get('rating') in ['HIGH GROWTH', 'MODERATE GROWTH']:
            thesis_parts.append("Growth prospects appear favorable.")
        elif growth.get('rating') == 'NO GROWTH':
            thesis_parts.append("Growth prospects are limited.")
        
        return " ".join(thesis_parts)
        



# Global instance
fundamental_analysis_tool = FundamentalAnalysisTool()