import time
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool, StructuredTool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from app.core.config import settings

#Import analysis tools
from app.services.tools.technical_analysis import technical_analysis_tool
from app.services.tools.fundamental_analysis import fundamental_analysis_tool
from app.services.tools.sentiment_analysis import sentiment_analysis_tool
from app.services.tools.ml_prediction import ml_prediction_tool

class StockAnalysisAgent:
    """
    Comprehensive AI Stock Analysis Agent using LangChain.
    
    Orchestrates multiple specialized analysis tools:
    - Technical Analysis
    - Fundamental Analysis  
    - Sentiment Analysis
    - ML Predictions
    
    Provides intelligent, context-aware stock analysis and recommendations.
    """
    
    def __init__(self):
        self.name = "stock_analysis_agent"
        
        #Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,  #Low temperature for consistent analysis
            openai_api_key=settings.openai_api_key
        )
        
        #Initialize memory for conversation context
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        #Create tools
        self.tools = self._create_analysis_tools()
        
        #Create agent
        self.agent_executor = self._create_agent_executor()
        
        #Analysis cache
        self.cache = {}
        self.cache_ttl = 1800  #30 minutes
        
        print("✅ Stock Analysis Agent initialized successfully")

    def _create_analysis_tools(self) -> List[BaseTool]:
        """Create LangChain tools from analysis services"""
        
        def technical_analysis(ticker: str, period: str = "1y") -> str:
            """
            Perform comprehensive technical analysis on a stock.
            
            Args:
                ticker: Stock symbol (e.g., 'AAPL', 'GOOGL')
                period: Analysis period ('1y', '6mo', '3mo', '1mo')
                
            Returns:
                JSON string with technical analysis results including indicators,
                signals, trend analysis, and trading recommendations.
            """
            try:
                result = technical_analysis_tool.analyze_stock(ticker, period)
                if result.success:
                    return json.dumps({
                        "success": True,
                        "tool": "technical_analysis",
                        "ticker": ticker,
                        "data": result.data,
                        "execution_time": result.execution_time_seconds
                    }, indent=2, default=str)
                else:
                    return json.dumps({
                        "success": False,
                        "tool": "technical_analysis",
                        "error": result.error_message
                    })
            except Exception as e:
                return json.dumps({
                    "success": False,
                    "tool": "technical_analysis",
                    "error": str(e)
                })
        
        def fundamental_analysis(ticker: str) -> str:
            """
            Perform comprehensive fundamental analysis on a stock.
            
            Args:
                ticker: Stock symbol (e.g., 'AAPL', 'GOOGL')
                
            Returns:
                JSON string with fundamental analysis results including valuation,
                financial health, growth metrics, profitability, and investment rating.
            """
            try:
                result = fundamental_analysis_tool.analyze_stock(ticker)
                if result.success:
                    return json.dumps({
                        "success": True,
                        "tool": "fundamental_analysis",
                        "ticker": ticker,
                        "data": result.data,
                        "execution_time": result.execution_time_seconds
                    }, indent=2, default=str)
                else:
                    return json.dumps({
                        "success": False,
                        "tool": "fundamental_analysis",
                        "error": result.error_message
                    })
            except Exception as e:
                return json.dumps({
                    "success": False,
                    "tool": "fundamental_analysis",
                    "error": str(e)
                })
        
        def sentiment_analysis(ticker: str) -> str:
            """
            Perform comprehensive sentiment analysis on a stock.
            
            Args:
                ticker: Stock symbol (e.g., 'AAPL', 'GOOGL')
                
            Returns:
                JSON string with sentiment analysis results including news sentiment,
                social media sentiment, analyst sentiment, and overall market sentiment.
            """
            try:
                result = sentiment_analysis_tool.analyze_stock(ticker)
                if result.success:
                    return json.dumps({
                        "success": True,
                        "tool": "sentiment_analysis",
                        "ticker": ticker,
                        "data": result.data,
                        "execution_time": result.execution_time_seconds
                    }, indent=2, default=str)
                else:
                    return json.dumps({
                        "success": False,
                        "tool": "sentiment_analysis",
                        "error": result.error_message
                    })
            except Exception as e:
                return json.dumps({
                    "success": False,
                    "tool": "sentiment_analysis",
                    "error": str(e)
                })
        
        def ml_prediction(ticker: str) -> str:
            """
            Perform ML-based stock price predictions using ensemble models.
            
            Args:
                ticker: Stock symbol (e.g., 'AAPL', 'GOOGL')
                
            Returns:
                JSON string with ML prediction results including short-term,
                medium-term, and long-term predictions with confidence scores.
            """
            try:
                result = ml_prediction_tool.analyze_stock(ticker)
                if result.success:
                    return json.dumps({
                        "success": True,
                        "tool": "ml_prediction",
                        "ticker": ticker,
                        "data": result.data,
                        "execution_time": result.execution_time_seconds
                    }, indent=2, default=str)
                else:
                    return json.dumps({
                        "success": False,
                        "tool": "ml_prediction",
                        "error": result.error_message
                    })
            except Exception as e:
                return json.dumps({
                    "success": False,
                    "tool": "ml_prediction",
                    "error": str(e)
                })
        
        #Create structured tools
        tools = [
            StructuredTool.from_function(
                func=technical_analysis,
                name="technical_analysis",
                description="Analyze stock using technical indicators, chart patterns, and price trends"
            ),
            StructuredTool.from_function(
                func=fundamental_analysis,
                name="fundamental_analysis", 
                description="Analyze stock using financial metrics, valuation ratios, and company fundamentals"
            ),
            StructuredTool.from_function(
                func=sentiment_analysis,
                name="sentiment_analysis",
                description="Analyze stock sentiment from news, social media, and analyst opinions"
            ),
            StructuredTool.from_function(
                func=ml_prediction,
                name="ml_prediction",
                description="Generate ML-based price predictions using ensemble models"
            )
        ]
        
        return tools

    def _create_agent_executor(self) -> AgentExecutor:
        """Create the LangChain agent executor"""
        
        #Create system prompt
        system_prompt = """You are a specialized trading signal AI agent with access to multiple analysis tools.

Your EXCLUSIVE role is to generate structured trading signals for THREE timeframes:
- SHORT TERM (5-20 days): Technical momentum, price action, short-term sentiment, ML predictions
- MEDIUM TERM (20-60 days): Trend analysis, technical patterns, market sentiment, ML predictions
- LONG TERM (60+ days): Fundamental value, long-term trends, ML predictions

MANDATORY TOOL USAGE:
- Use technical_analysis for ALL timeframes to get trend analysis and overall signals
- Use ml_prediction for ALL timeframes to get ML-based predictions with confidence
- Use fundamental_analysis for LONG TERM signals to assess intrinsic value
- Use sentiment_analysis for SHORT and MEDIUM term signals to gauge market mood

SIGNAL GENERATION RULES:
1. ALWAYS call ALL relevant analysis tools for comprehensive data
2. Extract specific signals (BUY/SELL/HOLD) for each timeframe
3. Assign confidence levels (0.0-1.0) based on tool agreement and data quality
4. Provide brief reasoning for each signal based on tool outputs
5. Ensure signals are timeframe-appropriate (short=momentum, medium=trends, long=fundamentals)

OUTPUT FORMAT:
Structure your response with clear sections for each timeframe. Use this format:

SHORT TERM (5-20 days): 
Signal: BUY/SELL/HOLD
Confidence: 0.XX (as decimal, e.g., 0.85 for 85%)
Reasoning: [Brief explanation based on technical momentum, sentiment and ML predictions]

MEDIUM TERM (20-60 days):
Signal: BUY/SELL/HOLD  
Confidence: 0.XX
Reasoning: [Brief explanation based on trends, market sentiment and ML predictions]

LONG TERM (60+ days):
Signal: BUY/SELL/HOLD
Confidence: 0.XX  
Reasoning: [Brief explanation based on fundamentals and ML predictions]

Remember: You are generating trading signals for educational purposes only. Focus on data-driven analysis."""

        #Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        #Create agent
        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        #Create agent executor
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=10,
            early_stopping_method="generate"
        )

    def get_trading_signals(self, ticker: str) -> Dict[str, Any]:
        """
        Generate trading signals for 3 timeframes using comprehensive analysis.
        
        Args:
            ticker: Stock symbol to analyze
            
        Returns:
            Dictionary with structured signals for short, medium, and long term timeframes
        """
        start_time = time.time()
        
        try:
            ticker = ticker.upper().strip()
            print(f"🤖 Generating trading signals for {ticker}")
            
            #Check cache
            cache_key = f"signals_{ticker}"
            if cache_key in self.cache:
                cache_data = self.cache[cache_key]
                if time.time() - cache_data['timestamp'] < self.cache_ttl:
                    print(f"📋 Using cached trading signals for {ticker}")
                    return cache_data['data']
            
            #Construct specialized prompt for signal generation
            input_text = f"""Generate comprehensive trading signals for {ticker} for three timeframes:

SHORT TERM (5-20 days): Focus on technical momentum, immediate price action, and short-term sentiment
MEDIUM TERM (20-60 days): Focus on trend analysis, technical patterns, and market sentiment  
LONG TERM (60+ days): Focus on fundamental value, long-term trends, and ML predictions

For each timeframe, provide:
1. Signal: BUY, SELL, or HOLD
2. Confidence level: 0.0 to 1.0 (higher = more confident)
3. Key reasoning: Brief explanation of the signal

Use all available analysis tools to make informed decisions. Ensure signals are consistent with the timeframe characteristics."""
            
            #Run agent
            print(f"🔄 Running signal generation agent...")
            response = self.agent_executor.invoke({"input": input_text})
            
            # Debug intermediate steps
            intermediate_steps = response.get("intermediate_steps", [])
            print(f"🔍 Debug: Found {len(intermediate_steps)} intermediate steps")
            
            #Extract and structure the signals from agent response
            structured_signals = self._extract_structured_signals(response.get("output", ""), intermediate_steps)
            
            # Process response
            result = {
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
                "signals": structured_signals,
                "agent_response": response.get("output", ""),
                "execution_time_seconds": time.time() - start_time,
                "tools_used": self._extract_tools_used(intermediate_steps),
                "success": True
            }
            
            # Cache result
            self.cache[cache_key] = {
                'data': result,
                'timestamp': time.time()
            }
            
            print(f"✅ Trading signals generated for {ticker} in {result['execution_time_seconds']:.2f}s")
            return result
            
        except Exception as e:
            error_result = {
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": str(e),
                "execution_time_seconds": time.time() - start_time
            }
            print(f"❌ Signal generation failed for {ticker}: {str(e)}")
            return error_result


    def _process_intermediate_steps(self, intermediate_steps: List) -> List[Dict[str, Any]]:
        """Process and format intermediate steps from agent execution"""
        processed_steps = []
        
        for step in intermediate_steps:
            if isinstance(step, tuple) and len(step) == 2:
                action, observation = step
                
                step_info = {
                    "tool": action.tool if hasattr(action, 'tool') else "unknown",
                    "tool_input": action.tool_input if hasattr(action, 'tool_input') else {},
                    "observation_preview": str(observation)[:200] + "..." if len(str(observation)) > 200 else str(observation),
                    "timestamp": datetime.now().isoformat()
                }
                processed_steps.append(step_info)
        
        return processed_steps

    def _extract_tools_used(self, intermediate_steps: List) -> List[str]:
        """Extract list of tools used during analysis"""
        tools_used = []
        
        for step in intermediate_steps:
            if isinstance(step, tuple) and len(step) == 2:
                action, _ = step
                if hasattr(action, 'tool') and action.tool not in tools_used:
                    tools_used.append(action.tool)
        
        return tools_used

    def _extract_structured_signals(self, agent_response: str, intermediate_steps: List) -> Dict[str, Any]:
        """Extract structured trading signals from agent response and tool outputs"""
        
        #Initialize default signal structure
        signals = {
            "short_term": {
                "signal": "HOLD",
                "confidence": 0.5,
                "reasoning": "Insufficient data for signal generation"
            },
            "medium_term": {
                "signal": "HOLD", 
                "confidence": 0.5,
                "reasoning": "Insufficient data for signal generation"
            },
            "long_term": {
                "signal": "HOLD",
                "confidence": 0.5,
                "reasoning": "Insufficient data for signal generation"
            }
        }
        
        try:
            #First, try to parse the agent's structured response
            print(f"🔍 Debug: Attempting to parse agent response (length: {len(agent_response)})")
            signals_from_agent = self._parse_agent_signals(agent_response)
            if signals_from_agent:
                print("✅ Successfully extracted signals from agent response")
                return signals_from_agent
            
            print("⚠️ Could not parse agent response, falling back to tool data extraction")
            print(f"🔍 Debug: Have {len(intermediate_steps)} intermediate steps to process")
            
            #Extract data from tool results
            tool_data = {}
            for step in intermediate_steps:
                if isinstance(step, tuple) and len(step) == 2:
                    action, observation = step
                    if hasattr(action, 'tool'):
                        try:
                            #Parse JSON observation from tools
                            tool_result = json.loads(str(observation))
                            if tool_result.get('success'):
                                tool_data[action.tool] = tool_result.get('data', {})
                        except (json.JSONDecodeError, AttributeError):
                            continue
            
            #Extract signals from technical analysis
            if 'technical_analysis' in tool_data:
                tech_data = tool_data['technical_analysis']
                
                #Overall technical signal
                overall_analysis = tech_data.get('overall_analysis', {})
                tech_signal = overall_analysis.get('signal', 'HOLD')
                tech_confidence = overall_analysis.get('confidence', 0.5)
                
                #Trend analysis for different timeframes
                trend_analysis = tech_data.get('trend_analysis', {})
                
                #Short term (20-day trend)
                short_trend = trend_analysis.get('short_term', {})
                if short_trend:
                    short_signal = self._convert_trend_to_signal(short_trend.get('trend', 'SIDEWAYS'))
                    signals['short_term'] = {
                        "signal": short_signal,
                        "confidence": min(tech_confidence + 0.1, 0.95),
                        "reasoning": f"Technical: {short_trend.get('trend', 'SIDEWAYS')} trend with {tech_confidence:.1%} confidence"
                    }
                
                #Medium term (50-day trend)  
                medium_trend = trend_analysis.get('medium_term', {})
                if medium_trend:
                    medium_signal = self._convert_trend_to_signal(medium_trend.get('trend', 'SIDEWAYS'))
                    signals['medium_term'] = {
                        "signal": medium_signal,
                        "confidence": tech_confidence,
                        "reasoning": f"Technical: {medium_trend.get('trend', 'SIDEWAYS')} trend with supporting indicators"
                    }
                
                #Long term (200-day trend)
                long_trend = trend_analysis.get('long_term', {})
                if long_trend:
                    long_signal = self._convert_trend_to_signal(long_trend.get('trend', 'SIDEWAYS'))
                    signals['long_term']['signal'] = long_signal
                    signals['long_term']['reasoning'] = f"Technical: {long_trend.get('trend', 'SIDEWAYS')} long-term trend"
            
            #Extract signals from ML predictions
            if 'ml_prediction' in tool_data:
                ml_data = tool_data['ml_prediction']
                predictions = ml_data.get('predictions', {})
                
                for timeframe in ['short_term', 'medium_term', 'long_term']:
                    if timeframe in predictions:
                        pred_data = predictions[timeframe]
                        ml_signal = pred_data.get('signal', 'HOLD')
                        ml_confidence = pred_data.get('confidence', 0.5)
                        
                        #Combine with existing signal if available
                        existing_confidence = signals[timeframe]['confidence']
                        combined_confidence = (existing_confidence + ml_confidence) / 2
                        
                        #Use ML signal if confidence is higher
                        if ml_confidence > existing_confidence:
                            signals[timeframe]['signal'] = ml_signal
                            signals[timeframe]['confidence'] = combined_confidence
                            signals[timeframe]['reasoning'] = f"ML Prediction: {ml_signal} with {ml_confidence:.1%} confidence"
            
            #Adjust long-term signal based on fundamental analysis
            if 'fundamental_analysis' in tool_data:
                fund_data = tool_data['fundamental_analysis']
                investment_rating = fund_data.get('investment_rating', {})
                fund_signal = investment_rating.get('rating', 'HOLD')
                fund_confidence = investment_rating.get('confidence', 0.5)
                
                #Convert fundamental rating to signal
                if fund_signal in ['STRONG BUY', 'BUY']:
                    fund_trading_signal = 'BUY'
                elif fund_signal in ['STRONG SELL', 'SELL']:
                    fund_trading_signal = 'SELL'
                else:
                    fund_trading_signal = 'HOLD'
                
                #Apply to long-term signal
                existing_confidence = signals['long_term']['confidence']
                combined_confidence = (existing_confidence + fund_confidence) / 2
                
                if fund_confidence > existing_confidence:
                    signals['long_term']['signal'] = fund_trading_signal
                    signals['long_term']['confidence'] = combined_confidence
                    signals['long_term']['reasoning'] = f"Fundamental: {fund_signal} rating with strong financial metrics"
            
            #Adjust signals based on sentiment analysis
            if 'sentiment_analysis' in tool_data:
                sent_data = tool_data['sentiment_analysis']
                overall_sentiment = sent_data.get('overall_sentiment', {})
                sentiment_score = overall_sentiment.get('sentiment_score', 0)
                
                #Apply sentiment adjustment to short and medium term
                for timeframe in ['short_term', 'medium_term']:
                    current_confidence = signals[timeframe]['confidence']
                    
                    #Positive sentiment boosts confidence for BUY signals
                    if signals[timeframe]['signal'] == 'BUY' and sentiment_score > 0:
                        signals[timeframe]['confidence'] = min(current_confidence + 0.1, 0.95)
                    #Negative sentiment boosts confidence for SELL signals  
                    elif signals[timeframe]['signal'] == 'SELL' and sentiment_score < 0:
                        signals[timeframe]['confidence'] = min(current_confidence + 0.1, 0.95)
            
            return signals
            
        except Exception as e:
            print(f"⚠️ Error extracting structured signals: {e}")
            return signals

    def _convert_trend_to_signal(self, trend: str) -> str:
        """Convert trend description to trading signal"""
        trend_upper = trend.upper()
        if 'UPTREND' in trend_upper or 'BULLISH' in trend_upper:
            return 'BUY'
        elif 'DOWNTREND' in trend_upper or 'BEARISH' in trend_upper:
            return 'SELL'
        else:
            return 'HOLD'

    def _parse_agent_signals(self, agent_response: str) -> Optional[Dict[str, Any]]:
        """Parse trading signals from the agent's natural language response"""
        if not agent_response:
            return None
        
        try:
            #Initialize signal structure
            signals = {}
            
            #Patterns to handle multi-line format and decimal confidence values
            timeframe_patterns = {
                'short_term': [
                    r'\bSHORT TERM[^:]*:\s*\n?Signal:\s*([A-Z]+)\s*\n?Confidence:\s*([0-9.]+)',
                    r'\bSHORT TERM.*?Signal:\s*([A-Z]+).*?Confidence:\s*([0-9.]+)'
                ],
                'medium_term': [
                    r'\bMEDIUM TERM[^:]*:\s*\n?Signal:\s*([A-Z]+)\s*\n?Confidence:\s*([0-9.]+)',
                    r'\bMEDIUM TERM.*?Signal:\s*([A-Z]+).*?Confidence:\s*([0-9.]+)'
                ],
                'long_term': [
                    r'\bLONG TERM[^:]*:\s*\n?Signal:\s*([A-Z]+)\s*\n?Confidence:\s*([0-9.]+)',
                    r'\bLONG TERM.*?Signal:\s*([A-Z]+).*?Confidence:\s*([0-9.]+)'
                ]
            }
            
            #Extract signals for each timeframe
            for timeframe, patterns in timeframe_patterns.items():
                signal_found = False
                
                for pattern in patterns:
                    matches = list(re.finditer(pattern, agent_response, re.IGNORECASE | re.DOTALL | re.MULTILINE))
                    
                    # Only process the first valid match to avoid duplicates
                    for match in matches:
                        try:
                            signal = match.group(1).upper()
                            confidence = float(match.group(2))
                            
                            #Validate signal
                            if signal in ['BUY', 'SELL', 'HOLD']:
                                #Extract reasoning - look for "Reasoning:" line after the match
                                reasoning = self._extract_reasoning_for_timeframe(agent_response, timeframe, match.end())
                                
                                signals[timeframe] = {
                                    "signal": signal,
                                    "confidence": min(max(confidence, 0.0), 1.0),  #Clamp between 0 and 1
                                    "reasoning": reasoning
                                }
                                signal_found = True
                                break  # Break after first valid match
                        except (ValueError, IndexError) as e:
                            print(f"⚠️ Error parsing match for {timeframe}: {e}")
                            continue
                    
                    # Break out of pattern loop if we found a signal for this timeframe
                    if signal_found:
                        break
            
            #If we found signals for all timeframes, return them
            if len(signals) >= 2:  #At least 2 timeframes found
                print(f"✅ Parsed {len(signals)} signals from agent response")
                return signals
            else:
                print(f"⚠️ Only found {len(signals)} signals in agent response, need at least 2")
                return None
                
        except Exception as e:
            print(f"⚠️ Error parsing agent response: {e}")
            return None

    def _extract_reasoning_for_timeframe(self, agent_response: str, timeframe: str, start_pos: int) -> str:
        """Extract reasoning text for a specific timeframe"""
        try:
            # Look for "Reasoning:" after the current position - simplified pattern
            reasoning_match = re.search(r'Reasoning:\s*(.+?)(?=\n\n|\n(?:SHORT|MEDIUM|LONG)|$)', 
                                      agent_response[start_pos:], 
                                      re.IGNORECASE | re.DOTALL)
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()
                # Clean up the reasoning - remove extra whitespace and newlines
                reasoning = re.sub(r'\s+', ' ', reasoning)
                return reasoning
            
            # Fallback: look for any text after "Reasoning:" on the same line
            simple_reasoning_match = re.search(r'Reasoning:\s*([^\n]+)', agent_response[start_pos:], re.IGNORECASE)
            if simple_reasoning_match:
                return simple_reasoning_match.group(1).strip()
            
            # Fallback: extract text between current position and next timeframe section
            next_section = re.search(r'\n(?:SHORT|MEDIUM|LONG) TERM', agent_response[start_pos:], re.IGNORECASE)
            if next_section:
                section_text = agent_response[start_pos:start_pos + next_section.start()]
            else:
                section_text = agent_response[start_pos:start_pos + 300]  # Take next 300 chars
            
            # Extract meaningful text from the section, excluding headers and labels
            lines = [line.strip() for line in section_text.split('\n') if line.strip()]
            reasoning_lines = []
            
            for line in lines:
                # Skip headers, signal/confidence lines, and empty lines
                if (not line.startswith(('Signal:', 'Confidence:', 'Reasoning:', 'SHORT', 'MEDIUM', 'LONG')) 
                    and line 
                    and not re.match(r'^\s*$', line)):
                    reasoning_lines.append(line)
            
            if reasoning_lines:
                return ' '.join(reasoning_lines)
            
            return f"Agent recommends {timeframe.replace('_', ' ')} signal based on analysis"
            
        except Exception as e:
            print(f"⚠️ Error extracting reasoning for {timeframe}: {e}")
            return f"Agent recommends signal for {timeframe.replace('_', ' ')} timeframe"


#Create global agent instance
stock_analysis_agent = StockAnalysisAgent()
