'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Search, TrendingUp, TrendingDown, Minus, Loader2 } from 'lucide-react'

interface SignalResult {
  ticker: string
  signals: {
    short_term: {
      signal: 'BUY' | 'SELL' | 'HOLD'
      confidence: number
      reasoning: string
    }
    medium_term: {
      signal: 'BUY' | 'SELL' | 'HOLD'
      confidence: number
      reasoning: string
    }
    long_term: {
      signal: 'BUY' | 'SELL' | 'HOLD'
      confidence: number
      reasoning: string
    }
  }
  execution_time: number
  tools_used: string[]
}

export default function DemoSection() {
  const [ticker, setTicker] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<SignalResult | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!ticker.trim()) return

    setLoading(true)
    
    // Simulate API call with realistic delay
    await new Promise(resolve => setTimeout(resolve, 2000))
    
    // Mock response - mimics real backend API response structure
    const signals = ['BUY', 'SELL', 'HOLD']
    const mockResult: SignalResult = {
      ticker: ticker.toUpperCase(),
      signals: {
        short_term: {
          signal: signals[Math.floor(Math.random() * signals.length)] as 'BUY' | 'SELL' | 'HOLD',
          confidence: Math.random() * 0.4 + 0.6, // 60-100%
          reasoning: 'Technical analysis shows strong momentum with RSI indicating overbought conditions. ML prediction suggests moderate confidence in short-term direction.'
        },
        medium_term: {
          signal: signals[Math.floor(Math.random() * signals.length)] as 'BUY' | 'SELL' | 'HOLD',
          confidence: Math.random() * 0.3 + 0.4, // 40-70%
          reasoning: 'Trend analysis reveals mixed signals with technical patterns showing consolidation. Market sentiment remains neutral with moderate volatility expected.'
        },
        long_term: {
          signal: signals[Math.floor(Math.random() * signals.length)] as 'BUY' | 'SELL' | 'HOLD',
          confidence: Math.random() * 0.2 + 0.5, // 50-70%
          reasoning: 'Fundamental analysis indicates strong company metrics with favorable growth prospects. Long-term technical trends support positive outlook despite current market uncertainty.'
        }
      },
      execution_time: Math.random() * 30 + 45, // 45-75 seconds
      tools_used: ['technical_analysis', 'ml_prediction', 'sentiment_analysis', 'fundamental_analysis']
    }
    
    setResult(mockResult)
    setLoading(false)
  }

  const getSignalIcon = (signal: string) => {
    switch (signal) {
      case 'BUY': return <TrendingUp className="h-5 w-5" />
      case 'SELL': return <TrendingDown className="h-5 w-5" />
      default: return <Minus className="h-5 w-5" />
    }
  }

  const getSignalColor = (signal: string) => {
    switch (signal) {
      case 'BUY': return 'bg-green-100 text-green-800 border-green-200'
      case 'SELL': return 'bg-red-100 text-red-800 border-red-200'
      default: return 'bg-yellow-100 text-yellow-800 border-yellow-200'
    }
  }

  return (
    <section className="py-20 bg-gradient-to-br from-gray-50 to-blue-50">
      <div className="container mx-auto px-4">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
            See Our AI Trading Signals in Action
          </h2>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Enter any stock ticker to see the exact data structure our API returns. Experience multi-timeframe analysis with confidence levels and detailed reasoning.
          </p>
        </div>

        <div className="max-w-2xl mx-auto">
          <Card className="shadow-xl border-0 bg-white/80 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-center text-2xl">Stock Signal Generator</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="flex gap-2">
                  <div className="relative flex-1">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-5 w-5" />
                    <Input
                      type="text"
                      placeholder="Enter stock ticker (e.g., AAPL, TSLA, MSFT)"
                      value={ticker}
                      onChange={(e) => setTicker(e.target.value)}
                      className="pl-10 py-3 text-lg"
                      disabled={loading}
                    />
                  </div>
                  <Button 
                    type="submit" 
                    size="lg" 
                    disabled={loading || !ticker.trim()}
                    className="px-8"
                  >
                    {loading ? (
                      <Loader2 className="h-5 w-5 animate-spin" />
                    ) : (
                      'Generate Signal'
                    )}
                  </Button>
                </div>
              </form>

              {result && (
                <div className="mt-8 space-y-6">
                  <div className="text-center">
                    <h3 className="text-2xl font-bold text-gray-900 mb-2">{result.ticker}</h3>
                    <div className="flex items-center justify-center space-x-4 text-sm text-gray-600">
                      <span>Execution Time: {result.execution_time.toFixed(2)}s</span>
                      <span>•</span>
                      <span>Tools Used: {result.tools_used.join(', ')}</span>
                    </div>
                  </div>

                  {/* Timeframe Signals */}
                  <div className="grid gap-4">
                    {Object.entries(result.signals).map(([timeframe, signal]) => (
                      <div key={timeframe} className="p-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg border">
                        <div className="flex items-center justify-between mb-3">
                          <h4 className="text-lg font-semibold text-gray-900 capitalize">
                            {timeframe.replace('_', ' ')} Term
                          </h4>
                          <Badge className={`${getSignalColor(signal.signal)} px-3 py-1 text-sm font-semibold`}>
                            {getSignalIcon(signal.signal)}
                            <span className="ml-1">{signal.signal}</span>
                          </Badge>
                        </div>
                        
                        <div className="grid grid-cols-2 gap-4 mb-3">
                          <div>
                            <div className="text-sm text-gray-600">Confidence</div>
                            <div className="text-xl font-bold text-blue-600">
                              {(signal.confidence * 100).toFixed(1)}%
                            </div>
                          </div>
                          <div>
                            <div className="text-sm text-gray-600">Timeframe</div>
                            <div className="text-lg font-medium text-gray-900">
                              {timeframe === 'short_term' ? '5-20 days' : 
                               timeframe === 'medium_term' ? '20-60 days' : '60+ days'}
                            </div>
                          </div>
                        </div>
                        
                        <div className="bg-white/60 p-3 rounded-lg">
                          <div className="text-sm text-gray-600 mb-1">AI Reasoning</div>
                          <div className="text-gray-800 text-sm">{signal.reasoning}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="text-center text-sm text-gray-500">
                <p>This demo shows the actual data structure returned by our AI trading signals API. Sign up for API access to get real-time analysis.</p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  )
}
