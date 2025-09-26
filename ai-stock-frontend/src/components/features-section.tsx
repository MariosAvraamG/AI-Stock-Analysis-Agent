import { Brain, BarChart3, Clock, Shield, Zap, TrendingUp } from 'lucide-react'

const features = [
  {
    icon: Brain,
    title: 'AI-Powered Analysis',
    description: 'Machine learning algorithms analyze market patterns, news sentiment, and technical indicators to generate accurate signals.',
  },
  {
    icon: BarChart3,
    title: 'Confidence Scoring',
    description: 'Each signal comes with a confidence level from 0-100%, helping you make informed decisions based on signal strength.',
  },
  {
    icon: Clock,
    title: 'Real-Time Signals',
    description: 'Get instant trading signals as market conditions change, ensuring you never miss profitable opportunities.',
  },
  {
    icon: Shield,
    title: 'Secure API Access',
    description: 'Enterprise-grade security with encrypted API keys and rate limiting to protect your trading strategies.',
  },
  {
    icon: Zap,
    title: 'Lightning Fast',
    description: 'Fast response times ensure you get signals when they matter most in fast-moving markets.',
  },
  {
    icon: TrendingUp,
    title: 'Multi-Asset Support',
    description: 'Analyze stocks, ETFs, and major indices with comprehensive coverage of global markets.',
  },
]

export default function FeaturesSection() {
  return (
    <section className="py-20 bg-white">
      <div className="container mx-auto px-4">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
            Why Choose Our AI Trading Signals?
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Leverage cutting-edge artificial intelligence to make smarter trading decisions 
            with confidence levels that help you manage risk effectively.
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <div key={index} className="group p-6 rounded-xl border border-gray-200 hover:border-blue-300 hover:shadow-lg transition-all duration-300">
              <div className="flex items-center mb-4">
                <div className="p-3 bg-blue-100 rounded-lg group-hover:bg-blue-200 transition-colors">
                  <feature.icon className="h-6 w-6 text-blue-600" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 ml-4">{feature.title}</h3>
              </div>
              <p className="text-gray-600 leading-relaxed">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
