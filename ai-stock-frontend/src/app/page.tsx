import { Metadata } from 'next'
import HeroSection from '@/components/hero-section'
import FeaturesSection from '@/components/features-section'
import DemoSection from '@/components/demo-section'
import CTASection from '@/components/cta-section'

export const metadata: Metadata = {
  title: 'AI Trading Signals - Smart Stock Analysis',
  description: 'Get AI-powered trading signals with confidence levels for any stock ticker. Our algorithms analyze market data to provide actionable insights.',
  keywords: 'AI trading, stock signals, market analysis, trading bot, financial AI',
}

export default function HomePage() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      <HeroSection />
      <FeaturesSection />
      <DemoSection />
      <CTASection />
    </main>
  )
}
