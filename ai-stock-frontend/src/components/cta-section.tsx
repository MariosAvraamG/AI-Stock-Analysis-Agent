import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { ArrowRight, Key, Zap } from 'lucide-react'
import Link from 'next/link'

export default function CTASection() {
  return (
    <section className="py-20 bg-gradient-to-r from-blue-600 to-purple-600">
      <div className="container mx-auto px-4">
        <div className="max-w-4xl mx-auto text-center text-white">
          <h2 className="text-3xl md:text-4xl font-bold mb-6">
            Ready to Start Trading Smarter?
          </h2>
          <p className="text-xl mb-12 text-blue-100">
            Join thousands of traders who trust our AI-powered signals to make better investment decisions.
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
            <Card className="bg-white/10 backdrop-blur-sm border-white/20 text-white">
              <CardContent className="p-8 text-center">
                <Key className="h-12 w-12 mx-auto mb-4 text-yellow-400" />
                <h3 className="text-2xl font-bold mb-4">Get API Access</h3>
                <p className="text-blue-100 mb-6">
                  Generate and manage your API keys to integrate our trading signals into your applications.
                </p>
                <Link href="/api-keys">
                  <Button className="bg-white text-blue-600 hover:bg-blue-50 font-semibold">
                    Manage API Keys
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Button>
                </Link>
              </CardContent>
            </Card>
            
            <Card className="bg-white/10 backdrop-blur-sm border-white/20 text-white">
              <CardContent className="p-8 text-center">
                <Zap className="h-12 w-12 mx-auto mb-4 text-yellow-400" />
                <h3 className="text-2xl font-bold mb-4">Start Free Trial</h3>
                <p className="text-blue-100 mb-6">
                  Try our service with 100 free API calls. No credit card required to get started.
                </p>
                <Button variant="outline" className="border-white/30 text-white hover:bg-white/10 font-semibold">
                  Start Free Trial
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </section>
  )
}
