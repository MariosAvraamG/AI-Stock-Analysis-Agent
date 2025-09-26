'use client'

import { useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { supabase } from '@/lib/supabase'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { CheckCircle } from 'lucide-react'

export default function AuthSuccessPage() {
  const router = useRouter()

  useEffect(() => {
    const handleAuthSuccess = async () => {
      // Check if we have a valid session
      const { data: { session }, error } = await supabase.auth.getSession()
      
      if (session && !error) {
        // Successful authentication - redirect to home after a short delay
        setTimeout(() => {
          router.push('/')
        }, 2000)
      } else {
        // No session or error - redirect to error page
        router.push('/auth/auth-code-error')
      }
    }

    handleAuthSuccess()
  }, [router])

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 py-12">
      <div className="container mx-auto px-4">
        <Card className="max-w-md mx-auto">
          <CardHeader className="text-center">
            <div className="mx-auto mb-4 h-12 w-12 rounded-full bg-green-100 flex items-center justify-center">
              <CheckCircle className="h-6 w-6 text-green-600" />
            </div>
            <CardTitle className="text-2xl text-green-600">Authentication Successful</CardTitle>
            <p className="text-gray-600">You have been successfully signed in</p>
          </CardHeader>
          <CardContent className="text-center">
            <p className="text-sm text-gray-500">
              Redirecting you to the home page...
            </p>
            <div className="mt-4 flex justify-center">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
