'use client'

import React, { useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { ArrowLeft, Key, AlertCircle, Loader2 } from 'lucide-react'
import Link from 'next/link'
import { useAuth } from '@/lib/auth-context'
import { ApiKeyCard } from '@/components/api-key-card'
import { CreateApiKeyForm } from '@/components/create-api-key-form'
import { supabase } from '@/lib/supabase'
import type { ApiKey } from '@/lib/supabase'

export default function ApiKeysPage() {
  const { user, loading: authLoading } = useAuth()
  const [apiKeys, setApiKeys] = useState<ApiKey[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (user) {
      fetchApiKeys()
    } else if (!authLoading) {
      setLoading(false)
    }
  }, [user, authLoading])

  const fetchApiKeys = async () => {
    try {
      setLoading(true)
      
      // Get the current session to include in the request
      const { data: { session } } = await supabase.auth.getSession()
      
      const headers: HeadersInit = {
        'Content-Type': 'application/json'
      }
      
      if (session?.access_token) {
        headers['Authorization'] = `Bearer ${session.access_token}`
      }
      
      const response = await fetch('/api/api-keys', { headers })
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || 'Failed to fetch API keys')
      }

      setApiKeys(data.apiKeys)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  const handleApiKeyCreated = (newApiKey: ApiKey) => {
    setApiKeys(prev => [newApiKey, ...prev])
  }

  const handleApiKeyUpdate = async (id: string, updates: Partial<ApiKey>) => {
    try {
      const { data: { session } } = await supabase.auth.getSession()
      
      const headers: HeadersInit = {
        'Content-Type': 'application/json'
      }
      
      if (session?.access_token) {
        headers['Authorization'] = `Bearer ${session.access_token}`
      }
      
      const response = await fetch(`/api/api-keys/${id}`, {
        method: 'PUT',
        headers,
        body: JSON.stringify(updates)
      })

      const data = await response.json()

      if (!response.ok) {
        // Throw error with server message for better error handling
        throw new Error(data.error || 'Failed to update API key')
      }

      setApiKeys(prev => 
        prev.map(key => key.id === id ? data.apiKey : key)
      )
    } catch (err) {
      console.error('Failed to update API key:', err)
      // Re-throw the error so the component can handle it
      throw err
    }
  }

  const handleApiKeyDelete = async (id: string) => {
    if (!confirm('Are you sure you want to delete this API key? This action cannot be undone.')) {
      return
    }

    try {
      const { data: { session } } = await supabase.auth.getSession()
      
      const headers: HeadersInit = {}
      
      if (session?.access_token) {
        headers['Authorization'] = `Bearer ${session.access_token}`
      }
      
      const response = await fetch(`/api/api-keys/${id}`, {
        method: 'DELETE',
        headers
      })

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.error || 'Failed to delete API key')
      }

      setApiKeys(prev => prev.filter(key => key.id !== id))
    } catch (err) {
      console.error('Failed to delete API key:', err)
      // You might want to show a toast notification here
    }
  }

  // Get existing names for validation
  const existingNames = apiKeys.map(key => key.name)

  if (authLoading || loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 py-12">
        <div className="container mx-auto px-4">
          <div className="flex items-center justify-center h-64">
            <Loader2 className="h-8 w-8 animate-spin" />
          </div>
        </div>
      </div>
    )
  }

  if (!user) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 py-12">
        <div className="container mx-auto px-4">
          <div className="mb-8">
            <Link href="/">
              <Button variant="ghost" className="mb-4">
                <ArrowLeft className="mr-2 h-4 w-4" />
                Back to Home
              </Button>
            </Link>
          </div>
          
          <Card className="max-w-md mx-auto">
            <CardContent className="pt-6">
              <div className="text-center">
                <AlertCircle className="h-12 w-12 text-amber-500 mx-auto mb-4" />
                <h2 className="text-xl font-semibold mb-2">Authentication Required</h2>
                <p className="text-gray-600 mb-4">
                  You need to be signed in to manage API keys.
                </p>
                <Link href="/signin">
                  <Button>Sign In</Button>
                </Link>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 py-12">
      <div className="container mx-auto px-4">
        <div className="mb-8">
          <Link href="/">
            <Button variant="ghost" className="mb-4">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Home
            </Button>
          </Link>
          <h1 className="text-4xl font-bold text-gray-900 mb-2">API Key Management</h1>
          <p className="text-xl text-gray-600">Generate and manage your API keys for accessing trading signals</p>
        </div>

        <div className="max-w-4xl mx-auto space-y-6">
          {/* Create API Key Section */}
          <CreateApiKeyForm 
            onApiKeyCreated={handleApiKeyCreated}
            existingNames={existingNames}
          />

          {/* API Keys List */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Key className="h-5 w-5" />
                Your API Keys ({apiKeys.length})
              </CardTitle>
            </CardHeader>
            <CardContent>
              {error && (
                <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-200 rounded-md text-red-700 mb-4">
                  <AlertCircle className="h-4 w-4" />
                  <span className="text-sm">{error}</span>
                </div>
              )}

              {apiKeys.length === 0 ? (
                <div className="text-center py-12">
                  <Key className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-600 mb-2">No API keys yet</p>
                  <p className="text-sm text-gray-500">
                    Create your first API key to start accessing the trading signals API
                  </p>
                </div>
              ) : (
                <div className="grid gap-4">
                  {apiKeys.map((apiKey) => (
                    <ApiKeyCard
                      key={apiKey.id}
                      apiKey={apiKey}
                      onDelete={handleApiKeyDelete}
                      onUpdate={handleApiKeyUpdate}
                      existingNames={existingNames.filter(name => name !== apiKey.name)}
                    />
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Usage Information */}
          <Card>
            <CardHeader>
              <CardTitle>Using Your API Keys</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <h4 className="font-medium mb-2">Authentication</h4>
                <p className="text-sm text-gray-600 mb-2">
                  Include your API key in the Authorization header:
                </p>
                <div className="bg-gray-100 p-3 rounded-md font-mono text-sm">
                  Authorization: Bearer your_api_key_here
                </div>
              </div>
              
              <div>
                <h4 className="font-medium mb-2">Example Request</h4>
                <div className="bg-gray-100 p-3 rounded-md font-mono text-sm">
                  curl -X POST -H &quot;Authorization: Bearer your_api_key_here&quot; \<br />
                  &nbsp;&nbsp;&nbsp;&nbsp; -H &quot;Content-Type: application/json&quot; \<br />
                  &nbsp;&nbsp;&nbsp;&nbsp; -d &apos;&#123;&quot;ticker&quot;: &quot;AAPL&quot;&#125;&apos; \<br />
                  &nbsp;&nbsp;&nbsp;&nbsp; &quot;http://localhost:8000/ai-agent/signals&quot;
                </div>
              </div>

              <div>
                <h4 className="font-medium mb-2">Rate Limits</h4>
                <p className="text-sm text-gray-600">
                  API keys are subject to rate limiting. Please refer to the API documentation for current limits.
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
