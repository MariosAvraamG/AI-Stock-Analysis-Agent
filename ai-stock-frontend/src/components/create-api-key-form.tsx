'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Plus, Key, AlertCircle, CheckCircle, Copy, Shield, Edit } from 'lucide-react'
import { supabase } from '@/lib/supabase'
import type { CreateApiKeyData, ApiKeyPermissions } from '@/lib/supabase'

interface CreateApiKeyFormProps {
  onApiKeyCreated: (apiKey: any, plainKey: string) => void
  existingNames?: string[]
}

export function CreateApiKeyForm({ onApiKeyCreated, existingNames = [] }: CreateApiKeyFormProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [newKey, setNewKey] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)
  const [nameError, setNameError] = useState<string | null>(null)
  
  // Default: Read always enabled, write optional
  const [formData, setFormData] = useState<CreateApiKeyData>({
    name: '',
    permissions: { read: true, write: false },
    expires_at: null
  })

  const validateName = (name: string): string | null => {
    const trimmedName = name.trim()
    
    if (trimmedName === '') {
      return 'Name is required'
    }
    
    // Check if name already exists (case insensitive)
    const duplicateExists = existingNames.some(
      existingName => existingName.toLowerCase() === trimmedName.toLowerCase()
    )
    
    if (duplicateExists) {
      return 'An API key with this name already exists'
    }
    
    return null
  }

  const handleNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newName = e.target.value
    setFormData({ ...formData, name: newName })
    
    // Clear previous errors
    setNameError(null)
    setError(null)
    
    // Validate name in real-time
    if (newName.trim()) {
      const validationError = validateName(newName)
      setNameError(validationError)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    // Final validation
    const validationError = validateName(formData.name)
    if (validationError) {
      setNameError(validationError)
      return
    }
    
    setLoading(true)
    setError(null)
    setNameError(null)

    try {
      const { data: { session } } = await supabase.auth.getSession()
      
      const headers: HeadersInit = {
        'Content-Type': 'application/json'
      }
      
      if (session?.access_token) {
        headers['Authorization'] = `Bearer ${session.access_token}`
      }
      
      const response = await fetch('/api/api-keys', {
        method: 'POST',
        headers,
        body: JSON.stringify(formData)
      })

      const result = await response.json()

      if (!response.ok) {
        throw new Error(result.error || 'Failed to create API key')
      }

      setNewKey(result.plainKey)
      onApiKeyCreated(result.apiKey, result.plainKey)
      
      // Reset form - keep read enabled by default
      setFormData({
        name: '',
        permissions: { read: true, write: false },
        expires_at: null
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  const handleWritePermissionChange = (checked: boolean) => {
    setFormData({
      ...formData,
      permissions: {
        read: true, // Always true
        write: checked
      }
    })
  }

  const handleCopyKey = async () => {
    if (!newKey) return
    
    try {
      await navigator.clipboard.writeText(newKey)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (error) {
      console.error('Failed to copy:', error)
    }
  }

  const handleClose = () => {
    setIsOpen(false)
    setNewKey(null)
    setError(null)
    setCopied(false)
  }

  if (newKey) {
    return (
      <Card className="border-green-200 bg-green-50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-green-800">
            <CheckCircle className="h-5 w-5" />
            API Key Created Successfully
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="bg-white p-4 rounded-md border">
            <p className="text-sm text-gray-600 mb-2">
              <strong>Important:</strong> Copy this API key now. You won't be able to see it again!
            </p>
            <div className="flex items-center gap-2 p-3 bg-gray-50 rounded-md font-mono text-sm break-all">
              <span className="flex-1">{newKey}</span>
              <Button
                variant="outline"
                size="sm"
                onClick={handleCopyKey}
                className="flex items-center gap-1"
              >
                <Copy className="h-4 w-4" />
                {copied ? 'Copied!' : 'Copy'}
              </Button>
            </div>
          </div>
          
          {/* Show permissions summary */}
          <div className="bg-blue-50 p-3 rounded-md border border-blue-200">
            <p className="text-sm text-blue-800 font-medium mb-1">API Key Permissions:</p>
            <div className="text-xs text-blue-700 space-y-1">
              <div className="flex items-center gap-2">
                <Shield className="h-3 w-3" />
                <span>✅ Read access - Can view trading signals</span>
              </div>
              {formData.permissions?.write && (
                <div className="flex items-center gap-2">
                  <Edit className="h-3 w-3" />
                  <span>✅ Write access - Can create/modify data</span>
                </div>
              )}
              {!formData.permissions?.write && (
                <div className="flex items-center gap-2">
                  <Edit className="h-3 w-3 opacity-50" />
                  <span className="opacity-50">❌ Write access - Not enabled</span>
                </div>
              )}
            </div>
          </div>
          
          <Button onClick={handleClose} className="w-full">
            Done
          </Button>
        </CardContent>
      </Card>
    )
  }

  if (!isOpen) {
    return (
      <Button 
        onClick={() => setIsOpen(true)}
        className="flex items-center gap-2"
      >
        <Plus className="h-4 w-4" />
        Create New API Key
      </Button>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Key className="h-5 w-5" />
          Create New API Key
        </CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          {error && (
            <div className="flex items-center gap-2 p-3 bg-red-50 border border-red-200 rounded-md text-red-700">
              <AlertCircle className="h-4 w-4" />
              <span className="text-sm">{error}</span>
            </div>
          )}

          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700">
              Key Name *
            </label>
            <Input
              type="text"
              value={formData.name}
              onChange={handleNameChange}
              placeholder="e.g., Production API, Development Key"
              className={nameError ? 'border-red-500 focus-visible:border-red-500 focus-visible:ring-red-500/20' : ''}
              required
            />
            {nameError && (
              <div className="flex items-center gap-2 text-red-600 text-sm">
                <AlertCircle className="h-4 w-4" />
                <span>{nameError}</span>
              </div>
            )}
            <p className="text-xs text-gray-500">
              Choose a descriptive name to help you identify this key
            </p>
          </div>

          <div className="space-y-3">
            <label className="text-sm font-medium text-gray-700">
              Permissions
            </label>
            
            {/* Read Permission - Always Enabled */}
            <div className="flex items-center gap-3 p-3 bg-green-50 border border-green-200 rounded-md">
              <input
                type="checkbox"
                checked={true}
                disabled={true}
                className="h-4 w-4 text-green-600"
              />
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <Shield className="h-4 w-4 text-green-600" />
                  <span className="text-sm font-medium text-green-800">
                    Read access (view trading signals)
                  </span>
                </div>
                <p className="text-xs text-green-600 mt-1">
                  Always included - Required for API functionality
                </p>
              </div>
            </div>
            
            {/* Write Permission - Optional */}
            <div className="flex items-center gap-3 p-3 border border-gray-200 rounded-md hover:bg-gray-50 transition-colors">
              <input
                type="checkbox"
                checked={formData.permissions?.write || false}
                onChange={(e) => handleWritePermissionChange(e.target.checked)}
                className="h-4 w-4 text-blue-600"
              />
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <Edit className="h-4 w-4 text-blue-600" />
                  <span className="text-sm font-medium text-gray-700">
                    Write access (create/modify data)
                  </span>
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  Optional - Enable for future write operations
                </p>
              </div>
            </div>
            
            <p className="text-xs text-gray-500">
              All API keys include read access by default. Add write permissions only if needed for your use case.
            </p>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700">
              Expiration Date (Optional)
            </label>
            <Input
              type="date"
              value={formData.expires_at ? formData.expires_at.split('T')[0] : ''}
              onChange={(e) => setFormData({
                ...formData,
                expires_at: e.target.value ? new Date(e.target.value).toISOString() : null
              })}
              min={new Date().toISOString().split('T')[0]}
            />
            <p className="text-xs text-gray-500">
              Leave empty for no expiration
            </p>
          </div>

          <div className="flex gap-2 pt-4">
            <Button 
              type="submit" 
              disabled={loading || !formData.name.trim() || !!nameError}
              className="flex items-center gap-2"
            >
              {loading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                  Creating...
                </>
              ) : (
                <>
                  <Key className="h-4 w-4" />
                  Create API Key
                </>
              )}
            </Button>
            <Button type="button" variant="outline" onClick={handleClose}>
              Cancel
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  )
}
