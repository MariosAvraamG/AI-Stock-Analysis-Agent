'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { 
  Eye, 
  EyeOff, 
  Copy, 
  Trash2, 
  Edit3, 
  Calendar,
  Shield,
  Clock,
  Pencil,
  Check,
  X,
  AlertCircle
} from 'lucide-react'
import type { ApiKey } from '@/lib/supabase'

interface ApiKeyCardProps {
  apiKey: ApiKey
  onDelete: (id: string) => void
  onUpdate: (id: string, updates: Partial<ApiKey>) => void
  existingNames?: string[] // Add prop to pass existing names for validation
}

export function ApiKeyCard({ apiKey, onDelete, onUpdate, existingNames = [] }: ApiKeyCardProps) {
  const [showFullKey, setShowFullKey] = useState(false)
  const [copying, setCopying] = useState(false)
  const [isEditing, setIsEditing] = useState(false)
  const [editName, setEditName] = useState(apiKey.name)
  const [isUpdating, setIsUpdating] = useState(false)
  const [editError, setEditError] = useState<string | null>(null)

  const handleCopy = async () => {
    setCopying(true)
    try {
      await navigator.clipboard.writeText(apiKey.key_prefix)
      // Note: We can only copy the prefix since we don't store the full key
    } catch (error) {
      console.error('Failed to copy:', error)
    } finally {
      setTimeout(() => setCopying(false), 1000)
    }
  }

  const handleToggleActive = () => {
    onUpdate(apiKey.id, { is_active: !apiKey.is_active })
  }

  const handleStartEdit = () => {
    setIsEditing(true)
    setEditName(apiKey.name)
    setEditError(null)
  }

  const handleCancelEdit = () => {
    setIsEditing(false)
    setEditName(apiKey.name)
    setEditError(null)
  }

  const validateName = (name: string): string | null => {
    const trimmedName = name.trim()
    
    if (trimmedName === '') {
      return 'Name cannot be empty'
    }
    
    if (trimmedName === apiKey.name) {
      return null // No change
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

  const handleSaveEdit = async () => {
    const trimmedName = editName.trim()
    
    // Validate name
    const validationError = validateName(editName)
    if (validationError) {
      setEditError(validationError)
      return
    }
    
    if (trimmedName === apiKey.name) {
      setIsEditing(false)
      setEditError(null)
      return
    }

    setIsUpdating(true)
    setEditError(null)
    
    try {
      await onUpdate(apiKey.id, { name: trimmedName })
      setIsEditing(false)
    } catch (error: any) {
      console.error('Failed to update name:', error)
      // Show server error message if available
      const errorMessage = error?.message || 'Failed to update API key name'
      setEditError(errorMessage)
      // Don't reset the name so user can try again
    } finally {
      setIsUpdating(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSaveEdit()
    } else if (e.key === 'Escape') {
      handleCancelEdit()
    }
  }

  const handleNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setEditName(e.target.value)
    // Clear error when user starts typing
    if (editError) {
      setEditError(null)
    }
  }

  const formatDate = (dateString: string | null) => {
    if (!dateString) return 'Never'
    return new Date(dateString).toLocaleDateString()
  }

  const isExpired = apiKey.expires_at && new Date(apiKey.expires_at) < new Date()

  return (
    <Card className={`${!apiKey.is_active || isExpired ? 'opacity-60' : ''}`}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 flex-1 min-w-0">
            {isEditing ? (
              <div className="flex flex-col gap-2 flex-1">
                <div className="flex items-center gap-2">
                  <Input
                    value={editName}
                    onChange={handleNameChange}
                    onKeyDown={handleKeyPress}
                    className={`text-lg font-semibold h-8 flex-1 ${editError ? 'border-red-500 focus-visible:border-red-500 focus-visible:ring-red-500/20' : ''}`}
                    disabled={isUpdating}
                    autoFocus
                  />
                  <div className="flex items-center gap-1">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={handleSaveEdit}
                      disabled={isUpdating || !!editError}
                      className="h-8 w-8 p-0 text-green-600 hover:text-green-700 hover:bg-green-50 disabled:opacity-50"
                      title="Save changes"
                    >
                      <Check className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={handleCancelEdit}
                      disabled={isUpdating}
                      className="h-8 w-8 p-0 text-gray-500 hover:text-gray-600 hover:bg-gray-50"
                      title="Cancel editing"
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
                {editError && (
                  <div className="flex items-center gap-2 text-red-600 text-sm">
                    <AlertCircle className="h-4 w-4" />
                    <span>{editError}</span>
                  </div>
                )}
              </div>
            ) : (
              <>
                <CardTitle className="text-lg font-semibold truncate">{apiKey.name}</CardTitle>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleStartEdit}
                  className="h-8 w-8 p-0 text-gray-400 hover:text-gray-600 hover:bg-gray-50 shrink-0"
                  title="Rename API key"
                >
                  <Pencil className="h-4 w-4" />
                </Button>
              </>
            )}
          </div>
          <div className="flex items-center gap-2 shrink-0">
            {isExpired && (
              <Badge variant="destructive" className="text-xs">
                Expired
              </Badge>
            )}
            <Badge 
              variant={apiKey.is_active ? "default" : "secondary"}
              className="text-xs"
            >
              {apiKey.is_active ? 'Active' : 'Inactive'}
            </Badge>
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {/* API Key Display */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-gray-700">API Key</label>
          <div className="flex items-center gap-2 p-3 bg-gray-50 rounded-md font-mono text-sm">
            <span className="flex-1">
              {apiKey.key_prefix}
            </span>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleCopy}
              className="h-6 w-6 p-0"
              title="Copy key prefix"
            >
              <Copy className={`h-4 w-4 ${copying ? 'text-green-600' : ''}`} />
            </Button>
          </div>
          <p className="text-xs text-gray-500">
            Note: Full API key is only shown once during creation for security
          </p>
        </div>

        {/* Permissions */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-gray-700 flex items-center gap-1">
            <Shield className="h-4 w-4" />
            Permissions
          </label>
          <div className="flex gap-2">
            {apiKey.permissions.read && (
              <Badge variant="outline" className="text-xs">Read</Badge>
            )}
            {apiKey.permissions.write && (
              <Badge variant="outline" className="text-xs">Write</Badge>
            )}
          </div>
        </div>

        {/* Metadata */}
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div className="space-y-1">
            <div className="flex items-center gap-1 text-gray-600">
              <Calendar className="h-4 w-4" />
              <span>Created</span>
            </div>
            <div className="font-medium">{formatDate(apiKey.created_at)}</div>
          </div>
          
          <div className="space-y-1">
            <div className="flex items-center gap-1 text-gray-600">
              <Clock className="h-4 w-4" />
              <span>Last Used</span>
            </div>
            <div className="font-medium">{formatDate(apiKey.last_used_at)}</div>
          </div>
        </div>

        {apiKey.expires_at && (
          <div className="space-y-1 text-sm">
            <div className="flex items-center gap-1 text-gray-600">
              <Calendar className="h-4 w-4" />
              <span>Expires</span>
            </div>
            <div className={`font-medium ${isExpired ? 'text-red-600' : ''}`}>
              {formatDate(apiKey.expires_at)}
            </div>
          </div>
        )}

        {/* Actions */}
        <div className="flex justify-between pt-4 border-t">
          <Button
            variant="outline"
            size="sm"
            onClick={handleToggleActive}
            className="flex items-center gap-2"
          >
            <Edit3 className="h-4 w-4" />
            {apiKey.is_active ? 'Deactivate' : 'Activate'}
          </Button>
          
          <Button
            variant="destructive"
            size="sm"
            onClick={() => onDelete(apiKey.id)}
            className="flex items-center gap-2"
          >
            <Trash2 className="h-4 w-4" />
            Delete
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}

