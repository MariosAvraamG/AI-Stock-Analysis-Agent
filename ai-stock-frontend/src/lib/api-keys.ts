import { supabase } from './supabase'
import type { ApiKey, CreateApiKeyData, ApiKeyPermissions } from './supabase'

// These functions will only work on the server side
let crypto: typeof import('crypto')
if (typeof window === 'undefined') {
  crypto = require('crypto')
}

// Generate a secure API key
export function generateApiKey(): string {
  if (typeof window !== 'undefined') {
    throw new Error('API key generation must happen server-side')
  }
  
  const prefix = 'ak_'
  const randomPart = crypto.randomBytes(32).toString('hex')
  return prefix + randomPart
}

// Hash an API key for storage
export function hashApiKey(key: string): string {
  if (typeof window !== 'undefined') {
    throw new Error('API key hashing must happen server-side')
  }
  
  return crypto.createHash('sha256').update(key).digest('hex')
}

// Get the display prefix for an API key (first 12 characters)
export function getKeyPrefix(key: string): string {
  return key.substring(0, 12) + '...'
}

// Create a new API key (server-side only)
export async function createApiKey(data: CreateApiKeyData): Promise<{ apiKey: ApiKey; plainKey: string } | { error: string }> {
  try {
    // Check if user is authenticated
    const { data: { user } } = await supabase.auth.getUser()
    if (!user) {
      return { error: 'User not authenticated' }
    }

    // Generate the API key
    const plainKey = generateApiKey()
    const keyHash = hashApiKey(plainKey)
    const keyPrefix = getKeyPrefix(plainKey)

    // Insert into database
    const { data: apiKey, error } = await supabase
      .from('api_keys')
      .insert({
        user_id: user.id,
        name: data.name,
        key_hash: keyHash,
        key_prefix: keyPrefix,
        permissions: data.permissions || { read: true, write: false },
        expires_at: data.expires_at
      })
      .select()
      .single()

    if (error) {
      console.error('Error creating API key:', error)
      return { error: 'Failed to create API key' }
    }

    return { apiKey, plainKey }
  } catch (error) {
    console.error('Error in createApiKey:', error)
    return { error: 'Failed to create API key' }
  }
}

// Get all API keys for the current user
export async function getApiKeys(): Promise<{ apiKeys: ApiKey[] } | { error: string }> {
  try {
    const { data: { user } } = await supabase.auth.getUser()
    if (!user) {
      return { error: 'User not authenticated' }
    }

    const { data: apiKeys, error } = await supabase
      .from('api_keys')
      .select('*')
      .eq('user_id', user.id)
      .order('created_at', { ascending: false })

    if (error) {
      console.error('Error fetching API keys:', error)
      return { error: 'Failed to fetch API keys' }
    }

    return { apiKeys: apiKeys || [] }
  } catch (error) {
    console.error('Error in getApiKeys:', error)
    return { error: 'Failed to fetch API keys' }
  }
}

// Update an API key
export async function updateApiKey(
  id: string, 
  updates: Partial<Pick<ApiKey, 'name' | 'permissions' | 'is_active' | 'expires_at'>>
): Promise<{ apiKey: ApiKey } | { error: string }> {
  try {
    const { data: { user } } = await supabase.auth.getUser()
    if (!user) {
      return { error: 'User not authenticated' }
    }

    const { data: apiKey, error } = await supabase
      .from('api_keys')
      .update(updates)
      .eq('id', id)
      .eq('user_id', user.id)
      .select()
      .single()

    if (error) {
      console.error('Error updating API key:', error)
      return { error: 'Failed to update API key' }
    }

    return { apiKey }
  } catch (error) {
    console.error('Error in updateApiKey:', error)
    return { error: 'Failed to update API key' }
  }
}

// Delete an API key
export async function deleteApiKey(id: string): Promise<{ success: boolean } | { error: string }> {
  try {
    const { data: { user } } = await supabase.auth.getUser()
    if (!user) {
      return { error: 'User not authenticated' }
    }

    const { error } = await supabase
      .from('api_keys')
      .delete()
      .eq('id', id)
      .eq('user_id', user.id)

    if (error) {
      console.error('Error deleting API key:', error)
      return { error: 'Failed to delete API key' }
    }

    return { success: true }
  } catch (error) {
    console.error('Error in deleteApiKey:', error)
    return { error: 'Failed to delete API key' }
  }
}

// Verify an API key (for API endpoints)
export async function verifyApiKey(key: string): Promise<{ apiKey: ApiKey; user_id: string } | { error: string }> {
  try {
    const keyHash = hashApiKey(key)
    
    const { data: apiKey, error } = await supabase
      .from('api_keys')
      .select('*')
      .eq('key_hash', keyHash)
      .eq('is_active', true)
      .single()

    if (error || !apiKey) {
      return { error: 'Invalid API key' }
    }

    // Check if key has expired
    if (apiKey.expires_at && new Date(apiKey.expires_at) < new Date()) {
      return { error: 'API key has expired' }
    }

    // Update last_used_at
    await supabase
      .from('api_keys')
      .update({ last_used_at: new Date().toISOString() })
      .eq('id', apiKey.id)

    return { apiKey, user_id: apiKey.user_id }
  } catch (error) {
    console.error('Error in verifyApiKey:', error)
    return { error: 'Failed to verify API key' }
  }
}

// Revoke an API key (set to inactive)
export async function revokeApiKey(id: string): Promise<{ success: boolean } | { error: string }> {
  const result = await updateApiKey(id, { is_active: false })
  if ('error' in result) {
    return { error: result.error }
  }
  return { success: true }
}

