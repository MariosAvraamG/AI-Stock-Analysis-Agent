import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!

export const supabase = createClient(supabaseUrl, supabaseAnonKey, {
  auth: {
    autoRefreshToken: true,
    persistSession: true,
    detectSessionInUrl: true
  }
})

// Types for our database
export type Profile = {
  id: string
  full_name: string | null
  email: string | null
  created_at: string
  updated_at: string
}

export type ApiKeyPermissions = {
  read: boolean
  write: boolean
}

export type ApiKey = {
  id: string
  user_id: string
  name: string
  key_hash: string
  key_prefix: string
  permissions: ApiKeyPermissions
  is_active: boolean
  last_used_at: string | null
  expires_at: string | null
  created_at: string
  updated_at: string
}

export type CreateApiKeyData = {
  name: string
  permissions?: ApiKeyPermissions
  expires_at?: string | null
}