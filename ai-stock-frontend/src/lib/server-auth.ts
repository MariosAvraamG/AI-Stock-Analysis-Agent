import { NextRequest } from 'next/server'
import { createClient } from '@supabase/supabase-js'
import type { User } from '@supabase/supabase-js'

// Create service role client for admin operations
export const createServiceClient = () => {
  return createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!,
    {
      auth: {
        autoRefreshToken: false,
        persistSession: false
      }
    }
  )
}

// Helper function to authenticate user from JWT token
export async function authenticateUser(request: NextRequest): Promise<
  { user: User; error: null } | { user: null; error: string; status: number }
> {
  const authHeader = request.headers.get('Authorization')
  
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return { user: null, error: 'Missing or invalid authorization header', status: 401 }
  }

  const accessToken = authHeader.replace('Bearer ', '')
  
  // Create service client to validate the JWT token
  const serviceClient = createServiceClient()
  
  try {
    // Use service client to validate the JWT and get user info
    const { data: { user }, error: userError } = await serviceClient.auth.getUser(accessToken)
    
    if (userError || !user) {
      console.error('User authentication error:', userError)
      return { user: null, error: 'User not authenticated', status: 401 }
    }
    
    return { user, error: null }
  } catch (error) {
    console.error('Authentication error:', error)
    return { user: null, error: 'Authentication failed', status: 401 }
  }
}

// Type for authentication result
export type AuthResult = 
  | { user: User; error: null }
  | { user: null; error: string; status: number }
