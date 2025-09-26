import { NextRequest } from 'next/server'
import { verifyApiKey } from './api-keys'

export async function authenticateApiKey(request: NextRequest): Promise<
  { success: true; user_id: string; apiKey: any } | 
  { success: false; error: string; status: number }
> {
  const authHeader = request.headers.get('authorization')
  
  if (!authHeader) {
    return { 
      success: false, 
      error: 'Missing Authorization header', 
      status: 401 
    }
  }

  const match = authHeader.match(/^Bearer\s+(.+)$/)
  if (!match) {
    return { 
      success: false, 
      error: 'Invalid Authorization header format. Use: Bearer <api_key>', 
      status: 401 
    }
  }

  const apiKey = match[1]
  const result = await verifyApiKey(apiKey)

  if ('error' in result) {
    return { 
      success: false, 
      error: result.error, 
      status: 401 
    }
  }

  return { 
    success: true, 
    user_id: result.user_id, 
    apiKey: result.apiKey 
  }
}

