import { NextRequest, NextResponse } from 'next/server'
import { authenticateUser, createServiceClient } from '@/lib/server-auth'

// GET - Fetch all API keys for the authenticated user
export async function GET(request: NextRequest) {
  try {
    // Authenticate the user
    const authResult = await authenticateUser(request)
    if (authResult.error) {
      return NextResponse.json({ error: authResult.error }, { status: authResult.status })
    }

    const { user } = authResult
    if (!user) {
      return NextResponse.json({ error: 'User not authenticated' }, { status: 401 })
    }
    
    // Use service client for database operations
    const serviceClient = createServiceClient()
    
    // Fetch API keys for this user
    const { data: apiKeys, error } = await serviceClient
      .from('api_keys')
      .select('*')
      .eq('user_id', user.id)
      .order('created_at', { ascending: false })

    if (error) {
      console.error('Error fetching API keys:', error)
      return NextResponse.json({ error: 'Failed to fetch API keys' }, { status: 500 })
    }

    return NextResponse.json({ apiKeys: apiKeys || [] })
  } catch (error) {
    console.error('API Keys GET error:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}

// POST - Create a new API key
export async function POST(request: NextRequest) {
  try {
    // Authenticate the user
    const authResult = await authenticateUser(request)
    if (authResult.error) {
      return NextResponse.json({ error: authResult.error }, { status: authResult.status })
    }

    const { user } = authResult
    if (!user) {
      return NextResponse.json({ error: 'User not authenticated' }, { status: 401 })
    }
    
    const body = await request.json()
    const { name, permissions, expires_at } = body

    if (!name || typeof name !== 'string' || name.trim().length === 0) {
      return NextResponse.json(
        { error: 'Name is required' },
        { status: 400 }
      )
    }

    // Use service client for database operations (admin privileges)
    const serviceClient = createServiceClient()
    
    // Check for duplicate name
    const { data: existingKey, error: checkError } = await serviceClient
      .from('api_keys')
      .select('id')
      .eq('user_id', user.id)
      .eq('name', name.trim())
      .single()

    if (checkError && checkError.code !== 'PGRST116') { // PGRST116 = no rows found
      console.error('Error checking for duplicate name:', checkError)
      return NextResponse.json({ error: 'Failed to validate API key name' }, { status: 500 })
    }

    if (existingKey) {
      return NextResponse.json(
        { error: 'An API key with this name already exists. Please choose a different name.' },
        { status: 400 }
      )
    }

    // Generate API key
    const { generateApiKey, hashApiKey, getKeyPrefix } = await import('@/lib/api-keys')
    
    const plainKey = generateApiKey()
    const keyHash = hashApiKey(plainKey)
    const keyPrefix = getKeyPrefix(plainKey)
    
    // Insert into database
    const { data: apiKey, error } = await serviceClient
      .from('api_keys')
      .insert({
        user_id: user.id,
        name: name.trim(),
        key_hash: keyHash,
        key_prefix: keyPrefix,
        permissions: permissions || { read: true, write: false },
        is_active: true, // Set default active status
        expires_at: expires_at
      })
      .select()
      .single()

    if (error) {
      console.error('Error creating API key:', error)
      return NextResponse.json({ error: 'Failed to create API key' }, { status: 500 })
    }

    return NextResponse.json({
      apiKey,
      plainKey
    })
  } catch (error) {
    console.error('API Keys POST error:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}
