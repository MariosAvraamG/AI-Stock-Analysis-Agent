import { NextRequest, NextResponse } from 'next/server'
import { authenticateUser, createServiceClient } from '@/lib/server-auth'

// PUT - Update an API key
export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
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
    const { id } = await params

    const body = await request.json()
    const { name, permissions, is_active, expires_at } = body

    const updates: any = {}
    if (name !== undefined) {
      // Validate name if it's being updated
      if (typeof name !== 'string' || name.trim().length === 0) {
        return NextResponse.json(
          { error: 'Name cannot be empty' },
          { status: 400 }
        )
      }
      updates.name = name.trim()
    }
    if (permissions !== undefined) updates.permissions = permissions
    if (is_active !== undefined) updates.is_active = is_active
    if (expires_at !== undefined) updates.expires_at = expires_at

    // Use service client for database operations
    const serviceClient = createServiceClient()
    
    // If name is being updated, check for duplicates
    if (updates.name) {
      const { data: existingKey, error: checkError } = await serviceClient
        .from('api_keys')
        .select('id')
        .eq('user_id', user.id)
        .eq('name', updates.name)
        .neq('id', id) // Exclude the current key
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
    }
    
    const { data: apiKey, error } = await serviceClient
      .from('api_keys')
      .update(updates)
      .eq('id', id)
      .eq('user_id', user.id)
      .select()
      .single()

    if (error) {
      console.error('Error updating API key:', error)
      return NextResponse.json({ error: 'Failed to update API key' }, { status: 500 })
    }

    return NextResponse.json({ apiKey })
  } catch (error) {
    console.error('API Key PUT error:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}

// DELETE - Delete an API key
export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
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
    const { id } = await params

    // Use service client for database operations
    const serviceClient = createServiceClient()
    
    const { error } = await serviceClient
      .from('api_keys')
      .delete()
      .eq('id', id)
      .eq('user_id', user.id)

    if (error) {
      console.error('Error deleting API key:', error)
      return NextResponse.json({ error: 'Failed to delete API key' }, { status: 500 })
    }

    return NextResponse.json({ success: true })
  } catch (error) {
    console.error('API Key DELETE error:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}
