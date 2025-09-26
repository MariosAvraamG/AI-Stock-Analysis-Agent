import { NextRequest, NextResponse } from 'next/server'
import { createClient } from '@supabase/supabase-js'

export async function GET(request: NextRequest) {
  const { searchParams, origin } = new URL(request.url)
  const code = searchParams.get('code')
  const next = searchParams.get('next') ?? '/'

  console.log('Auth callback - Code:', code ? 'present' : 'missing')
  console.log('Auth callback - Origin:', origin)
  console.log('Auth callback - Next:', next)

  if (code) {
    const supabase = createClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
    )

    try {
      const { data, error } = await supabase.auth.exchangeCodeForSession(code)
      
      console.log('Exchange result - Error:', error)
      console.log('Exchange result - Session exists:', !!data?.session)
      
      if (!error && data?.session) {
        console.log('Redirecting to:', `${origin}${next}`)
        return NextResponse.redirect(`${origin}${next}`)
      } else {
        console.error('Auth callback error:', error)
        return NextResponse.redirect(`${origin}/auth/auth-code-error`)
      }
    } catch (err) {
      console.error('Auth callback exception:', err)
      return NextResponse.redirect(`${origin}/auth/auth-code-error`)
    }
  }

  console.log('No code parameter - redirecting to error page')
  return NextResponse.redirect(`${origin}/auth/auth-code-error`)
}
