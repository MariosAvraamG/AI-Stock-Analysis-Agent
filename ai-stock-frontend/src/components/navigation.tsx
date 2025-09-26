'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Menu, X, BarChart3, LogIn, UserPlus, LogOut, User } from 'lucide-react'
import Link from 'next/link'
import { useAuth } from '@/lib/auth-context'
import UserAvatar from './user-avatar'

export default function Navigation() {
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const { user, signOut, loading } = useAuth()

  return (
    <nav className="relative z-10 bg-white/10 backdrop-blur-md border-b border-white/20">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link href="/" className="flex items-center space-x-2">
            <BarChart3 className="h-8 w-8 text-yellow-400" />
            <span className="text-xl font-bold text-white">TradingAI</span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-8">
            <Link href="#features" className="text-white/80 hover:text-white transition-colors">
              Features
            </Link>
            <Link href="#demo" className="text-white/80 hover:text-white transition-colors">
              Demo
            </Link>
            <Link href="/api-keys" className="text-white/80 hover:text-white transition-colors">
              API Keys
            </Link>
            <Link href="/pricing" className="text-white/80 hover:text-white transition-colors">
              Pricing
            </Link>
          </div>

          {/* Desktop Auth Buttons */}
          <div className="hidden md:flex items-center space-x-4">
            {loading ? (
              <div className="h-10 w-24 bg-white/10 rounded animate-pulse"></div>
            ) : user ? (
              <div className="flex items-center space-x-4">
                <UserAvatar user={user} showName={true} showWelcome={true} />
                <Button 
                  variant="ghost" 
                  className="text-white hover:bg-white/10 border-white/30"
                  onClick={() => signOut()}
                >
                  <LogOut className="mr-2 h-4 w-4" />
                  Sign Out
                </Button>
              </div>
            ) : (
              <>
                <Link href="/signin">
                  <Button variant="ghost" className="text-white hover:bg-white/10 border-white/30">
                    <LogIn className="mr-2 h-4 w-4" />
                    Sign In
                  </Button>
                </Link>
                <Link href="/signup">
                  <Button className="bg-white text-blue-600 hover:bg-blue-50 font-semibold">
                    <UserPlus className="mr-2 h-4 w-4" />
                    Sign Up
                  </Button>
                </Link>
              </>
            )}
          </div>

          {/* Mobile menu button */}
          <button
            className="md:hidden text-white"
            onClick={() => setIsMenuOpen(!isMenuOpen)}
          >
            {isMenuOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
          </button>
        </div>

        {/* Mobile Navigation */}
        {isMenuOpen && (
          <div className="md:hidden py-4 border-t border-white/20">
            <div className="flex flex-col space-y-4">
              <Link 
                href="#features" 
                className="text-white/80 hover:text-white transition-colors py-2"
                onClick={() => setIsMenuOpen(false)}
              >
                Features
              </Link>
              <Link 
                href="#demo" 
                className="text-white/80 hover:text-white transition-colors py-2"
                onClick={() => setIsMenuOpen(false)}
              >
                Demo
              </Link>
              <Link 
                href="/api-keys" 
                className="text-white/80 hover:text-white transition-colors py-2"
                onClick={() => setIsMenuOpen(false)}
              >
                API Keys
              </Link>
              <Link 
                href="/pricing" 
                className="text-white/80 hover:text-white transition-colors py-2"
                onClick={() => setIsMenuOpen(false)}
              >
                Pricing
              </Link>
              <div className="flex flex-col space-y-2 pt-4 border-t border-white/20">
                {loading ? (
                  <div className="h-10 bg-white/10 rounded animate-pulse"></div>
                ) : user ? (
                  <div className="space-y-2">
                    <div className="py-2">
                      <UserAvatar user={user} showName={true} showWelcome={true} className="text-sm" />
                    </div>
                    <Button 
                      variant="ghost" 
                      className="w-full text-white hover:bg-white/10 border-white/30 justify-start"
                      onClick={() => {
                        signOut()
                        setIsMenuOpen(false)
                      }}
                    >
                      <LogOut className="mr-2 h-4 w-4" />
                      Sign Out
                    </Button>
                  </div>
                ) : (
                  <>
                    <Link href="/signin">
                      <Button variant="ghost" className="w-full text-white hover:bg-white/10 border-white/30 justify-start">
                        <LogIn className="mr-2 h-4 w-4" />
                        Sign In
                      </Button>
                    </Link>
                    <Link href="/signup">
                      <Button className="w-full bg-white text-blue-600 hover:bg-blue-50 font-semibold justify-start">
                        <UserPlus className="mr-2 h-4 w-4" />
                        Sign Up
                      </Button>
                    </Link>
                  </>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </nav>
  )
}
