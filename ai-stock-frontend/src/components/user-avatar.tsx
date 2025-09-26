import { User } from '@supabase/supabase-js'

interface UserAvatarProps {
  user: User
  size?: 'sm' | 'md' | 'lg'  // This is correct - it's optional with default
  showName?: boolean
  showWelcome?: boolean
  className?: string
}

export default function UserAvatar({ 
  user, 
  size = 'md', 
  showName = false, 
  showWelcome = false,
  className = '' 
}: UserAvatarProps) {
  const sizeClasses = {
    sm: 'h-6 w-6 text-xs',
    md: 'h-8 w-8 text-sm',
    lg: 'h-12 w-12 text-base'
  }

  const avatarUrl = user.user_metadata?.avatar_url || user.user_metadata?.picture
  const displayName = user.user_metadata?.full_name || user.email
  const initials = (displayName || 'U').charAt(0).toUpperCase()

  return (
    <div className={`flex items-center space-x-3 ${className}`}>
      <div className="relative">
        {avatarUrl ? (
          <>
            <img
              src={avatarUrl}
              alt={`${displayName}'s avatar`}
              className={`${sizeClasses[size]} rounded-full border-2 border-white/20 object-cover`}
              onError={(e) => {
                // Fallback to initials if image fails to load
                const target = e.currentTarget as HTMLImageElement
                target.style.display = 'none'
                const fallback = target.nextElementSibling as HTMLElement
                if (fallback) {
                  fallback.classList.remove('hidden')
                }
              }}
            />
            <div className={`${sizeClasses[size]} rounded-full bg-white/20 flex items-center justify-center text-white font-medium hidden absolute inset-0`}>
              {initials}
            </div>
          </>
        ) : (
          <div className={`${sizeClasses[size]} rounded-full bg-white/20 flex items-center justify-center text-white font-medium`}>
            {initials}
          </div>
        )}
      </div>
      {showName && (
        <span className="text-white/80 text-sm font-medium">
          {showWelcome ? `Welcome, ${displayName}` : displayName}
        </span>
      )}
    </div>
  )
}
