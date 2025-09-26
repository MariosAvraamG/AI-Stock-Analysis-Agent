# AI Stock Analysis API

A full-stack Next.js application providing AI-powered stock analysis with secure API key management and user authentication.

## 🚀 Features

- **Secure API Key Management**: Create, manage, and monitor API keys with granular permissions
- **User Authentication**: Supabase-powered authentication with social login support
- **Trading Signals API**: Access AI-generated stock trading signals
- **Modern UI**: Built with Next.js 15, React 19, and Tailwind CSS
- **Enterprise Security**: SHA-256 hashing, row-level security, and permission-based access control

## 🛠️ Tech Stack

- **Frontend**: Next.js 15, React 19, TypeScript
- **Styling**: Tailwind CSS, Radix UI components
- **Backend**: Next.js API Routes
- **Database**: Supabase (PostgreSQL)
- **Authentication**: Supabase Auth
- **Icons**: Lucide React

## 📋 Prerequisites

- Node.js 18+ 
- npm
- Supabase account and project

## ⚡ Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/MariosAvraamG/AI-Stock-Analysis-API
npm install
```

### 2. Environment Setup

Create a `.env.local` file in the project root:

```env
# Supabase Configuration
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url_here
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key_here

# Service Role Key (REQUIRED for API key creation)
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key_here
```

**To get your Supabase keys:**
1. Go to your Supabase project dashboard
2. Navigate to **Settings** → **API**
3. Copy the required values:
   - **URL**: `NEXT_PUBLIC_SUPABASE_URL`
   - **anon public**: `NEXT_PUBLIC_SUPABASE_ANON_KEY` 
   - **service_role secret**: `SUPABASE_SERVICE_ROLE_KEY` ⚠️ **Critical for API key creation**

### 3. Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the application.

## 🔐 API Key Management

### Creating API Keys

1. **Sign in** to your account
2. **Navigate** to `/api-keys`
3. **Click** "Create New API Key"
4. **Configure** your key:
   - **Name**: Descriptive identifier (e.g., "Production API", "Mobile App")
   - **Permissions**: Read access (always included) + optional write access
   - **Expiration**: Optional expiration date for enhanced security
5. **⚠️ Important**: Copy the generated API key immediately - it won't be shown again!

### Permission System

Following industry best practices (GitHub, Stripe, Auth0):

- **✅ Read Access**: Always enabled - required for basic API functionality
- **☐ Write Access**: Optional - enable for future write operations (watchlists, alerts, portfolio)

### Managing API Keys

- **View All Keys**: See all your API keys with status and usage information
- **Activate/Deactivate**: Toggle keys on/off without deleting them
- **Delete**: Permanently remove API keys (cannot be undone)
- **Monitor Usage**: Track when keys were last used

## 🔌 API Usage

### Authentication

Include your API key in the `Authorization` header:

```bash
curl -X POST -H "Authorization: Bearer ak_your_api_key_here" \
     -H "Content-Type: application/json" \
     -d '{"ticker": "AAPL"}' \
     "http://localhost:8000/ai-agent/signals"
```

### JavaScript Example

```javascript
const response = await fetch('http://localhost:8000/ai-agent/signals', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer ak_your_api_key_here',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({ ticker: 'AAPL' })
});

const data = await response.json();
console.log(data);
```

### Response Format

```json
{
  "ticker": "AAPL",
  "timestamp": "2024-01-15T10:30:00Z",
  "signals": {
    "short_term": {
      "signal": "BUY",
      "confidence": 0.85,
      "reasoning": "Technical analysis shows strong momentum with RSI indicating bullish conditions."
    },
    "medium_term": {
      "signal": "HOLD",
      "confidence": 0.62,
      "reasoning": "Mixed signals with consolidation patterns suggesting neutral outlook."
    },
    "long_term": {
      "signal": "BUY",
      "confidence": 0.73,
      "reasoning": "Strong fundamental metrics and favorable growth prospects support long-term position."
    }
  },
  "execution_time_seconds": 45.2,
  "tools_used": ["technical_analysis", "ml_prediction", "sentiment_analysis"],
  "success": true
}
```

## 📊 API Endpoints

### API Key Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/api-keys` | List user's API keys |
| `POST` | `/api/api-keys` | Create new API key |
| `PUT` | `/api/api-keys/[id]` | Update API key |
| `DELETE` | `/api/api-keys/[id]` | Delete API key |

### Protected Endpoints

| Method | Endpoint | Description | Permissions |
|--------|----------|-------------|-------------|
| `POST` | `/ai-agent/signals` | Generate AI trading signals | Read |

## 🛡️ Security Features

### Key Security
- **SHA-256 Hashing**: API keys are hashed before storage
- **Prefix Display**: Only first 12 characters shown for identification
- **One-time Display**: Full keys only shown during creation
- **Secure Generation**: Uses `crypto.randomBytes` for key generation

### Access Control
- **Row Level Security**: Database-level access control
- **User Isolation**: Users can only access their own keys
- **Permission-based**: Granular read/write permissions per key
- **Expiration Support**: Optional time-limited access

### Monitoring
- **Usage Tracking**: Last used timestamps
- **Activity Logging**: Monitor API key usage patterns
- **Instant Revocation**: Deactivate compromised keys immediately

## 🔧 Development

### Adding New Protected Endpoints

1. **Import the middleware**:
```typescript
import { authenticateApiKey } from '@/lib/auth-middleware'
```

2. **Use in your API route**:
```typescript
export async function GET(request: NextRequest) {
  // Authenticate the API key
  const authResult = await authenticateApiKey(request)
  
  if (!authResult.success) {
    return NextResponse.json(
      { error: authResult.error },
      { status: authResult.status }
    )
  }

  // Check permissions
  if (!authResult.apiKey.permissions.read) {
    return NextResponse.json(
      { error: 'Insufficient permissions' },
      { status: 403 }
    )
  }

  // Your API logic here
  return NextResponse.json({ data: 'success' })
}
```

### Error Handling

The middleware returns standardized error responses:
- **401 Unauthorized**: Missing or invalid API key
- **403 Forbidden**: Insufficient permissions
- **400 Bad Request**: Malformed request

## 📝 Best Practices

### For API Users
1. **Use descriptive names** for API keys
2. **Set expiration dates** for enhanced security
3. **Regularly rotate keys** in production
4. **Monitor usage** to detect anomalies
5. **Revoke unused keys** immediately

### For Developers
1. **Always check permissions** before processing requests
2. **Log API key usage** for monitoring
3. **Rate limit** API endpoints appropriately
4. **Validate all inputs** before processing
5. **Use HTTPS** in production

## 🚨 Troubleshooting

### Common Issues

**"User not authenticated" error**
- Ensure user is signed in to Supabase
- Check RLS policies are correctly set

**"Invalid API key" error**
- Verify the key format (should start with `ak_`)
- Check if the key is active and not expired
- Ensure the key exists in the database

**Permission errors**
- Verify the API key has the required permissions
- Check if the key is active

### Database Queries for Debugging

```sql
-- Check API keys for a user
SELECT * FROM api_keys WHERE user_id = 'user_uuid';

-- Check active keys
SELECT * FROM api_keys WHERE is_active = true;

-- Check expired keys
SELECT * FROM api_keys WHERE expires_at < NOW();
```

## 📈 Architecture

### Authentication Flow
```
Client Request (with JWT) 
    ↓
Server-side Authentication (Service Role Key validates JWT)
    ↓
Database Operations (Service Role Key with admin privileges)
    ↓
Response to authenticated user
```

### Security Benefits
1. **🔐 JWT Validation**: Properly validates user tokens server-side
2. **🛡️ Admin Operations**: Uses admin privileges for reliable database access
3. **🔒 User Isolation**: Enforces user ownership via `user_id` filtering
4. **⚡ No RLS Dependencies**: Bypasses potential policy conflicts

## 🔄 Deployment

### Build and Deploy

```bash
npm run build
npm run start
```

### Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out the [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review Supabase logs for errors
3. Verify database permissions and policies
4. Check browser console for client-side errors

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request