# Local Development Notes

## Environment

1. Copy `.env.example` to `.env`.
2. Keep cookie defaults for same-origin local development:
   - `SESSION_COOKIE_SAMESITE=Lax`
   - `CSRF_COOKIE_SAMESITE=Lax`
   - `SESSION_COOKIE_SECURE=0`
   - `CSRF_COOKIE_SECURE=0`

`backend/settings.py` already has local defaults, so `.env` is optional for a basic run.

## Recommended Frontend Integration (Dev)

Use a Vite proxy so browser requests stay same-origin from the backend point of view:
- proxy `/api/*` and `/api-auth/*` to Django (`http://127.0.0.1:8000`)
- use `credentials: "include"` in the frontend API client

This avoids CORS complexity during development and works with session cookies + CSRF.

## Optional Cross-Origin Setup

If you must run frontend/backend on different origins, configure:
- `CSRF_TRUSTED_ORIGINS`
- `CORS_ALLOWED_ORIGINS`
- `CORS_ALLOW_CREDENTIALS`
- secure cookie flags for HTTPS environments (`*_SECURE=1`, often `SameSite=None`)

## API Auth Flow (Session + CSRF)

1. `GET /api/auth/csrf`
2. `POST /api/auth/login` with `X-CSRFToken` and `credentials: include`
3. `GET /api/auth/me`
4. `POST /api/auth/logout` with `X-CSRFToken`
