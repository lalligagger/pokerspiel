# Google authentication for the solver API

This project already has a FastAPI app and a read-only API surface. The standard way to add a familiar “Sign in with Google” flow is to use Google OAuth 2.0 / OpenID Connect and protect the API with a secure session cookie.

## Recommended pattern

Use Google login on the frontend, then verify the identity on the backend:

1. User clicks “Sign in with Google”
2. Google authenticates the user and returns an ID token or auth code
3. FastAPI verifies the Google token
4. Backend creates a secure session cookie
5. Protected routes check the session before returning data

This is the same pattern used by most web apps that show a Google sign-in button.

## Why this fits this repo

The API surface is already cleanly organized in:

- [api/app.py](../api/app.py)
- [api/router.py](../api/router.py)

The app is a Python/FastAPI service, so auth belongs in the backend app layer, not in Docker startup commands or VM shell scripts.

## Best practical setup for this project

For a small internal or personal deployment, the simplest and safest setup is:

- keep read-only endpoints available only to authenticated users
- allow only your Google account or a small allowlist
- protect admin or compute-heavy routes as well
- leave the public firewall open only if you truly want a public read-only service

## Google setup

In Google Cloud Console:

1. Create a project
2. Enable OAuth 2.0 / Identity services
3. Create an OAuth 2.0 Client ID
4. Choose “Web application”
5. Add a redirect URI such as:
   - `https://your-domain.example.com/auth/callback`
   - `http://localhost:8000/auth/callback` for local testing

Do not put the client secret in frontend code. Keep it in environment variables on the server.

## Environment variables

Use something like:

```bash
export GOOGLE_CLIENT_ID="..."
export GOOGLE_CLIENT_SECRET="..."
export SESSION_SECRET_KEY="..."
```

## Minimal FastAPI flow

A simple approach is:

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from google.oauth2 import id_token
from google.auth.transport import requests
import os

app = FastAPI()
oauth2_scheme = HTTPBearer(auto_error=False)


def get_google_user(creds: HTTPAuthorizationCredentials | None):
    if creds is None:
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = creds.credentials
    try:
        idinfo = id_token.verify_oauth2_token(
            token,
            requests.Request(),
            os.environ["GOOGLE_CLIENT_ID"],
        )
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid Google token")

    return idinfo


@app.get("/me")
def me(user=Depends(get_google_user)):
    return {"email": user["email"]}
```

For browser-based cookies, the usual pattern is to verify the Google token server-side and then set an `HttpOnly` session cookie.

## Access control rule

After the user is verified, you can restrict by email:

```python
allowed = {"you@gmail.com", "team@yourdomain.com"}
if user["email"] not in allowed:
    raise HTTPException(status_code=403, detail="Forbidden")
```

This is the easiest production-safe policy for a small app.

## Important security notes

- Do not rely on frontend-only checks
- Do not expose the client secret to the browser
- Do not leave raw 0.0.0.0/0 firewall access open if you do not need it
- Validate the Google token on every protected request
- Prefer secure cookies with `HttpOnly`, `Secure`, and `SameSite` settings

## Practical recommendation for this repo

The next clean milestone is:

1. add Google login to the app
2. protect the read-only API routes with session auth
3. restrict to your own Google account(s)
4. only then decide whether to keep the VM public or gate it behind a reverse proxy / IAP layer

This is the simplest path from a public prototype to a workable authenticated API without making deployment much more complex.
