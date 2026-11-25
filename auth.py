import os
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt

# In production (and anywhere this repo is public on GitHub), ALWAYS set the
# JWT_SECRET_KEY environment variable to a long, random value. Generate one
# with:  python -c "import secrets; print(secrets.token_hex(32))"
#
# If it's not set, we fall back to a randomly generated key for this process
# only — never a guessable hardcoded string that anyone reading a public repo
# could use to forge valid login tokens. The trade-off: tokens issued before
# a restart stop working after one, since the key changes every time the
# server starts without JWT_SECRET_KEY set. That's fine for local testing,
# but set the real environment variable before deploying anywhere public.
SECRET_KEY = os.environ.get("JWT_SECRET_KEY")


if not SECRET_KEY:
    SECRET_KEY = secrets.token_hex(32)
    print(
        "⚠️  JWT_SECRET_KEY environment variable not set — using a randomly "
        "generated key for this session only. Existing login sessions will "
        "need to sign in again every time the server restarts. Set "
        "JWT_SECRET_KEY before deploying anywhere public."
    )

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day

def get_password_hash(password: str) -> str:
    """Hashes a password using a random salt and SHA-256."""
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
    return f"{salt}${hashed}"

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain password against the stored salt$hash."""
    try:
        salt, expected_hash = hashed_password.split("$")
        calculated = hashlib.sha256((salt + plain_password).encode("utf-8")).hexdigest()
        return secrets.compare_digest(calculated, expected_hash)
    except Exception:
        return False

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_access_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None