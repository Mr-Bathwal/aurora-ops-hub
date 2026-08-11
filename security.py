"""Password hashing, opaque tokens, and encryption of stored credentials.

Three deliberate choices worth stating, because each has an obvious alternative that is worse
here:

1. **scrypt from the standard library, not bcrypt/argon2.** scrypt is memory-hard, is in
   `hashlib`, and at the parameters below costs ~64 MB and ~100 ms per verification — which is
   the point: it prices out offline cracking. bcrypt would mean a compiled dependency to do
   the same job slightly worse (bcrypt is only CPU-hard, and silently truncates at 72 bytes).

2. **Opaque random session tokens, not JWTs.** A JWT cannot be revoked without keeping a
   denylist, at which point you have the database lookup you were trying to avoid *plus* a
   token you cannot invalidate. A 256-bit random string checked against a table is simpler,
   revocable instantly, and leaks nothing if decoded.

3. **Only the hash of every secret is stored.** Session tokens, enrolment tokens and agent
   keys are all shown once and stored as SHA-256. Plain SHA-256 is correct here and would be
   wrong for passwords: these are 256-bit random values, so there is no dictionary to attack
   and no need for a slow KDF.
"""

import base64
import hashlib
import hmac
import os
import secrets

from cryptography.fernet import Fernet, InvalidToken

# --- password hashing -------------------------------------------------------------------

# OWASP's floor for scrypt is n=2^15, r=8, p=1 (~32 MB). Doubled here: this is an
# infrastructure control plane, and logins are rare enough that 100 ms is invisible.
_SCRYPT_N = 2 ** 16
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32
_SALT_BYTES = 16


def hash_password(password: str) -> str:
    """Returns 'scrypt$n$r$p$salt_b64$hash_b64'. Parameters travel with the hash so they can
    be raised later without invalidating every existing password."""
    if not password or len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    salt = os.urandom(_SALT_BYTES)
    dk = hashlib.scrypt(
        password.encode("utf-8"), salt=salt,
        n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=_SCRYPT_DKLEN,
        maxmem=_SCRYPT_N * _SCRYPT_R * 256,
    )
    return "scrypt${}${}${}${}${}".format(
        _SCRYPT_N, _SCRYPT_R, _SCRYPT_P,
        base64.b64encode(salt).decode(), base64.b64encode(dk).decode(),
    )


def verify_password(password: str, stored: str) -> bool:
    """Constant-time verify. Returns False on any malformed hash rather than raising — a
    corrupt row must fail closed, not 500."""
    try:
        scheme, n, r, p, salt_b64, hash_b64 = stored.split("$")
        if scheme != "scrypt":
            return False
        n, r, p = int(n), int(r), int(p)
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
        dk = hashlib.scrypt(
            password.encode("utf-8"), salt=salt,
            n=n, r=r, p=p, dklen=len(expected), maxmem=n * r * 256,
        )
        return hmac.compare_digest(dk, expected)
    except Exception:
        return False


# --- opaque tokens ----------------------------------------------------------------------

def new_token(prefix: str = "") -> str:
    """A 256-bit URL-safe random token. The prefix is cosmetic but load-bearing in practice:
    it makes a leaked key identifiable on sight (and greppable by secret scanners)."""
    return f"{prefix}{secrets.token_urlsafe(32)}"


def token_digest(token: str) -> str:
    """What actually goes in the database."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def tokens_match(presented: str, stored_digest: str) -> bool:
    return hmac.compare_digest(token_digest(presented), stored_digest)


# --- credential encryption --------------------------------------------------------------

_ENV_KEY = "ITOPS_SECRET_KEY"


def _load_fernet() -> Fernet:
    """The key comes from the environment, never the database — storing both together means a
    single stolen file contains the lock and the key. In development a key is derived from a
    fixed local string so the app still boots, and that path is loudly not for production."""
    raw = os.environ.get(_ENV_KEY)
    if raw:
        try:
            return Fernet(raw.encode() if isinstance(raw, str) else raw)
        except Exception as exc:
            raise RuntimeError(
                f"{_ENV_KEY} is set but is not a valid Fernet key. "
                f"Generate one with: python -c \"from cryptography.fernet import Fernet; "
                f"print(Fernet.generate_key().decode())\""
            ) from exc
    dev_key = base64.urlsafe_b64encode(
        hashlib.sha256(b"itops-hub-development-key-not-for-production").digest()
    )
    return Fernet(dev_key)


def is_production_key() -> bool:
    """Surfaced on the health endpoint so 'we shipped with the dev key' is visible rather
    than discovered."""
    return bool(os.environ.get(_ENV_KEY))


def encrypt_secret(plaintext: str) -> bytes:
    return _load_fernet().encrypt(plaintext.encode("utf-8"))


def decrypt_secret(ciphertext: bytes) -> str:
    try:
        return _load_fernet().decrypt(ciphertext).decode("utf-8")
    except InvalidToken as exc:
        raise RuntimeError(
            "Could not decrypt a stored credential — the encryption key has changed "
            "since it was saved. The credential must be re-entered."
        ) from exc
