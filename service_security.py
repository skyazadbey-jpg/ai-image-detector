"""Small, testable controls for the single-worker public and pilot service."""
import hashlib
import ipaddress
import os
import secrets
import socket
import threading
import time
from collections import OrderedDict, deque
from urllib.parse import urlsplit

def signing_secret():
    value = os.getenv('JWT_SECRET', '')
    if len(value) >= 32 and value != 'change-me-in-production':
        return value, True
    # Never authenticate with a publicly known fallback key. This fallback logs
    # out sessions after a restart; operators should configure a stable secret.
    return secrets.token_urlsafe(48), False

def public_image_url(url):
    """Validate every destination, including redirects. Reject mixed DNS answers."""
    parsed = urlsplit(url)
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('Only public HTTP/HTTPS image URLs are supported.')
    try:
        port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    except ValueError:
        raise ValueError('Invalid image URL port.') from None
    if port not in (80, 443):
        raise ValueError('Only standard HTTP/HTTPS ports are supported.')
    host = parsed.hostname.rstrip('.')
    if host.lower() == 'localhost' or host.lower().endswith(('.localhost', '.local', '.internal')):
        raise ValueError('Private destinations are not supported.')
    try:
        answers = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        raise ValueError('Image hostname could not be resolved.') from None
    if not answers or any(not ipaddress.ip_address(item[4][0]).is_global for item in answers):
        raise ValueError('Private destinations are not supported.')
    return parsed, sorted({item[4][0] for item in answers})

def download_public_image(url, max_bytes=20 * 1024 * 1024):
    """Pin validated DNS address while retaining HTTPS certificate validation.

    Do not use environment proxies or allow a second hostname resolution between
    validation and connection. Revalidate each redirect; never forward cookies.
    """
    import certifi
    import urllib3
    from urllib.parse import urljoin
    for _ in range(4):
        parsed, addresses = public_image_url(url)
        port = parsed.port or (443 if parsed.scheme == 'https' else 80)
        host = parsed.hostname.rstrip('.')
        kwargs = {'host': addresses[0], 'port': port, 'timeout': urllib3.Timeout(connect=5, read=10)}
        if parsed.scheme == 'https':
            pool = urllib3.HTTPSConnectionPool(**kwargs, assert_hostname=host,
                server_hostname=host, cert_reqs='CERT_REQUIRED', ca_certs=certifi.where())
        else:
            pool = urllib3.HTTPConnectionPool(**kwargs)
        response = None
        try:
            target = (parsed.path or '/') + ('?' + parsed.query if parsed.query else '')
            response = pool.request('GET', target, headers={'Host': host, 'Accept': 'image/*',
                'User-Agent': 'AIImageDetector/1.0'}, preload_content=False, redirect=False, retries=False)
            if response.status in (301, 302, 303, 307, 308):
                location = response.headers.get('Location')
                if not location:
                    raise ValueError('Image redirect has no destination.')
                url = urljoin(url, location)
                continue
            if response.status != 200:
                raise ValueError('Image download failed.')
            chunks, size = [], 0
            for chunk in response.stream(64 * 1024, decode_content=True):
                size += len(chunk)
                if size > max_bytes:
                    raise ValueError('Image exceeds the 20 MB limit.')
                chunks.append(chunk)
            return b''.join(chunks), response.headers.get('content-type', 'application/octet-stream')
        finally:
            if response is not None:
                response.close()
            pool.close()
    raise ValueError('Too many image redirects.')

class WindowLimiter:
    """Bounded memory; monotonic clock; no raw IP/email retained."""
    def __init__(self, max_keys=10000, clock=time.monotonic):
        self.entries = OrderedDict()
        self.lock = threading.Lock()
        self.max_keys = max_keys
        self.clock = clock

    def allow(self, identity, bucket, limit, seconds=60):
        key = (hashlib.sha256(identity.encode()).hexdigest(), bucket)
        now = self.clock()
        with self.lock:
            # Expire inactive keys before deciding capacity; never evict active
            # keys to allow an attacker to reset other clients' counters.
            while self.entries and next(iter(self.entries.values()))[0] <= now - seconds:
                self.entries.popitem(last=False)
            entry = self.entries.pop(key, None)
            if entry is None and len(self.entries) >= self.max_keys:
                return False
            hits = entry[1] if entry else deque()
            while hits and hits[0] <= now - seconds:
                hits.popleft()
            allowed = len(hits) < limit
            if allowed:
                hits.append(now)
            self.entries[key] = (now, hits)
            return allowed
