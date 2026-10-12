"""Encrypted offline backup; restore refuses a non-empty destination.

Usage: python backup_storage.py backup backup.aidb
       python backup_storage.py restore backup.aidb
Set BACKUP_PASSPHRASE in the process environment (never a command argument).
This command does not schedule backups or upload them anywhere.
"""
import argparse
import json
import os
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
import storage

MAGIC = b'AIDB01'
FIELDS = {
    'users': ('id', 'email', 'password_hash', 'google_id', 'is_verified', 'is_pro', 'credits', 'otp_code', 'otp_expires', 'created_at'),
    'feedback': ('id', 'image_hash', 'ai_score', 'predicted', 'actual', 'mode', 'created_at'),
    'guest_trials': ('guest_id', 'used'),
}


def key(passphrase, salt):
    if len(passphrase) < 24:
        raise ValueError('Use a backup passphrase of at least 24 characters.')
    return Scrypt(salt=salt, length=32, n=2**15, r=8, p=1).derive(passphrase.encode())


def encrypt(snapshot, passphrase):
    salt, nonce = os.urandom(16), os.urandom(12)
    header = MAGIC + salt + nonce
    payload = json.dumps(snapshot, ensure_ascii=False, allow_nan=False).encode()
    return header + AESGCM(key(passphrase, salt)).encrypt(nonce, payload, header)


def decrypt(data, passphrase):
    if not data.startswith(MAGIC) or len(data) < 50:
        raise ValueError('Unsupported backup format.')
    salt, nonce, header = data[6:22], data[22:34], data[:34]
    payload = AESGCM(key(passphrase, salt)).decrypt(nonce, data[34:], header)
    snapshot = json.loads(payload)
    if not isinstance(snapshot, dict) or snapshot.get('version') != 1 or set(snapshot.get('tables', {})) != set(FIELDS):
        raise ValueError('Unsupported backup schema.')
    for table, fields in FIELDS.items():
        rows = snapshot['tables'][table]
        if not isinstance(rows, list) or any(not isinstance(row, dict) or set(row) != set(fields) for row in rows):
            raise ValueError('Invalid backup rows.')
    return snapshot


def snapshot():
    conn = storage.get_db()
    try:
        if isinstance(conn, storage.PostgresConnection):
            conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        else:
            conn.execute('BEGIN')
        tables = {table: [dict(row) for row in conn.execute(f"SELECT {','.join(fields)} FROM {table}").fetchall()]
                  for table, fields in FIELDS.items()}
        return {'version': 1, 'tables': tables}
    finally:
        conn.rollback()
        conn.close()


def restore(data):
    conn = storage.get_db()
    postgres = isinstance(conn, storage.PostgresConnection)
    try:
        if postgres:
            conn.execute('LOCK TABLE users, feedback, guest_trials IN ACCESS EXCLUSIVE MODE')
        else:
            conn.execute('BEGIN IMMEDIATE')
        if any(conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0] for table in FIELDS):
            raise ValueError('Restore requires an empty database; existing data was not changed.')
        for table, fields in FIELDS.items():
            for row in data['tables'][table]:
                conn.execute(f"INSERT INTO {table} ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})",
                             tuple(row[field] for field in fields))
            if postgres and 'id' in fields:
                conn.execute(f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), COALESCE(MAX(id), 1), MAX(id) IS NOT NULL) FROM {table}")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['backup', 'restore'])
    parser.add_argument('file', type=Path)
    args = parser.parse_args()
    passphrase = os.environ.get('BACKUP_PASSPHRASE', '')
    try:
        if args.action == 'backup':
            encrypted = encrypt(snapshot(), passphrase)
            # Exclusive creation prevents silently replacing the last good backup.
            with args.file.open('xb') as handle:
                handle.write(encrypted)
        else:
            data = decrypt(args.file.read_bytes(), passphrase)
            storage.initialize()
            restore(data)
    except Exception as error:
        # Error strings from drivers or malformed rows may contain private data.
        parser.exit(1, f'Backup operation failed ({type(error).__name__}); no credentials or rows logged.\n')
    print(f'{args.action} completed.')


if __name__ == '__main__':
    main()
