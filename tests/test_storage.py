import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import storage
import backup_storage


class StorageTests(unittest.TestCase):
    def test_postgres_uses_separate_password_and_verified_tls(self):
        from unittest.mock import Mock
        raw = Mock()
        with patch.dict(os.environ, {'DATABASE_URL': 'postgresql://owner@database.example/db?sslmode=verify-full',
                                     'DATABASE_PASSWORD': 'fixture-password-with-@-symbols'}):
            with patch('psycopg.connect', return_value=raw) as connect:
                connection = storage.get_db()
                self.assertIsInstance(connection, storage.PostgresConnection)
                self.assertEqual(connect.call_args.kwargs['password'], 'fixture-password-with-@-symbols')
                self.assertEqual(connect.call_args.kwargs['sslmode'], 'verify-full')
                self.assertEqual(Path(connect.call_args.kwargs['sslrootcert']).name, 'supabase-ca.crt')
                connection.close()

    def test_encrypted_backup_restore_and_overwrite_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            source = str(Path(directory) / 'source.db')
            destination = str(Path(directory) / 'destination.db')
            with patch.dict(os.environ, {'DATABASE_URL': '', 'DB_PATH': source}):
                storage.initialize()
                conn = storage.get_db()
                conn.execute('INSERT INTO users (email,password_hash,is_verified) VALUES (?,?,?)',
                             ('fixture@example.invalid', 'private-test-hash', 1))
                conn.commit()
                conn.close()
                blob = backup_storage.encrypt(backup_storage.snapshot(), 'long-test-passphrase-123456789')
            self.assertNotIn(b'private-test-hash', blob)
            data = backup_storage.decrypt(blob, 'long-test-passphrase-123456789')
            with patch.dict(os.environ, {'DATABASE_URL': '', 'DB_PATH': destination}):
                storage.initialize()
                backup_storage.restore(data)
                with self.assertRaises(ValueError):
                    backup_storage.restore(data)
                conn = storage.get_db()
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM users').fetchone()[0], 1)
                conn.close()
            with self.assertRaises(Exception):
                backup_storage.decrypt(blob[:-1] + bytes([blob[-1] ^ 1]), 'long-test-passphrase-123456789')
            with self.assertRaises(Exception):
                backup_storage.decrypt(blob, 'different-test-passphrase-123456789')

    def test_accounts_survive_reinitialization(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {'DATABASE_URL': '', 'DB_PATH': str(Path(directory) / 'users.db')}):
                storage.initialize()
                conn = storage.get_db()
                conn.execute('INSERT INTO users (email,password_hash,is_verified) VALUES (?,?,?)',
                             ('fixture@example.invalid', 'test-hash', 1))
                conn.commit()
                conn.close()
                storage.initialize()
                conn = storage.get_db()
                row = conn.execute('SELECT * FROM users').fetchone()
                self.assertEqual(row['password_hash'], 'test-hash')
                self.assertEqual(row['is_verified'], 1)
                self.assertEqual(storage.check()['backend'], 'sqlite')
                conn.close()

    def test_configured_database_never_silently_falls_back(self):
        with patch.dict(os.environ, {'DATABASE_URL': 'postgresql://user:secret@localhost/db?sslmode=disable'}):
            with self.assertRaisesRegex(RuntimeError, '^Persistent database unavailable or misconfigured.$'):
                storage.get_db()

    def test_postgres_adapter_preserves_parameters_and_conflict_handling(self):
        from unittest.mock import Mock
        raw = Mock()
        conn = storage.PostgresConnection(raw)
        conn.execute('INSERT OR IGNORE INTO guest_trials (guest_id,used) VALUES (?,?)', ('fixture', 0))
        raw.execute.assert_called_once_with(
            'INSERT INTO guest_trials (guest_id,used) VALUES (%s,%s) ON CONFLICT (guest_id) DO NOTHING', ('fixture', 0))
        row = storage.Row([('count', 3)])
        self.assertEqual(row[0], 3)
        self.assertEqual(row['count'], 3)


if __name__ == '__main__':
    unittest.main()
