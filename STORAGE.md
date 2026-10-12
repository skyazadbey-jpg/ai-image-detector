# Durable account storage

Production uses `DATABASE_URL` with PostgreSQL. SQLite remains available only
when that variable is absent. A configured PostgreSQL failure stops startup;
it must never silently create a second empty account database.

Set the server-only environment variable using the Supabase session pooler
connection shown in the dashboard. The password may be omitted from that URI
and set separately as server-only `DATABASE_PASSWORD`, avoiding URI encoding.
Require `sslmode=verify-full` and configure
`DATABASE_SSL_ROOT_CERT` to a trusted CA certificate file if the provider needs
one. Do not put connection strings, passwords or signing secrets into Git,
frontend JavaScript, screenshots or chat. Use a stable server-only
`JWT_SECRET` of at least 32 characters.

The app creates `detector_private`, outside the exposed Data API schemas.
Keep Supabase Data API disabled for this service. Table RLS is enabled; the
server database owner can access tables. Production should eventually use a
dedicated least-privilege database role instead of the project owner.

Before cutover, obtain a consistent backup of the existing SQLite database.
An empty Supabase database does not contain existing Render accounts. Existing
account preservation cannot be claimed without recovering and validating the
source database. Do not deploy a cutover over unrecovered production data.

## Encrypted offline backups

Install dependencies and set `BACKUP_PASSPHRASE` securely in the process
environment to at least 24 characters. Run `python backup_storage.py backup
backup.aidb`. The backup contains private account and feedback data encrypted
with AES-GCM and a scrypt-derived key. Keep the passphrase separately. File
creation is exclusive, so a prior backup is not overwritten.

Restore with `python backup_storage.py restore backup.aidb`, pointing
`DATABASE_URL` or `DB_PATH` at an **empty** destination. Restore is transactional
and refuses a non-empty database. Test recovery in a separate environment.
Neither this command nor creating a Supabase project schedules backups.
Free-tier persistence alone is not a backup or availability guarantee.

`/readyz` checks the connection and exposes only backend type and whether
persistent storage and stable sessions are configured. It does not validate
backup recovery, model accuracy or service availability guarantees.

Tests: `python tests/test_storage.py` verifies local account persistence,
explicit failure rather than fallback, SQL parameter adaptation, encrypted
round-trip, tampering/wrong-passphrase rejection, and overwrite refusal.
Actual PostgreSQL connectivity still needs deployment verification.
