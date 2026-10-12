# Platform pilot â€” operating guide

This is an opt-in research pilot, not a production SLA or a fraud detector.
No current real-listing accuracy claim is established. Public results predate
required category checks. Do not automatically reject or approve listings.

## Deployment

- Single uvicorn worker on the current free Render service.
- Set JWT_SECRET to a random secret with at least 32 characters. Without a
  configured secret, an unpredictable process-local key is used; restarting
  invalidates sessions and guest cookies. Known fallback keys are never used.
- ALLOWED_ORIGINS is a comma-separated exact browser-origin allowlist. Default:
  https://ai-image-detector.com,https://www.ai-image-detector.com. A Render browser
  origin may be added explicitly. CORS is not authentication.
- DB_PATH optionally points to an existing durable storage location. The default
  users.db on free Render remains ephemeral. Do not claim durable accounts,
  credits, orders or backups without provisioning and testing persistence.
- ENTERPRISE_API_KEYS is a comma-separated list of random >=32-character pilot
  secrets. Leave unset to keep /v1/analyze unavailable. Give keys only to the
  intended pilot's backend; never embed them in a browser or public demo.
- The old debug endpoint and unauthenticated Gumroad webhook return 410.
  No new payment webhook or paid credit delivery is enabled by this release.

## API

POST /v1/analyze: multipart file plus mode=general|car|realestate.
Header X-API-Key authenticates the server caller. Upload limits: 20 MiB, 20 MP.
The same category and detector pipeline as /predict is used. Image URL fetching
is deliberately not offered by the pilot endpoint.

```sh
curl "$DETECTOR_BASE_URL/v1/analyze" \
  -H "X-API-Key: $DETECTOR_API_KEY" \
  -F "file=@authorized-photo.jpg" -F "mode=car"
```

Response: request_id, status, mode, ai_model_score, score_type, verdict,
recommendation, fraud_assessment, category_check, model_breakdown,
evidence_report and limitations. no_ai_flag means no AI flag from this model;
it does NOT mean authentic, safe or approved. human_review is a review-routing
suggestion. Feedback tokens, image hashes and EXIF values are not returned.

401 invalid key; 422 missing upload/category mismatch; 429 throttled;
503 disabled pilot, unavailable model or busy engine. Retry-After on throttling.
Success score is not a calibrated probability. Never treat a redirect,
HTTP 200 on /healthz, or a test fixture as detection-accuracy evidence.

/healthz checks the web process and reports whether the model is warmed; it does
not run inference and is not an end-to-end model health or accuracy test.
/docs is the generated schema. /business.html explains the pilot.

## Current safeguards and limits

Rate limits are per process (single-worker requirement): 12 auth POSTs / minute,
30 analysis/feedback requests / minute per verified account or pilot key.
Unauthenticated callers use the observed peer, not untrusted forwarding headers.
An upstream proxy may group guest callers; validate this in deployment. Counters
expire and are bounded. Restarts reset counters; use an external limiter before
scaling to multiple workers. Failed requests count toward rate limits.

URL downloads validate public IPv4/IPv6 destinations, reject credentials,
nonstandard ports and private DNS answers, pin the validated destination and
retain HTTPS hostname/certificate checks. Each redirect is revalidated. No
environment proxy, caller cookies or authentication headers are forwarded.
Decoded response bytes and upload reads are bounded. DNS resolution may still
depend on the platform resolver timeout; production network controls remain
necessary for a contracted deployment.

## Release evidence

Run tests/test_service.py, frontend syntax checks and browser checks in both
languages at desktop/mobile widths. Automated model tests use controlled
fixtures for failure/auth behavior, not to claim accuracy. For inference smoke
tests use authorized research examples with the pinned local weights. Record
model revision, category version, threshold, latency and complete failures.

Before any commercial pilot: obtain authorized representative data, run the
current full workflow, measure false flags/misses/uncertainty/category rejection
and p50/p95 latency. Establish a baseline with the customer's existing process.
Document storage, deletion, provider agreements, incident handling and support
contact. Conduct an independent security review before production commitments.
