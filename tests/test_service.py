import io
import asyncio
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
temporary = tempfile.TemporaryDirectory()
os.environ['DB_PATH'] = str(Path(temporary.name) / 'users.db')
os.environ['JWT_SECRET'] = 'test-secret-only-never-use-in-production-12345'
os.environ['ENTERPRISE_API_KEYS'] = 'test-pilot-key-only-never-use-in-production'
from fastapi.testclient import TestClient
from PIL import Image
import app
import service_security as security

class ServiceTests(unittest.TestCase):
    def test_concurrent_requests_busy_without_consuming_guest_trial(self):
        import httpx
        async def scenario():
            entered, release = asyncio.Event(), asyncio.Event()
            async def infer(*args):
                entered.set()
                await release.wait()
                return 80.0, [{'model': 'concurrency-fixture', 'ai_score': 80.0}]
            app.ANALYSIS_GATE = asyncio.Lock()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app.app), base_url='https://testserver') as client:
                await client.get('/auth/guest')
                with patch.object(app, 'call_ai_model_ensemble', infer):
                    active = asyncio.create_task(client.post('/v1/analyze', headers={'X-API-Key': os.environ['ENTERPRISE_API_KEYS']},
                        files={'file': ('sample.jpg', self.image, 'image/jpeg')}))
                    await asyncio.wait_for(entered.wait(), 5)
                    try:
                        busy = await client.post('/predict', files={'file': ('sample.jpg', self.image, 'image/jpeg')})
                        self.assertEqual(busy.status_code, 503)
                        self.assertEqual(busy.headers['retry-after'], '2')
                        self.assertEqual(busy.json()['error_code'], 'ANALYSIS_BUSY')
                        self.assertEqual((await client.get('/auth/guest')).json()['remaining'], 1)
                    finally:
                        release.set()
                        response = await active
                    self.assertEqual(response.status_code, 200)
                    self.assertFalse(app.ANALYSIS_GATE.locked())
        asyncio.run(scenario())

    def setUp(self):
        app.REQUEST_LIMITER = security.WindowLimiter()
        self.client = TestClient(app.app)
        buffer = io.BytesIO()
        Image.new('RGB', (300, 300), (20, 80, 90)).save(buffer, format='JPEG')
        self.image = buffer.getvalue()

    def upload(self, key=None, data=None, mode='general'):
        headers = {'X-API-Key': key} if key else {}
        return self.client.post('/v1/analyze', headers=headers,
            files={'file': ('sample.jpg', self.image if data is None else data, 'image/jpeg')}, data={'mode': mode})

    def test_pilot_disabled_and_bad_key(self):
        with patch.dict(os.environ, {'ENTERPRISE_API_KEYS': ''}):
            self.assertEqual(self.upload().status_code, 503)
        self.assertEqual(self.upload('wrong').status_code, 401)
        self.assertEqual(self.upload().status_code, 401)

    def test_pilot_success_uses_same_model_and_no_feedback_credentials(self):
        async def infer(*args):
            return 80.0, [{'model': 'test-fixture', 'ai_score': 80.0}]
        with patch.object(app, 'call_ai_model_ensemble', infer):
            response = self.upload(os.environ['ENTERPRISE_API_KEYS'])
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result['recommendation'], 'human_review')
        self.assertEqual(result['fraud_assessment'], 'not_determined')
        self.assertEqual(result['score_type'], 'uncalibrated_model_score')
        self.assertNotIn('analysis_token', result)
        self.assertNotIn('image_hash', result)
        self.assertTrue(response.headers['cache-control'] == 'no-store')

    def test_invalid_upload_is_not_success(self):
        response = self.upload(os.environ['ENTERPRISE_API_KEYS'], b'not an image')
        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(response.json()['status'], 'unavailable')
        self.assertNotIn('Traceback', response.text)

    def test_model_failure_does_not_leak_details(self):
        async def fail(*args):
            raise RuntimeError('private/server/model/path')
        with patch.object(app, 'call_ai_model_ensemble', fail):
            response = self.upload(os.environ['ENTERPRISE_API_KEYS'])
        self.assertEqual(response.status_code, 503)
        self.assertNotIn('private/server', response.text)

    def test_category_rejection_and_engine_failure(self):
        import category_detector
        with patch.object(category_detector, 'classify', return_value={'status':'mismatch','detected':'property'}):
            response = self.upload(os.environ['ENTERPRISE_API_KEYS'], mode='car')
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()['error_code'], 'CATEGORY_MISMATCH')
        with patch.object(category_detector, 'classify', side_effect=RuntimeError('secret path')):
            response = self.upload(os.environ['ENTERPRISE_API_KEYS'], mode='car')
        self.assertEqual(response.status_code, 503)

    def test_scoped_tokens_cannot_access_account(self):
        connection = app.get_db()
        connection.execute('INSERT OR IGNORE INTO users(email,is_verified,created_at) VALUES (?,1,0)', ('fixture@example.com',))
        user_id = connection.execute('SELECT id FROM users WHERE email=?', ('fixture@example.com',)).fetchone()[0]
        connection.commit(); connection.close()
        for token in (app.make_setup_token(user_id, 'fixture@example.com'), app.make_reset_token(user_id, 'fixture@example.com')):
            response = self.client.get('/auth/me', headers={'Authorization':'Bearer '+token})
            self.assertEqual(response.status_code, 401)
        response = self.client.get('/auth/me', headers={'Authorization':'Bearer '+app.make_jwt(user_id,'fixture@example.com')})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.get('/auth/me',headers={'Authorization':'Bearer '+app.make_jwt(999999,'missing@example.com')}).status_code,401)

    def test_legacy_payment_cannot_grant_pro(self):
        self.assertEqual(self.client.post('/webhooks/gumroad',data={'email':'fixture@example.com'}).status_code,410)
        self.assertEqual(self.client.get('/test-ateeqq-raw').status_code,410)

    def test_cors_and_security_headers(self):
        headers={'Origin':'https://evil.example','Access-Control-Request-Method':'POST'}
        self.assertEqual(self.client.options('/predict',headers=headers).status_code,400)
        headers['Origin']='https://ai-image-detector.com'
        self.assertEqual(self.client.options('/predict',headers=headers).status_code,200)
        response=self.client.get('/healthz')
        self.assertEqual(response.headers['x-content-type-options'],'nosniff')
        self.assertIn("frame-ancestors 'none'",response.headers['content-security-policy'])

    def test_rate_limit_and_expiration(self):
        now=[100.0]
        limiter=security.WindowLimiter(max_keys=2,clock=lambda:now[0])
        self.assertTrue(limiter.allow('a','analysis',1))
        self.assertFalse(limiter.allow('a','analysis',1))
        self.assertTrue(limiter.allow('b','analysis',1))
        self.assertFalse(limiter.allow('c','analysis',1))
        now[0]=161.0
        self.assertTrue(limiter.allow('c','analysis',1))
        for _ in range(30):
            self.client.post('/v1/analyze',files={'file':('a.jpg',b'x','image/jpeg')})
        response=self.client.post('/v1/analyze',files={'file':('a.jpg',b'x','image/jpeg')})
        self.assertEqual(response.status_code,429)
        self.assertEqual(response.headers['retry-after'],'60')

    def test_signing_secret_never_known_fallback(self):
        with patch.dict(os.environ,{'JWT_SECRET':'change-me-in-production'}):
            a,stable=security.signing_secret();b,_=security.signing_secret()
            self.assertFalse(stable);self.assertNotEqual(a,b);self.assertGreaterEqual(len(a),32)
        with patch.dict(os.environ,{'JWT_SECRET':'short'}):
            self.assertFalse(security.signing_secret()[1])

    def test_private_dns_and_mixed_answers_rejected(self):
        for address in ('127.0.0.1','10.0.0.1','169.254.169.254','::1','fc00::1','::ffff:127.0.0.1'):
            with patch('socket.getaddrinfo',return_value=[(2,1,6,'',(address,443))]):
                with self.assertRaises(ValueError):security.public_image_url('https://images.example/a')
        with patch('socket.getaddrinfo',return_value=[(2,1,6,'',('93.184.216.34',443)),(2,1,6,'',('10.0.0.1',443))]):
            with self.assertRaises(ValueError):security.public_image_url('https://images.example/a')
        for url in ('file:///a','https://user:pass@images.example/a','https://localhost/a','https://images.example:8080/a'):
            with self.assertRaises(ValueError):security.public_image_url(url)

    def test_download_pins_dns_and_checks_redirects_size(self):
        good=[(2,1,6,'',('93.184.216.34',443))]
        pool=MagicMock();response=pool.request.return_value
        response.status=200;response.headers={'content-type':'image/jpeg'};response.stream.return_value=[b'abc']
        with patch('socket.getaddrinfo',return_value=good),patch('urllib3.HTTPSConnectionPool',return_value=pool) as factory:
            self.assertEqual(security.download_public_image('https://images.example/a')[0],b'abc')
            self.assertEqual(factory.call_args.kwargs['host'],'93.184.216.34')
            self.assertEqual(factory.call_args.kwargs['server_hostname'],'images.example')
            self.assertEqual(factory.call_args.kwargs['cert_reqs'],'CERT_REQUIRED')
            response.stream.return_value=[b'abcd']
            with self.assertRaises(ValueError):security.download_public_image('https://images.example/a',max_bytes=3)
            response.status=302;response.headers={'Location':'http://127.0.0.1/a'}
            with patch('socket.getaddrinfo',side_effect=[good,[(2,1,6,'',('127.0.0.1',80))]]):
                with self.assertRaises(ValueError):security.download_public_image('https://images.example/a')

if __name__=='__main__':
    unittest.main()
