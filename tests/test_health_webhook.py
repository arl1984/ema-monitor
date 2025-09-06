import unittest
from unittest.mock import patch
import monitor

class TestHealthWebhook(unittest.TestCase):
    def test_extra_payload_passed(self):
        cfg = {'webhook_url': 'http://example.com', 'webhook_type': 'discord'}
        extra = {'foo': 'bar'}
        with patch('monitor.send_webhook') as mock_send:
            monitor.send_health_webhook(cfg, 'hello', extra)
            mock_send.assert_called_once()
            args, kwargs = mock_send.call_args
            # first positional arg is cfg
            self.assertEqual(args[0], cfg)
            # ensure payload_obj used and includes extra data
            self.assertIn('payload_obj', kwargs)
            self.assertEqual(kwargs['payload_obj'], {'content': 'hello', 'foo': 'bar'})
            # payload_text should not be used
            self.assertNotIn('payload_text', kwargs)

if __name__ == '__main__':
    unittest.main()
