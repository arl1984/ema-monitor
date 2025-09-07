import os
import sys
import unittest
from unittest.mock import patch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import monitor

class TestSendWebhookMarketHours(unittest.TestCase):
    @patch('monitor.requests.post')
    def test_skip_outside_market_hours(self, mock_post):
        cfg = {'webhook_url': 'http://example.com', 'webhook_type': 'discord'}
        with patch('monitor.is_market_hours', return_value=False):
            monitor.send_webhook(cfg, payload_text='hello')
        mock_post.assert_not_called()

    @patch('monitor.requests.post')
    def test_send_during_market_hours(self, mock_post):
        cfg = {'webhook_url': 'http://example.com', 'webhook_type': 'discord'}
        with patch('monitor.is_market_hours', return_value=True):
            monitor.send_webhook(cfg, payload_text='hello')
        mock_post.assert_called_once()

if __name__ == '__main__':
    unittest.main()
