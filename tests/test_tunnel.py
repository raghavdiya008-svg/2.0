#!/usr/bin/env python3
"""
tests/test_tunnel.py
--------------------
Unit tests for Cloudflare quick tunnel launcher and permission handling.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Ensure src in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import tunnel


class TestTunnel(unittest.TestCase):
    def test_01_is_cloud_environment_detection(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(tunnel.is_cloud_environment())

        with patch.dict(os.environ, {"COLAB_GPU": "1"}):
            self.assertTrue(tunnel.is_cloud_environment())

        with patch.dict(os.environ, {"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}):
            self.assertTrue(tunnel.is_cloud_environment())

    def test_02_ensure_executable(self):
        # Nonexistent path returns False
        self.assertFalse(tunnel.ensure_executable("/nonexistent/file/path"))

        # Existing file
        curr_file = os.path.abspath(__file__)
        self.assertTrue(tunnel.ensure_executable(curr_file))

    @patch("tunnel.shutil.which")
    def test_03_get_cloudflared_binary_system_path(self, mock_which):
        mock_which.return_value = "/usr/bin/cloudflared"
        with patch("tunnel.ensure_executable", return_value=True):
            binary = tunnel.get_cloudflared_binary()
            self.assertEqual(binary, "/usr/bin/cloudflared")

    @patch("subprocess.Popen")
    @patch("tunnel.get_cloudflared_binary")
    def test_04_start_cloudflare_tunnel_success(self, mock_get_bin, mock_popen):
        mock_get_bin.return_value = "/usr/bin/cloudflared"
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stdout.readline.side_effect = [
            "INF Tunnel credentials fetched\n",
            "INF +--------------------------------------------------------------------------------------------+\n",
            "INF |  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):  |\n",
            "INF |  https://fancy-alpha-beta.trycloudflare.com                                                |\n",
            "INF +--------------------------------------------------------------------------------------------+\n",
            "",
        ]
        mock_popen.return_value = mock_proc

        url, proc = tunnel.start_cloudflare_tunnel(port=5000, timeout=5)
        self.assertEqual(url, "https://fancy-alpha-beta.trycloudflare.com")
        self.assertEqual(proc, mock_proc)


if __name__ == "__main__":
    unittest.main()
