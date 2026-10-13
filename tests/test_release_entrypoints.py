"""Release-only HTTP/UI compatibility checks with no model or scientific result generation."""
from pathlib import Path
import runpy
import unittest
from unittest.mock import AsyncMock, Mock, patch

ROOT = Path(__file__).resolve().parents[1]


class ReleaseEntryPointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = runpy.run_path(str(ROOT / "app.py"), run_name="release_http_test")
        cls.namespace = cls.service["create_app"].__globals__

    def test_chat_accepts_request_without_browser_credentials(self):
        from fastapi.testclient import TestClient
        handler = AsyncMock(return_value={"answer": "transport check", "reasoning": {}})
        with patch.dict(self.namespace, {"run_SciToolAgent": handler}):
            with TestClient(self.service["app"]) as client:
                response = client.post("/chat", json={"question": "transport check", "file_path_list": []})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["response"], "transport check")
        handler.assert_awaited_once_with("transport check", [], is_add_filename=False)

    def test_legacy_credentials_do_not_override_server_configuration(self):
        from fastapi.testclient import TestClient
        handler = AsyncMock(return_value={"answer": "transport check", "reasoning": {}})
        configured_key = self.namespace["_cfg"].API_KEY
        with patch.dict(self.namespace, {"run_SciToolAgent": handler}):
            with TestClient(self.service["app"]) as client:
                response = client.post("/chat", json={"question": "transport check", "file_path_list": [],
                                                     "api_key": "dummy-legacy-key", "api_base": "http://localhost"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.namespace["_cfg"].API_KEY, configured_key)
        self.assertNotIn("dummy-legacy-key", response.text)

    def test_browser_uses_backend_configuration_controls(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT / "web_demo.py")).run(timeout=30)
        self.assertFalse(app.exception)
        labels = [widget.label for widget in app.text_input]
        self.assertIn("后端地址", labels)
        self.assertNotIn("API Key", labels)
        self.assertNotIn("API Base URL", labels)

    def test_browser_displays_backend_error_instead_of_missing_answer(self):
        from streamlit.testing.v1 import AppTest
        response = Mock(status_code=200)
        response.json.return_value = {"error_type": "RuntimeError", "message": "offline transport test error"}
        app = AppTest.from_file(str(ROOT / "web_demo.py")).run(timeout=30)
        with patch("requests.post", return_value=response):
            app.chat_input[0].set_value("transport check").run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(any("offline transport test error" in error.value for error in app.error))


if __name__ == "__main__":
    unittest.main()
