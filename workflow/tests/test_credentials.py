"""Credential routing and privacy tests; no real API calls or keys."""

from contextlib import redirect_stdout
import getpass
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from workflow import credentials
from workflow.backends import OpenAIBackend
from workflow.response_call import main


class CredentialTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="mathai-key-tests-")
        self.addCleanup(directory.cleanup)
        self.key_file = Path(directory.name) / ".secrets" / "openai.key"
        self.enterContext(patch.object(credentials, "KEY_FILE", self.key_file))
        self.enterContext(patch.dict(os.environ, {"OPENAI_API_KEY": ""}))

    def test_local_key_overrides_environment_without_mutating_it(self):
        self.assertIsNone(credentials.openai_api_key())
        with patch.dict(os.environ, {"OPENAI_API_KEY": "synthetic-shell-key"}):
            self.assertEqual(credentials.openai_api_key(), "synthetic-shell-key")
            credentials.save_openai_key("synthetic-project-key")
            self.assertEqual(credentials.openai_api_key(), "synthetic-project-key")
            self.assertEqual(os.environ["OPENAI_API_KEY"], "synthetic-shell-key")
            self.key_file.unlink()
            self.assertEqual(credentials.openai_api_key(), "synthetic-shell-key")

    def test_save_and_rotation_use_private_permissions(self):
        credentials.save_openai_key("first-synthetic-key")
        credentials.save_openai_key("replacement-synthetic-key")
        self.assertEqual(credentials.openai_api_key(), "replacement-synthetic-key")
        self.assertEqual(list(self.key_file.parent.iterdir()), [self.key_file])
        if os.name == "posix":
            self.assertEqual(self.key_file.stat().st_mode & 0o777, 0o600)
            self.assertEqual(self.key_file.parent.stat().st_mode & 0o777, 0o700)

    def test_invalid_input_and_symbolic_links_are_rejected(self):
        for value in ("", " ", "embedded\nnewline", "nul\0byte", "two words", "x" * 4097):
            with self.subTest(value_length=len(value)), self.assertRaises(ValueError):
                credentials.save_openai_key(value)
        self.assertFalse(self.key_file.exists())
        self.key_file.parent.mkdir()
        target = self.key_file.parent / "target"
        target.write_text("untouched")
        self.key_file.symlink_to(target)
        with self.assertRaises(ValueError):
            credentials.save_openai_key("synthetic-key")
        self.assertEqual(target.read_text(), "untouched")
        self.assertFalse(credentials.openai_key_configured())

    def test_setup_requires_hidden_terminal_input(self):
        with patch("workflow.credentials.sys.stdin.isatty", return_value=False), \
                patch("workflow.credentials.getpass.getpass") as prompt:
            with self.assertRaises(ValueError):
                credentials.configure_openai("gpt-4o-mini")
            prompt.assert_not_called()
        with patch("workflow.credentials.sys.stdin.isatty", return_value=True), \
                patch("workflow.credentials.getpass.getpass", side_effect=getpass.GetPassWarning), \
                redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):
                credentials.configure_openai("gpt-4o-mini")
        self.assertFalse(self.key_file.exists())

    def test_setup_saves_only_a_successfully_checked_key_without_printing_it(self):
        credentials.save_openai_key("previous-synthetic-key")
        output = io.StringIO()
        with patch("workflow.credentials.sys.stdin.isatty", return_value=True), \
                patch("workflow.credentials.getpass.getpass", return_value="new-synthetic-key"), \
                patch("workflow.credentials.check_openai_key", return_value=(False, "Rejected.")), \
                redirect_stdout(output):
            self.assertEqual(main(["configure-openai"]), 1)
        self.assertEqual(credentials.openai_api_key(), "previous-synthetic-key")
        with patch("workflow.credentials.sys.stdin.isatty", return_value=True), \
                patch("workflow.credentials.getpass.getpass", return_value="new-synthetic-key"), \
                patch("workflow.credentials.check_openai_key", return_value=(True, "Accepted.")), \
                redirect_stdout(output):
            self.assertEqual(main(["configure-openai"]), 0)
        self.assertEqual(credentials.openai_api_key(), "new-synthetic-key")
        self.assertNotIn("new-synthetic-key", output.getvalue())

    def test_check_never_prints_raw_provider_errors_or_generates_tokens(self):
        class Rejected(Exception):
            status_code = 401

        for error in (Rejected("secret-in-provider-response"), RuntimeError("secret-in-provider-response")):
            with patch("openai.OpenAI") as factory:
                client = factory.return_value.__enter__.return_value
                client.models.retrieve.side_effect = error
                success, message = credentials.check_openai_key("synthetic-key", "gpt-4o-mini")
                self.assertFalse(success)
                self.assertNotIn("secret-in-provider-response", message)
                client.chat.completions.create.assert_not_called()
                client.responses.create.assert_not_called()
        with patch("openai.OpenAI") as factory:
            self.assertTrue(credentials.check_openai_key("synthetic-key", "gpt-4o-mini")[0])
            factory.return_value.__enter__.return_value.models.retrieve.assert_called_once_with("gpt-4o-mini")

    def test_model_backend_receives_the_project_key(self):
        with patch("langchain.chat_models.init_chat_model") as factory:
            with self.assertRaises(ValueError):
                OpenAIBackend("gpt-4o-mini", 2400, 2026)
            factory.assert_not_called()
            credentials.save_openai_key("project-only-synthetic-key")
            OpenAIBackend("gpt-4o-mini", 2400, 2026)
            self.assertEqual(factory.call_args.kwargs["api_key"], "project-only-synthetic-key")


if __name__ == "__main__":
    unittest.main()
