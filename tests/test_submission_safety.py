import importlib.machinery
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "codex-auto-resume"
loader = importlib.machinery.SourceFileLoader("codex_auto_resume", str(SCRIPT))
spec = importlib.util.spec_from_loader(loader.name, loader)
app = importlib.util.module_from_spec(spec)
loader.exec_module(app)


class SubmissionSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        directory = Path(self.temp.name)
        for name, value in {
            "APP_DIR": directory,
            "STATE_PATH": directory / "state.json",
            "LOCK_PATH": directory / "state.lock",
        }.items():
            patcher = mock.patch.object(app, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.cfg = dict(app.DEFAULT_CONFIG)
        self.snap = {"latest": {"id": "failed-turn"}}
        self.state = {
            **app.default_state(), "armed": True, "thread_id": "test-thread",
            "mode": "message", "resume_count": 0, "max_resumes": 5,
        }
        app.save_state(self.state)

    def test_unreadable_or_nonempty_queue_never_sends(self):
        for pending in (None, ["another-queued-message"]):
            with self.subTest(pending=pending):
                with mock.patch.object(app, "pending_queue_items", return_value=pending), \
                     mock.patch.object(app, "ensure_loaded") as load, \
                     mock.patch.object(app, "reactivate_goal") as activate, \
                     mock.patch.object(app, "codex_queue") as send:
                    with app.mutate_state() as state:
                        state["mode"] = "goal"
                        self.assertFalse(app.do_resume(self.cfg, state, self.snap, "after_limit"))
                    load.assert_not_called()
                    activate.assert_not_called()
                    send.assert_not_called()
                    self.assertNotIn("unresolved_submission", app.load_state())

    def test_missing_queue_database_is_not_created_or_treated_as_empty(self):
        directory = Path(self.temp.name) / "missing-codex-home"
        with mock.patch.object(app, "CODEX_HOME", directory), \
             mock.patch.object(app.time, "sleep"), \
             mock.patch.object(app, "ensure_loaded") as load, \
             mock.patch.object(app, "codex_queue") as send:
            with app.mutate_state() as state:
                self.assertFalse(app.do_resume(self.cfg, state, self.snap, "after_limit"))
            load.assert_not_called()
            send.assert_not_called()
            self.assertFalse(directory.exists())

    def test_queue_change_while_loading_prevents_submission(self):
        for second_read in (None, ["another-queued-message"]):
            with self.subTest(second_read=second_read):
                app.save_state(self.state)
                with mock.patch.object(app, "pending_queue_items", side_effect=[[], second_read]), \
                     mock.patch.object(app, "ensure_loaded", return_value="loaded"), \
                     mock.patch.object(app, "reactivate_goal") as activate, \
                     mock.patch.object(app, "codex_queue") as send:
                    with app.mutate_state() as state:
                        state["mode"] = "goal"
                        self.assertFalse(app.do_resume(self.cfg, state, self.snap, "after_limit"))
                    activate.assert_not_called()
                    send.assert_not_called()
                    self.assertNotIn("unresolved_submission", app.load_state())

    def test_intent_is_on_disk_before_send_and_receipt_is_saved(self):
        def send(cfg, thread, prompt):
            intent = app.load_state()["unresolved_submission"]
            self.assertEqual(intent["thread_id"], "test-thread")
            self.assertEqual(intent["turn_before"], "failed-turn")
            self.assertIn("submission=" + intent["submission_id"], prompt)
            return "queue-receipt"

        with mock.patch.object(app, "pending_queue_items", return_value=[]), \
             mock.patch.object(app, "ensure_loaded", return_value="loaded"), \
             mock.patch.object(app, "codex_queue", side_effect=send):
            with app.mutate_state() as state:
                self.assertTrue(app.do_resume(self.cfg, state, self.snap, "after_limit"))
        saved = app.load_state()
        self.assertEqual(saved["phase"], "resuming")
        self.assertEqual(saved["resume_queue_id"], "queue-receipt")
        self.assertEqual(len(saved["resume_submission_id"]), 32)
        self.assertNotIn("unresolved_submission", saved)

    def test_lost_receipt_blocks_automatic_and_manual_retries(self):
        with mock.patch.object(app, "pending_queue_items", return_value=[]), \
             mock.patch.object(app, "ensure_loaded", return_value="loaded"), \
             mock.patch.object(app, "codex_queue", side_effect=TimeoutError("unknown outcome")):
            with self.assertRaises(TimeoutError):
                with app.mutate_state() as state:
                    app.do_resume(self.cfg, state, self.snap, "after_limit")

        saved = app.load_state()
        self.assertEqual(saved["phase"], "submission_unknown")
        self.assertEqual(saved["unresolved_submission"]["thread_id"], "test-thread")
        caf = mock.Mock()
        with mock.patch.object(app, "AppServer") as server:
            self.assertEqual(app.tick(self.cfg, caf), 3600)
            server.assert_not_called()
        with self.assertRaises(SystemExit):
            app.cmd_resume_now(self.cfg, SimpleNamespace(force=True, yes=True))
        with mock.patch.object(app, "signal_daemon"):
            app.cmd_disarm(self.cfg, None)
        self.assertIn("unresolved_submission", app.load_state())
        with self.assertRaises(SystemExit):
            app.cmd_arm(self.cfg, SimpleNamespace(thread="test-thread"))
        with self.assertRaises(SystemExit):
            app.cmd_reset(self.cfg, None)

    def test_failed_durable_write_prevents_all_side_effects(self):
        with mock.patch.object(app, "pending_queue_items", return_value=[]), \
             mock.patch.object(app.os, "fsync", side_effect=OSError("disk failure")), \
             mock.patch.object(app, "ensure_loaded") as load, \
             mock.patch.object(app, "codex_queue") as send:
            with self.assertRaises(OSError):
                app.do_resume(self.cfg, self.state, self.snap, "after_limit")
            load.assert_not_called()
            send.assert_not_called()

    def test_lost_receipt_state_write_does_not_allow_a_retry(self):
        save_state = app.save_state
        calls = 0

        def fail_after_send(state):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("receipt write failed")
            save_state(state)

        with mock.patch.object(app, "pending_queue_items", return_value=[]), \
             mock.patch.object(app, "ensure_loaded", return_value="loaded"), \
             mock.patch.object(app, "codex_queue", return_value="queue-receipt") as send, \
             mock.patch.object(app, "save_state", side_effect=fail_after_send):
            with self.assertRaises(OSError):
                with app.mutate_state() as state:
                    app.do_resume(self.cfg, state, self.snap, "after_limit")
            send.assert_called_once()
        self.assertTrue(app.has_unresolved_submission(app.load_state()))
        with mock.patch.object(app, "AppServer") as server:
            self.assertEqual(app.tick(self.cfg, mock.Mock()), 3600)
            server.assert_not_called()

    def test_process_exit_after_intent_does_not_lose_the_fence(self):
        program = """
import importlib.machinery, importlib.util, os, pathlib, sys
loader = importlib.machinery.SourceFileLoader('resume_child', sys.argv[1])
spec = importlib.util.spec_from_loader(loader.name, loader)
app = importlib.util.module_from_spec(spec)
loader.exec_module(app)
app.APP_DIR = pathlib.Path(sys.argv[2])
app.STATE_PATH = app.APP_DIR / 'state.json'
app.LOCK_PATH = app.APP_DIR / 'state.lock'
app.pending_queue_items = lambda thread: []
app.ensure_loaded = lambda cfg, thread: 'loaded'
app.codex_queue = lambda cfg, thread, prompt: os._exit(73)
with app.mutate_state() as state:
    app.do_resume(app.DEFAULT_CONFIG, state, {'latest': {'id': 'failed-turn'}}, 'after_limit')
"""
        result = subprocess.run([sys.executable, "-B", "-c", program, str(SCRIPT), self.temp.name])
        self.assertEqual(result.returncode, 73)
        self.assertEqual(app.load_state()["phase"], "submission_unknown")
        with mock.patch.object(app, "AppServer") as server:
            self.assertEqual(app.tick(self.cfg, mock.Mock()), 3600)
            server.assert_not_called()

    def test_unreadable_state_is_not_treated_as_a_fresh_install(self):
        for broken in ("{", "[]", "null"):
            with self.subTest(broken=broken):
                app.STATE_PATH.write_text(broken)
                with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                    app.load_state()
                self.assertEqual(app.STATE_PATH.read_text(), broken)

    def test_malformed_intent_and_unknown_phase_still_block_sends(self):
        for update in ({"unresolved_submission": None}, {"unresolved_submission": {}},
                       {"phase": "submission_unknown"}):
            with self.subTest(update=update):
                app.save_state({**self.state, **update})
                with mock.patch.object(app, "AppServer") as server:
                    self.assertEqual(app.tick(self.cfg, mock.Mock()), 3600)
                    server.assert_not_called()
                with self.assertRaises(SystemExit):
                    app.cmd_resume_now(self.cfg, SimpleNamespace(force=True, yes=True))

    def test_only_desktop_app_server_is_checked_for_open_lock_file(self):
        root = Path(self.temp.name).resolve() / "ChatGPT.app"
        main = root / "Contents/MacOS/ChatGPT"
        cli = root / "Contents/Resources/CodexCLI.app/Contents/MacOS/codex"
        for executable in (main, cli):
            executable.parent.mkdir(parents=True, exist_ok=True)
            executable.touch()
        helper = root / "Contents/Frameworks/desktop-helper"
        lock = root / "thread-writer-locks/test-thread.lock"
        lock.parent.mkdir()
        lock.touch()
        cfg = dict(self.cfg, codex_cli=str(cli))
        table = "\n".join([
            f"100 1 {main}", f"101 100 {helper}", f"102 101 {cli}",
            f"103 102 {cli}", f"104 1 {cli}", f"105 100 {cli}", f"106 105 {cli}",
        ])
        args = f"102 {cli} app-server\n105 {cli} exec-server\n106 {cli} app-server\n"
        calls = []

        def run(command, **kwargs):
            calls.append((command, kwargs))
            if command[0] == "/usr/sbin/lsof":
                return SimpleNamespace(returncode=0, stdout="102\n104\n", stderr="")
            if "-A" in command:
                return SimpleNamespace(returncode=0, stdout=table, stderr="")
            return SimpleNamespace(returncode=0, stdout=args, stderr="")

        with mock.patch.object(app, "CODEX_HOME", root), \
             mock.patch.object(app.subprocess, "run", side_effect=run):
            self.assertEqual(app.lock_holders(cfg, "test-thread"), [102])
        lsof = [call for call in calls if call[0][0] == "/usr/sbin/lsof"]
        self.assertEqual(len(lsof), 1)
        self.assertEqual(lsof[0][0][lsof[0][0].index("-p") + 1], "102")
        self.assertEqual(lsof[0][1]["timeout"], 5)

    def test_ownership_timeout_fails_closed(self):
        lock = Path(self.temp.name) / "thread-writer-locks/test-thread.lock"
        lock.parent.mkdir()
        lock.touch()
        with mock.patch.object(app, "CODEX_HOME", Path(self.temp.name)), \
             mock.patch.object(app, "desktop_server_pids", return_value=[102]), \
             mock.patch.object(app.subprocess, "run", side_effect=subprocess.TimeoutExpired("lsof", 5)):
            with self.assertRaisesRegex(app.AppServerError, "ownership"):
                app.lock_holders(self.cfg, "test-thread")

    def test_disappearing_server_is_not_counted(self):
        lock = Path(self.temp.name) / "thread-writer-locks/test-thread.lock"
        lock.parent.mkdir()
        lock.touch()
        result = SimpleNamespace(returncode=0, stdout="102\n", stderr="")
        with mock.patch.object(app, "CODEX_HOME", Path(self.temp.name)), \
             mock.patch.object(app, "desktop_server_pids", side_effect=[[102], [103]]), \
             mock.patch.object(app.subprocess, "run", return_value=result):
            self.assertEqual(app.lock_holders(self.cfg, "test-thread"), [])

    def test_open_without_desktop_lock_file_does_not_proceed(self):
        with mock.patch.object(app, "lock_holders", return_value=[]), \
             mock.patch.object(app, "codex_app_running", return_value=True), \
             mock.patch.object(app, "wait_for_owner", return_value=False), \
             mock.patch.object(app.subprocess, "run"):
            with self.assertRaisesRegex(app.AppServerError, "did not open"):
                app.ensure_loaded(self.cfg, "test-thread")

    def test_running_check_uses_desktop_app_server_detection(self):
        with mock.patch.object(app, "desktop_server_pids", side_effect=[[102], []]):
            self.assertTrue(app.codex_app_running(self.cfg))
            self.assertFalse(app.codex_app_running(self.cfg))

    def test_available_quota_ignores_stale_error_reset_time(self):
        state = {}
        windows = [{"used": 10, "resets_at": 1_010_000, "mins": minutes}
                   for minutes in (300, 10080)]
        with mock.patch.object(app, "now", return_value=1_000_000), \
             mock.patch.object(app, "parse_try_again", return_value=1_086_400):
            app.schedule_from_limit(self.cfg, state, {"error_message": "old limit"},
                                    {"allowed": True, "reset_at": None, "windows": windows})
        self.assertEqual(state["reset_at"], 1_000_090)
        self.assertEqual(state["reset_source"], "usage already available")

    def test_unknown_quota_never_submits_automatically(self):
        state = {**self.state, "phase": "waiting_reset", "limit_turn_id": "failed-turn",
                 "reset_at": 999_999, "next_check_at": 0}
        app.save_state(state)
        server = mock.MagicMock()
        server.__enter__.return_value.call.return_value = {"ordinaryUsageAllowed": None}
        snap = {"latest": {"id": "failed-turn"}, "goal": None}
        with mock.patch.object(app, "now", return_value=1_000_000), \
             mock.patch.object(app, "AppServer", return_value=server), \
             mock.patch.object(app, "read_snapshot", return_value=snap), \
             mock.patch.object(app, "classify", return_value=("limited", "quota limit")), \
             mock.patch.object(app, "do_resume") as send:
            app.tick(self.cfg, mock.Mock())
            send.assert_not_called()
        self.assertEqual(app.load_state()["phase"], "waiting_reset")

    def test_exhausted_window_blocks_even_if_allowed_flag_is_true(self):
        self.assertFalse(app.quota_available({"allowed": True, "reset_at": 1_000_100}))

    def test_missing_or_stale_quota_windows_never_authorize_a_send(self):
        base = {"allowed": True, "reset_at": None}
        with mock.patch.object(app, "now", return_value=1_000_000):
            self.assertFalse(app.quota_available(base))
            self.assertFalse(app.quota_available({**base, "windows": [
                {"used": 10, "resets_at": 1_010_000, "mins": 300}]}))
            self.assertFalse(app.quota_available({**base, "windows": [
                {"used": 100, "resets_at": 999_999, "mins": 300},
                {"used": 10, "resets_at": 1_010_000, "mins": 10080}]}))

    def test_submission_marker_is_extracted_from_the_actual_user_message(self):
        token = "a" * 32
        turn = {"id": "new-turn", "items": [{"type": "userMessage", "content": [
            {"type": "text", "text": "continue\n\n[codex-auto-resume] submission=" + token}]}]}
        self.assertEqual(app.summarize_turn(turn)["submission_id"], token)
        turn["items"][0]["content"][0]["text"] = "an unrelated message"
        self.assertIsNone(app.summarize_turn(turn)["submission_id"])

    def test_only_matching_submission_can_confirm_resume(self):
        token = "a" * 32
        for actual, expected_phase in ((token, "working"), ("b" * 32, "stopped"), (None, "stopped")):
            with self.subTest(actual=actual):
                state = {**self.state, "phase": "resuming", "resume_turn_before": "failed-turn",
                         "resume_submission_id": token, "resume_kind": "after_limit",
                         "resume_deadline": 1_000_100, "next_check_at": 0}
                app.save_state(state)
                server = mock.MagicMock()
                snap = {"latest": {"id": "new-turn", "submission_id": actual,
                                   "status": "inProgress"}, "goal": None}
                cfg = dict(self.cfg, notifications=False)
                with mock.patch.object(app, "now", return_value=1_000_000), \
                     mock.patch.object(app, "AppServer", return_value=server), \
                     mock.patch.object(app, "read_snapshot", return_value=snap), \
                     mock.patch.object(app, "classify", return_value=("working", "running")):
                    app.tick(cfg, mock.Mock())
                saved = app.load_state()
                self.assertEqual(saved["phase"], expected_phase)
                self.assertEqual(saved.get("resume_started_turn_id"),
                                 "new-turn" if actual == token else None)
                self.assertEqual(app.has_unresolved_submission(saved), actual != token)

    def test_failed_matching_turn_is_not_announced_as_resumed(self):
        token = "a" * 32
        state = {**self.state, "phase": "resuming", "resume_turn_before": "failed-turn",
                 "resume_submission_id": token, "resume_kind": "after_limit", "next_check_at": 0}
        app.save_state(state)
        server = mock.MagicMock()
        snap = {"latest": {"id": "new-turn", "submission_id": token, "status": "failed"}, "goal": None}
        with mock.patch.object(app, "AppServer", return_value=server), \
             mock.patch.object(app, "read_snapshot", return_value=snap), \
             mock.patch.object(app, "classify", return_value=("stop", "turn failed")), \
             mock.patch.object(app, "notify") as notify:
            app.tick(self.cfg, mock.Mock())
        saved = app.load_state()
        self.assertNotIn("resume_started_verified_at", saved)
        self.assertNotIn("resume_confirmed", [event["event"] for event in saved["history"]])
        self.assertNotIn("Codex task resumed", [call.args[1] for call in notify.call_args_list])

    def test_queued_submission_is_preserved_when_disarmed_or_cancelled(self):
        for command in (app.cmd_disarm, app.cmd_cancel):
            with self.subTest(command=command.__name__):
                state = {**self.state, "phase": "resuming", "resume_queue_id": "queue-receipt",
                         "resume_submission_id": "a" * 32, "resume_turn_before": "failed-turn"}
                app.save_state(state)
                with mock.patch.object(app, "signal_daemon"):
                    command(self.cfg, None)
                saved = app.load_state()
                self.assertTrue(app.has_unresolved_submission(saved))
                self.assertEqual(saved["unresolved_submission"]["queue_id"], "queue-receipt")
                with self.assertRaises(SystemExit):
                    app.cmd_reset(self.cfg, None)

    def test_manually_sending_or_rearming_while_queued_is_blocked(self):
        app.save_state({**self.state, "phase": "resuming", "resume_queue_id": "queue-receipt"})
        with self.assertRaises(SystemExit):
            app.cmd_resume_now(self.cfg, SimpleNamespace(force=True, yes=True))
        with self.assertRaises(SystemExit):
            app.cmd_arm(self.cfg, SimpleNamespace(thread="test-thread"))

    def test_unconfirmed_start_keeps_the_submission_for_review(self):
        state = {**self.state, "phase": "resuming", "resume_turn_before": "failed-turn",
                 "resume_queue_id": "queue-receipt", "resume_attempts": 2,
                 "resume_deadline": 999_999, "next_check_at": 0}
        app.save_state(state)
        server = mock.MagicMock()
        snap = {"latest": {"id": "failed-turn"}, "goal": None}
        with mock.patch.object(app, "now", return_value=1_000_000), \
             mock.patch.object(app, "AppServer", return_value=server), \
             mock.patch.object(app, "read_snapshot", return_value=snap), \
             mock.patch.object(app, "classify", return_value=("working", "running")), \
             mock.patch.object(app, "pending_queue_items", return_value=[]):
            app.tick(dict(self.cfg, notifications=False), mock.Mock())
        saved = app.load_state()
        self.assertEqual(saved["phase"], "stopped")
        self.assertEqual(saved["unresolved_submission"]["reason"], "start_unconfirmed")
        self.assertEqual(saved["unresolved_submission"]["queue_id"], "queue-receipt")


if __name__ == "__main__":
    unittest.main()
