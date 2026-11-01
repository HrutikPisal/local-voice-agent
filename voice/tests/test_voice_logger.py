import json
import tempfile
import unittest
from pathlib import Path

from voice.logger import VoiceLogger
from voice.telemetry import TurnTelemetry


class VoiceLoggerTests(unittest.TestCase):
    def test_writes_jsonl_with_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "voice_test.txt"
            vlog = VoiceLogger(source="voice_test", log_path=log_path)
            turn_id = "turn-abc"

            vlog.log_turn_start(turn_id, mode="cli", stt_model="base")
            telemetry = TurnTelemetry()
            telemetry.mark_record_start()
            telemetry.mark_record_end()
            telemetry.mark_stt_end()
            metrics = telemetry.to_metrics_dict()

            vlog.log_stt_result(
                turn_id,
                outcome="ok",
                transcript_raw="hello",
                stt_ms=metrics["stt_ms"],
                stt_model="base",
            )
            vlog.log_turn_summary(turn_id, outcome="confirmed", metrics=metrics)

            lines = log_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 3)

            for line in lines:
                entry = json.loads(line)
                self.assertEqual(entry["source"], "voice_test")
                self.assertEqual(entry["turn_id"], turn_id)
                self.assertIn("timestamp", entry)
                self.assertIn("event", entry)

            summary = json.loads(lines[-1])
            self.assertEqual(summary["event"], "voice_turn_summary")
            self.assertIn("record_ms", summary)
            self.assertIn("stt_ms", summary)


if __name__ == "__main__":
    unittest.main()
