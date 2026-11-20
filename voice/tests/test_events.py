import unittest

from voice.events import VoiceEventBus


class TestEvents(unittest.TestCase):
    def test_bus(self) -> None:
        bus = VoiceEventBus()
        seen = []

        bus.subscribe_confirmed(lambda t: seen.append(t))
        bus.emit_confirmed("hello")
        self.assertEqual(seen, ["hello"])


if __name__ == "__main__":
    unittest.main()