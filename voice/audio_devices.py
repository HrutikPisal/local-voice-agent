"""Microphone device enumeration."""

from __future__ import annotations


def list_input_devices() -> list[dict]:
    try:
        import sounddevice as sd
    except ImportError:
        return []
    devices = sd.query_devices()
    out = []
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) < 1:
            continue
        out.append(
            {
                "index": idx,
                "name": dev.get("name", ""),
                "channels": dev.get("max_input_channels"),
                "default_samplerate": dev.get("default_samplerate"),
            }
        )
    return out


def format_device_list() -> str:
    lines = ["Input devices:"]
    default_in = None
    try:
        import sounddevice as sd

        default_in = sd.default.device[0]
    except Exception:
        pass
    for dev in list_input_devices():
        mark = " (default)" if default_in is not None and dev["index"] == default_in else ""
        lines.append(f"  [{dev['index']}] {dev['name']}{mark}")
    return "\n".join(lines)