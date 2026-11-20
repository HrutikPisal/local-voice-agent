"""Voice pipeline events for CLI agent / TTS / LLM integration later."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

PartialHandler = Callable[[str], None]
FinalHandler = Callable[[str], None]
ConfirmedHandler = Callable[[str], None]


@dataclass
class VoiceEventBus:
  """Simple pub/sub for voice turn lifecycle."""

  on_partial: list[PartialHandler] = field(default_factory=list)
  on_final: list[FinalHandler] = field(default_factory=list)
  on_confirmed: list[ConfirmedHandler] = field(default_factory=list)

  def emit_partial(self, text: str) -> None:
    for handler in self.on_partial:
      handler(text)

  def emit_final(self, text: str) -> None:
    for handler in self.on_final:
      handler(text)

  def emit_confirmed(self, text: str) -> None:
    for handler in self.on_confirmed:
      handler(text)

  def subscribe_partial(self, handler: PartialHandler) -> None:
    self.on_partial.append(handler)

  def subscribe_final(self, handler: FinalHandler) -> None:
    self.on_final.append(handler)

  def subscribe_confirmed(self, handler: ConfirmedHandler) -> None:
    self.on_confirmed.append(handler)


_default_bus = VoiceEventBus()


def default_event_bus() -> VoiceEventBus:
  return _default_bus