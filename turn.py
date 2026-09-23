"""Endpointing and turn-taking for the voice bridge.

Why this exists: a sentence spoken with pauses in it produced one reply per pause. The
old loop treated any 700ms of silence as end-of-turn, and each of those silences started
its own STT/LLM/TTS pass, so "hey, so I was thinking... maybe we could... go tomorrow"
came back as three separate replies, each answering a fragment.

Three rules, in order of how much they matter:

1. Silence is not the end of a turn. A mid-thought pause and a finished thought are
   identical to an energy VAD, so a candidate endpoint is judged semantically before
   anything gets generated. Obviously-unfinished text keeps listening with no model call,
   obviously-finished text answers straight away, and only the ambiguous middle pays for
   one short classification call.
2. One turn, one reply. The whole utterance goes through a single pass. There is no
   drain loop, so leftover audio can never become a second reply.
3. New speech supersedes. Speech arriving while a reply is being generated or played
   cancels it through an epoch counter, so audio for an utterance the caller has already
   moved past can never reach the wire.

The pause we wait for is chosen by content, not by a flat timer: a clause dangling on a
conjunction earns a longer wait (plainly mid-thought), while an ambiguous clause is checked
again promptly so the final answer is not pushed out by dead air. If the caller holds an
utterance open and then goes quiet for good, a forced reply uses the transcript we already
have, without re-transcribing anything, so they are never left in silence.

This module is synchronous and free of I/O on purpose: frames go in, actions come out,
which is what makes the behaviour testable without a phone call (see test_turn.py).
"""
import os
from typing import Any, Dict, List, Optional

# --- tunables (env-overridable so a call can be tuned without a rebuild) -------------
VAD_LEVEL = int(os.getenv("VAD_LEVEL", "500"))
MIN_SILENCE_MS = int(os.getenv("ENDPOINT_MIN_MS", "1100"))
EXTENDED_SILENCE_MS = int(os.getenv("ENDPOINT_EXTENDED_MS", "1700"))
HARD_MAX_SILENCE_MS = int(os.getenv("ENDPOINT_HARD_MAX_MS", "3000"))
MAX_UTTERANCE_MS = int(os.getenv("MAX_UTTERANCE_MS", "30000"))

GATE_SYSTEM = (
    "You judge whether a speaker has finished their turn on a live phone call. You are "
    "given the transcript of what they just said. Reply with exactly one word. DONE if "
    "they completed a thought and are waiting for an answer. WAIT if they are "
    "mid-sentence, or trailing off in a way that suggests more is coming: a dangling "
    "conjunction, an unfinished clause, a filler word at the end. Output one word only."
)

# Words that leave a clause hanging: if the transcript ends on one of these the speaker is
# almost certainly still going, so we can wait without spending a model call.
_HANGING = {
    "and", "but", "so", "or", "nor", "because", "if", "unless", "until", "while", "when",
    "that", "which", "who", "whose", "to", "of", "in", "on", "at", "with", "for", "from",
    "by", "as", "than", "then", "plus", "about", "into", "over", "under", "the", "a", "an",
    "my", "your", "his", "her", "its", "our", "their", "is", "was", "were", "are", "am",
    "be", "been", "being", "do", "does", "did", "have", "has", "had", "can", "could",
    "would", "should", "will", "shall", "may", "might", "must", "not", "just", "also",
    "really", "very", "kind", "sort", "going", "gonna", "wanna", "gotta", "like",
}
# Trailing fillers and hesitation noises.
_FILLER = {"uh", "uhh", "uhm", "um", "umm", "er", "erm", "hmm", "hmmm", "mmm", "eh", "ah",
           "well", "okay", "ok", "kay", "yeah", "yep", "so", "right"}


def looks_complete(text: str) -> Optional[bool]:
    """Cheap tri-state read on a transcript.

    True  -> reads finished, answer it without spending a model call.
    False -> reads mid-thought, keep listening without spending a model call.
    None  -> genuinely ambiguous, worth one short classification call.

    Deliberately biased against True: answering early is the bug we are fixing, and the
    hard-silence cap upstream still guarantees a reply when the caller really has trailed
    off, so a wrong False costs a little dead air rather than a fragment answer.
    """
    t = (text or "").strip()
    if not t:
        return None
    words = t.split()
    last = words[-1].strip(".,!?;:—-").lower()
    if last in _HANGING or last in _FILLER:
        return False
    if t[-1] in ".!?" and len(words) >= 3:
        return True
    # A single short word with no terminator ("hey", "yeah") is a complete turn only when
    # it is not one of the hesitation fillers already handled above.
    if len(words) == 1 and t[-1] not in ",;:":
        return True
    return None


class TurnState:
    """Frame-driven utterance buffer and reply-ownership state machine.

    The caller feeds every inbound media frame to push() and performs the actions it
    returns. Nothing here touches a socket, a model or a clock.
    """

    def __init__(self, *, threshold: int = None, min_ms: int = None, extended_ms: int = None,
                 hard_max_ms: int = None, max_utterance_ms: int = None) -> None:
        self.threshold = threshold if threshold is not None else VAD_LEVEL
        self.min_ms = min_ms if min_ms is not None else MIN_SILENCE_MS
        self.extended_ms = extended_ms if extended_ms is not None else EXTENDED_SILENCE_MS
        self.hard_max_ms = hard_max_ms if hard_max_ms is not None else HARD_MAX_SILENCE_MS
        self.max_utterance_ms = (max_utterance_ms if max_utterance_ms is not None
                                 else MAX_UTTERANCE_MS)
        self.epoch = 0                 # bumped whenever we abandon work in flight
        self.buffered: List[bytes] = []
        self.buffered_ms = 0
        self.pending_text = ""         # transcript held back because the caller wasn't done
        self.silence_ms = 0
        self.speech_seen = False
        self.in_pass = False           # an STT/gate pass is running
        self.playing = False           # our audio is queued on the wire
        self.held_once = False         # at least one utterance held back for this turn
        self.next_window_ms = self.min_ms   # how long a pause must last before a pass
        self.new_speech_since_pass = False  # anything new to transcribe?

    # ------------------------------------------------------------------ inbound frames
    def push(self, level: int, chunk: bytes) -> List[Dict[str, Any]]:
        """Feed one media frame; returns the actions the caller must perform."""
        actions: List[Dict[str, Any]] = []
        chunk_ms = max(1, len(chunk) // 8)          # PCMU @ 8kHz = 8 bytes per ms

        if level > self.threshold:
            if self.playing:
                # Barge-in: stop our audio. This is separate from cancelling the pass,
                # because audio already queued at the carrier keeps playing otherwise.
                actions.append({"type": "barge_in"})
                self.playing = False
            if self.in_pass:
                # They resumed before we answered: abandon the work in flight so its
                # reply can never be sent over the top of them. The held transcript is
                # kept, so the eventual answer still covers the whole utterance.
                self.epoch += 1
                self.in_pass = False
                actions.append({"type": "cancel"})
            self.speech_seen = True
            self.new_speech_since_pass = True
            self.silence_ms = 0
            self.buffered.append(chunk)
            self.buffered_ms += chunk_ms
            if self.buffered_ms >= self.max_utterance_ms and not self.in_pass:
                # A monologue with no real pause: evaluate so the reply is not starved.
                actions.append({"type": "evaluate", "force": True})
            return actions

        # Silence only means something while we are collecting an utterance, or while we
        # are holding one open waiting for it to continue. Without the second case the
        # hard cap below could never be reached, because a held pass clears speech_seen.
        if not self.speech_seen and not self.held_once:
            return actions

        self.silence_ms += chunk_ms
        self.buffered.append(chunk)                 # keep trailing silence for the STT model
        self.buffered_ms += chunk_ms

        if self.in_pass:
            return actions                        # one pass at a time

        if self.held_once and not self.new_speech_since_pass:
            # We already hold this audio's transcript and verdict, so re-transcribing it
            # would only burn calls. Wait for a continuation, or answer at the cap with
            # what we already have.
            if self.silence_ms >= self.hard_max_ms:
                actions.append({"type": "force_reply"})
            return actions

        if self.silence_ms >= self.next_window_ms:
            actions.append({"type": "evaluate", "force": False})
        return actions

    # ------------------------------------------------------------------ pass lifecycle
    def begin_pass(self):
        """Claim the buffered audio for an STT + gate pass. Returns (epoch, audio, held)."""
        audio = b"".join(self.buffered)
        held = self.pending_text
        self.buffered.clear()
        self.buffered_ms = 0
        self.silence_ms = 0
        self.speech_seen = False
        self.new_speech_since_pass = False
        self.in_pass = True
        return self.epoch, audio, held

    def begin_forced_pass(self):
        """Give up waiting for a continuation. Returns (epoch, transcript) with no STT."""
        full = self.pending_text
        self.pending_text = ""
        self.held_once = False
        self.next_window_ms = self.min_ms
        self.buffered.clear()
        self.buffered_ms = 0
        self.silence_ms = 0
        self.speech_seen = False
        self.new_speech_since_pass = False
        self.in_pass = True
        return self.epoch, full

    def verdict_wait(self, text: str, hanging: bool = False) -> None:
        """Caller is mid-thought: hold this text and keep listening.

        `hanging` means the clause plainly dangles (an obvious mid-thought), which earns a
        longer wait before the next check. An ambiguous clause gets checked again promptly
        so the final answer is not pushed out by dead air.
        """
        self.pending_text = f"{self.pending_text} {text}".strip() if self.pending_text else text
        self.held_once = True
        self.next_window_ms = self.extended_ms if hanging else self.min_ms
        self.in_pass = False

    def verdict_reply(self) -> str:
        """Caller is done: hand back the whole utterance and start owning playback."""
        full = self.pending_text
        self.pending_text = ""
        self.held_once = False
        self.next_window_ms = self.min_ms
        self.in_pass = False
        self.playing = True
        return full

    def verdict_empty(self) -> None:
        """Nothing transcribable (or a failure): release the pass, keep any held text."""
        self.in_pass = False

    def playback_started(self) -> None:
        self.playing = True

    def playback_finished(self) -> None:
        self.playing = False

    def superseded(self, epoch: int) -> bool:
        """True when a pass started at `epoch` should no longer produce audio."""
        return epoch != self.epoch

    def utterance_text(self, text: str) -> str:
        """Join a fresh transcript onto anything held back."""
        return f"{self.pending_text} {text}".strip() if self.pending_text else text
