"""Behavioural tests for turn.py. No network, no audio device, no phone call.

Run: python3 test_turn.py

The headline case is the reported bug: one sentence, three pauses, and the old loop
answered each pause. Here that must produce exactly one reply, covering the whole
sentence.
"""
import sys

from turn import TurnState, looks_complete

CHUNK_MS = 20
LOUD, QUIET = 700, 60          # RMS levels either side of the 500 threshold
FAILURES = []


def frames(ms, level):
    return [(level, b"\x00" * (CHUNK_MS * 8)) for _ in range(ms // CHUNK_MS)]


class Harness:
    """Drives TurnState the way the media loop does, with scripted STT text + verdicts."""

    def __init__(self, texts, verdicts=(), hold_first_pass=False):
        self.s = TurnState()
        self.texts = list(texts)          # transcript returned by each pass, in order
        self.verdicts = list(verdicts)    # gate verdicts consumed per ambiguous pass
        self.replies = []
        self.gate_calls = 0
        self.passes = 0
        self.barges = 0
        self.cancels = 0
        self.in_flight = False
        self.hold_first_pass = hold_first_pass
        self.last_text = ""

    def _next_verdict(self):
        self.gate_calls += 1
        return bool(self.verdicts.pop(0)) if self.verdicts else True

    def start_pass(self, force):
        self.passes += 1
        epoch, audio, held = self.s.begin_pass()
        if self.hold_first_pass and self.passes == 1:
            self.in_flight = True         # a pass that never completes, to test cancel
            return
        # Transcribing the same audio again yields the same words, so when the script runs
        # out we reuse the last transcript rather than pretending the caller went silent.
        text = self.texts.pop(0) if self.texts else self.last_text
        self.last_text = text
        self.in_flight = False
        if not text:
            self.s.verdict_empty()
            return
        full = f"{held} {text}".strip() if held else text
        if not force:
            verdict = looks_complete(text)
            hanging = verdict is False      # dangling clause: plainly mid-thought
            if verdict is None:
                verdict = self._next_verdict()
            if verdict is False:
                self.s.verdict_wait(text, hanging=hanging)
                return
        self.s.verdict_reply()
        self.replies.append(full)

    def forced_reply(self):
        """The cap fired: answer with the transcript we already hold, no STT call."""
        self.passes += 1
        epoch, full = self.s.begin_forced_pass()
        self.s.verdict_reply()
        self.replies.append(full)

    def feed(self, ms, level):
        for lv, chunk in frames(ms, level):
            for act in self.s.push(lv, chunk):
                if act["type"] == "barge_in":
                    self.barges += 1
                elif act["type"] == "cancel":
                    self.cancels += 1
                    self.in_flight = False
                elif act["type"] == "evaluate":
                    self.start_pass(act["force"])
                elif act["type"] == "force_reply":
                    self.forced_reply()


def check(name, got, want):
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {got!r}, want {want!r}")
    if not ok:
        FAILURES.append(name)


print("looks_complete (the cheap gate, no model call)")
check("hanging conjunction", looks_complete("so I was thinking and"), False)
check("trailing filler", looks_complete("yeah I mean um"), False)
check("finished sentence", looks_complete("What time is it?"), True)
check("single greeting", looks_complete("hey"), True)
check("ambiguous clause", looks_complete("could you tell me about the thing"), None)
check("empty", looks_complete("   "), None)

print("\nscenario A: one sentence, three pauses (the reported bug)")
h = Harness(texts=["so I was thinking", "maybe we could go", "to the museum tomorrow"],
            verdicts=[False, False, True])
h.feed(300, LOUD)
h.feed(1200, QUIET)          # first pause: pass runs, gate says WAIT
check("  no reply on the first pause", h.replies, [])
h.feed(400, LOUD)
h.feed(1200, QUIET)          # second pause: WAIT again
check("  no reply on the second pause", h.replies, [])
h.feed(500, LOUD)
h.feed(1200, QUIET)          # third pause: DONE
check("  exactly one reply", len(h.replies), 1)
check("  reply covers the whole sentence",
      h.replies[0] if h.replies else None,
      "so I was thinking maybe we could go to the museum tomorrow")
check("  one gate call per ambiguous pause", h.gate_calls, 3)

print("\nscenario B: a complete sentence answers immediately, with no gate call")
h = Harness(texts=["What time is it?"])
h.feed(300, LOUD)
h.feed(1200, QUIET)
check("  one reply", len(h.replies), 1)
check("  no gate call needed", h.gate_calls, 0)

print("\nscenario C: caller speaks over generation -> the pass is cancelled")
h = Harness(texts=["tell me a long story about"], hold_first_pass=True)
h.feed(300, LOUD)
h.feed(1200, QUIET)          # pass starts and stays in flight
check("  pass is in flight", h.in_flight, True)
h.feed(300, LOUD)            # caller resumes mid-generation
check("  pass cancelled", h.cancels, 1)
check("  still no reply", h.replies, [])

print("\nscenario D: caller speaks over playback -> barge-in, no cancel needed")
h = Harness(texts=["sure thing"])
h.s.playing = True
h.feed(200, LOUD)
check("  barge-in fired", h.barges, 1)
check("  playback state cleared", h.s.playing, False)

print("\nscenario E: silence alone never triggers a pass")
h = Harness(texts=[])
h.feed(5000, QUIET)
check("  no passes", h.passes, 0)
check("  no replies", h.replies, [])

print("\nscenario F: gate keeps saying WAIT, then the caller goes quiet")
h = Harness(texts=["I was going to say something about the"], verdicts=[False])
h.feed(300, LOUD)
h.feed(1200, QUIET)          # WAIT -> held, and the audio is not transcribed again
check("  held, no reply yet", h.replies, [])
check("  held audio not re-transcribed", h.passes, 1)
h.feed(4000, QUIET)          # past the hard cap -> forced answer, never left hanging
check("  forced reply at the cap", len(h.replies), 1)
check("  only one extra pass", h.passes, 2)

print("\nscenario G: speech during a held utterance keeps the whole transcript")
h = Harness(texts=["I wanted to ask about", "the voice line"], verdicts=[True])
h.feed(300, LOUD)
h.feed(1200, QUIET)          # dangling clause: held with no gate call at all
check("  held on the dangling clause, no gate call", (h.replies, h.gate_calls), ([], 0))
h.feed(300, LOUD)            # resumes before we ever answered
check("  resume cancelled nothing (nothing was in flight)", h.cancels, 0)
h.feed(1800, QUIET)          # completes the thought -> gate says DONE
check("  one reply", len(h.replies), 1)
check("  transcript stitched across the pause", h.replies[0] if h.replies else None,
      "I wanted to ask about the voice line")

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("all turn-taking checks passed")
