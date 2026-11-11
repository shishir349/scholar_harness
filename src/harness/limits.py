"""The quality gates of CONSTRAINTS.md as numbers. Every check, repair and prompt reads them from here; nothing re-types them.
Imports nothing, so any module may use it."""
MIN_SENTS = 3             # a body paragraph has at least this many sentences...
MIN_WORDS = 70            # ...and this many words
MIN_SKELETON_WORDS = 65   # a mined skeleton is dropped when a body paragraph of its chunk is shorter than this (or under MIN_SENTS sentences)
STD_MIN = 6.8             # intra-paragraph sentence-length std (words) of a 3+ sentence paragraph
MAX_SENT = 42             # longest sentence, words
SHORT_MAX = 11            # every 3+ sentence paragraph holds a sentence of this many words or fewer
FRAGMENT = 5              # two sentences under this are stacked fragments
STACK = 10                # three sentences in a row under this are stacked fragments
NARR_CALL = 2             # "Author (Year) verb ..." sentences per call (open / mid / close)
NARR_SECTION = 5          # ...and per section
PAREN_SHARE = 0.70        # at least this share of citations are parenthetical
DUP_JACCARD = 0.35        # content-word overlap at which two sentences of one section say the same thing
SURPRISAL_LOW = 3.6       # nats/token under which a sentence is "predictable" (default until scripts/calibrate_surprisal.py writes data/surprisal_baseline.json)
SURPRISAL_GAIN = 0.4      # a rewrite of a low-surprisal sentence must raise its mean by at least this
