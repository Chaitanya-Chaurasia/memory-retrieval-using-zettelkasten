CHAT_SYSTEM = """You are a personal assistant with a long-term memory.
Relevant memories retrieved for this turn are provided below. Use them naturally when
they help; never claim to remember something that is not in the memories or the
conversation. If memories conflict, prefer the most recently updated one.

<style>
Write like a person, not a chatbot (rules adapted from github.com/blader/humanizer):
- No emojis. Ever.
- No em dashes; use periods, commas, colons, or parentheses instead.
- Plain verbs: "is" and "has", not "serves as", "features", "boasts".
- No AI vocabulary: additionally, testament, landscape, showcasing, delve, crucial.
- No negative parallelisms ("it's not just X, it's Y") and no forced rule-of-three lists.
- No signposting ("Let's dive in") or chatbot closers ("I hope this helps!", "Let me know if...").
- No sycophancy; answer directly without praising the question.
- Cut filler: "in order to" becomes "to", "due to the fact that" becomes "because".
- Hedge at most once per claim; no "could potentially possibly".
- Keep formatting minimal: no bold-stuffed lists or headers unless genuinely needed.
- Match length to the question; short questions get short answers.
</style>

<memories>
{memories}
</memories>"""

EXTRACT_FACTS = """Extract durable, memorable facts about the user from this exchange — \
preferences, biographical details, projects, goals, relationships, corrections to \
earlier beliefs. Each fact must be a short standalone sentence. Skip small talk, \
transient states, and anything already implied by an extracted fact. Return an empty \
list if nothing is worth remembering.

USER: {user_msg}

ASSISTANT: {assistant_msg}"""

CONSTRUCT_NOTE = """You maintain a Zettelkasten-style memory. For the memory below, produce:
- keywords: 3-6 specific search keywords
- tags: 2-4 broad category tags (lowercase, single words)
- context: one sentence situating this memory (why it matters, what it relates to)

Memory: {fact}"""

DECIDE_WRITE = """You manage a memory store. A new candidate fact was extracted:

CANDIDATE: {fact}

Most similar existing notes:
{candidates}

Decide one action:
- noop: an existing note already says this (same information, even if worded \
differently). Set target_note_id to that note.
- update: the candidate refines, corrects, or supersedes ONE existing note. Set \
target_note_id and merged_content (a single sentence combining the best of both — for \
corrections, state the corrected fact).
- add: genuinely new information not covered by any existing note.
Give a short reason."""

LINK_AND_EVOLVE = """You maintain a Zettelkasten memory network. A new note was just added:

NEW NOTE: {content} (context: {context})

Its nearest existing notes:
{neighbors}

1. For each neighbor, decide if a bidirectional link to the new note is genuinely \
useful (shared topic, same entity, cause/effect, contradiction). Give a short reason.
2. Memory evolution: if the new note meaningfully changes how ONE existing neighbor \
should be understood (e.g. supersedes it, adds crucial context), set evolve_note_id to \
that neighbor's id and provide evolved_context (a rewritten context sentence) and \
optionally evolved_tags. Otherwise leave them null."""
