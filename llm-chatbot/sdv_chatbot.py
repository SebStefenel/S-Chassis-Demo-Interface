#!/usr/bin/env python3
"""
SDV In-Car AI Chatbot
=====================
Streamlit frontend backed by the Groq API (llama-3.1-8b-instant).
Maintains a persistent driver profile in vehicle_history.json with
dated entries so stale facts can be pruned and hot facts can be merged.

Run:
    streamlit run sdv_chatbot.py
"""

import json
import logging
import os
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SDV] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sdv_chatbot")

# ── Driver identity (injected by main.py orchestrator via env var) ────────────
# When launched standalone, DRIVER_NAME is empty and the single shared history
# file is used (backwards-compatible).  When launched by main.py after face
# recognition, DRIVER_NAME is set and each driver gets their own history file
# stored in ~/.sdv/ so memories never bleed between drivers.
DRIVER_NAME: str = os.environ.get("SDV_DRIVER_NAME", "").strip()

_HISTORY_DIR = Path(os.path.expanduser("~/.sdv"))
_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_ID          = "llama-3.1-8b-instant"
HISTORY_FILE      = (
    _HISTORY_DIR / f"chat_history_{DRIVER_NAME}.json"
    if DRIVER_NAME
    else Path("vehicle_history.json")
)
API_TIMEOUT       = 20          # seconds – critical on moving-vehicle networks
MAX_RETRIES       = 1           # one silent retry on transient failures

# History caps — exceed either and condensation is triggered.
HISTORY_ENTRY_CAP = 60          # max number of fact entries
HISTORY_CHAR_CAP  = 5_000       # max serialised chars of the entries array

# Facts not mentioned in this many days AND with low mention counts are pruned.
STALE_DAYS        = 180         # 6 months
STALE_MIN_COUNT   = 2           # keep if mentioned at least this many times

# Maximum chars of the history snippet injected into the system prompt.
# Keeps the context window healthy on low-memory Pi sessions.
PROMPT_SNIPPET_CAP = 3_000

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv()
ENV_API_KEY: str = os.getenv("GROQ_API_KEY", "")

# Placeholder sentinel so users can search for it easily.
GROQ_API_KEY = "YOUR_API_KEY_HERE"   # override via .env or sidebar


# ═════════════════════════════════════════════════════════════════════════════
#  History helpers
# ═════════════════════════════════════════════════════════════════════════════

def _today() -> str:
    return date.today().isoformat()


def _now_utc() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _blank_history() -> dict:
    return {
        "schema_version": "1.1",
        "created_at": _now_utc(),
        "last_updated": _now_utc(),
        "driver_profile": {
            "name": DRIVER_NAME or None,
        },
        "vehicle_context": {
            "vehicle_type": "SDV demo car",
            "notes": [],
        },
        # Each entry:
        # {
        #   "id": str,
        #   "category": str,
        #   "fact": str,
        #   "confidence": "high"|"medium"|"low",
        #   "first_mentioned": ISO date str,
        #   "last_mentioned": ISO date str,
        #   "mention_count": int,
        #   "mentioned_on": [ISO date str, ...]   ← used for recency & de-dup
        # }
        "entries": [],
    }


def load_history() -> dict:
    """Read vehicle_history.json; re-initialise cleanly if missing or corrupt."""
    if HISTORY_FILE.exists():
        try:
            data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            log.info("History loaded: %d entries from %s", len(data.get("entries", [])), HISTORY_FILE)
            return data
        except Exception as exc:
            log.warning("Corrupt history file – reinitialising. Reason: %s", exc)

    fresh = _blank_history()
    HISTORY_FILE.write_text(json.dumps(fresh, indent=2), encoding="utf-8")
    log.info("Initialised fresh history file at %s", HISTORY_FILE.resolve())
    return fresh


def save_history(history: dict) -> None:
    history["last_updated"] = _now_utc()
    HISTORY_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")
    log.info("History saved: %d entries, %.1f KB", len(history.get("entries", [])),
             HISTORY_FILE.stat().st_size / 1024)


def history_to_prompt_snippet(history: dict) -> str:
    """
    Convert the history dict to a compact text block for the system prompt.
    Truncates at PROMPT_SNIPPET_CAP chars so the Pi context window stays healthy.
    """
    lines = ["## Persistent Driver Profile"]

    profile = history.get("driver_profile", {})
    if profile.get("name"):
        lines.append(f"Driver name: {profile['name']}")

    ctx = history.get("vehicle_context", {})
    if ctx.get("vehicle_type"):
        lines.append(f"Vehicle: {ctx['vehicle_type']}")
    for note in ctx.get("notes", []):
        lines.append(f"  Vehicle note: {note}")

    entries = history.get("entries", [])
    if entries:
        lines.append("\n### Known Facts & Preferences (most recent dates shown)")
        for e in sorted(entries, key=lambda x: x.get("last_mentioned", ""), reverse=True):
            recent_dates = ", ".join(e.get("mentioned_on", [])[-3:])
            count        = e.get("mention_count", 1)
            last         = e.get("last_mentioned", "?")
            lines.append(
                f"- [{e.get('category','general')}] {e['fact']} "
                f"(×{count}, last {last}"
                + (f", dates: {recent_dates}" if recent_dates else "")
                + ")"
            )
    else:
        lines.append("No persistent facts stored yet.")

    snippet = "\n".join(lines)
    if len(snippet) > PROMPT_SNIPPET_CAP:
        snippet = snippet[:PROMPT_SNIPPET_CAP] + "\n... [history truncated for context limit]"
    return snippet


def _entries_char_len(history: dict) -> int:
    return len(json.dumps(history.get("entries", [])))


# ═════════════════════════════════════════════════════════════════════════════
#  Groq API helpers
# ═════════════════════════════════════════════════════════════════════════════

def make_client(api_key: str):
    """Lazy-import and return a Groq client to keep startup fast on Pi."""
    try:
        from groq import Groq
        return Groq(api_key=api_key)
    except ImportError as exc:
        raise RuntimeError(
            "groq package not installed. Run: pip install groq"
        ) from exc


def chat_completion(client, messages: list[dict], temperature: float = 0.7,
                    max_tokens: int = 1024) -> str:
    """
    Call the Groq API with retry-on-transient-failure.
    Returns the reply string or a user-visible error message.
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=API_TIMEOUT,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            log.warning("API attempt %d/%d failed: %s", attempt + 1, MAX_RETRIES + 1, exc)
            if attempt == MAX_RETRIES:
                return f"[Connection error — check network or API key: {exc}]"
    return "[Unexpected error]"


# ═════════════════════════════════════════════════════════════════════════════
#  Memory extraction and condensation prompts
# ═════════════════════════════════════════════════════════════════════════════

_EXTRACT_SYSTEM = """\
You are a memory-extraction assistant for an in-car AI system.
Given a conversation transcript, extract ONLY new long-term facts about the driver
that are worth remembering across future sessions. Do NOT re-extract facts already
listed in the existing profile.

Prioritise SPECIFIC, CONCRETE facts over vague generalisations. Examples of good facts:
  - "Driver visited Patty's burger joint and enjoyed it" (not "driver likes burgers")
  - "Driver prefers chicken and beef burger options" (not "driver asks about menu items")
  - "Driver likes jazz music" (not "driver asks about music")

Capture all of the following when present:
  - Specific places visited and whether the driver liked or disliked them
  - Named preferences (specific foods, artists, genres, settings)
  - Concrete habits or recurring behaviours
  - Personal details explicitly shared

Return ONLY a JSON array of objects with exactly these keys:
  "category": one of: food_preference | music_preference | driving_habit |
               vehicle_setting | personal_detail | frequent_request | experience | general
  "fact": a short, specific, self-contained statement about the driver (not the conversation).
          Always name specific places/items when mentioned. Never write generic summaries.
  "confidence": "high" | "medium" | "low"

If nothing new is worth storing, return an empty array [].
No prose, no markdown fences, no extra keys.
"""

_CONDENSE_SYSTEM = f"""\
You are a memory-condensation assistant for an in-car AI system.
The vehicle history entries array has exceeded its size budget. Condense it by:

1. MERGE: Combine entries that describe the same preference or habit into one
   (merge their mentioned_on date lists, set mention_count to the total sum).
2. PRUNE: Remove entries where last_mentioned is more than {STALE_DAYS} days ago
   AND mention_count < {STALE_MIN_COUNT} — these are likely stale noise.
3. KEEP ALWAYS: Any entry with mention_count >= 3 regardless of age.
4. DATE COMBINING: When merging, union the mentioned_on lists and keep them sorted.
5. Preserve the exact entry object schema — do not add or remove keys.

Return ONLY the condensed "entries" JSON array, no prose, no markdown fences.
"""


def extract_new_facts(client, conversation: list[dict], history: dict) -> list[dict]:
    """Ask the LLM to pull memorable facts from the completed session."""
    log.info("Extracting new facts from session (%d turns)…", len(conversation))

    existing_snippet = history_to_prompt_snippet(history)
    transcript = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in conversation
        if m["role"] in ("user", "assistant")
    )
    user_payload = (
        f"Existing driver profile:\n{existing_snippet}\n\n"
        f"Today's date: {_today()}\n\n"
        f"Conversation transcript:\n{transcript}"
    )
    raw = chat_completion(
        client,
        [
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user",   "content": user_payload},
        ],
        temperature=0.15,
        max_tokens=512,
    )
    try:
        facts = json.loads(raw)
        if not isinstance(facts, list):
            raise ValueError("Expected a JSON array")
        log.info("Extracted %d new fact(s) from session", len(facts))
        return facts
    except Exception as exc:
        log.warning("Could not parse extracted facts (%s). Raw: %.200s", exc, raw)
        return []


def merge_facts_into_history(history: dict, new_facts: list[dict]) -> dict:
    """
    Upsert extracted facts into the history entries list.
    Matching is done by lowercased fact text to avoid exact-duplicate entries.
    """
    today = _today()
    # Build lookup by normalised fact text
    existing: dict[str, dict] = {
        e["fact"].lower().strip(): e for e in history.setdefault("entries", [])
    }

    for fact_obj in new_facts:
        text = fact_obj.get("fact", "").strip()
        if not text:
            continue
        key = text.lower()

        if key in existing:
            # Update: add today's date if not already present, increment count
            entry = existing[key]
            if today not in entry.get("mentioned_on", []):
                entry.setdefault("mentioned_on", []).append(today)
                entry.setdefault("mentioned_on", []).sort()
            entry["mention_count"] = entry.get("mention_count", 1) + 1
            entry["last_mentioned"] = today
            log.info("  ↑ Updated: '%s' (×%d)", text[:70], entry["mention_count"])
        else:
            # New entry
            entry = {
                "id":              str(uuid.uuid4())[:8],
                "category":        fact_obj.get("category", "general"),
                "fact":            text,
                "confidence":      fact_obj.get("confidence", "medium"),
                "first_mentioned": today,
                "last_mentioned":  today,
                "mention_count":   1,
                "mentioned_on":    [today],
            }
            history["entries"].append(entry)
            existing[key] = entry
            log.info("  + Added:   '%s'", text[:70])

    return history


def condense_history_if_needed(client, history: dict) -> dict:
    """
    If the entries array exceeds caps, ask the LLM to condense it.
    Two independent caps: character length and entry count.
    """
    char_len    = _entries_char_len(history)
    entry_count = len(history.get("entries", []))
    log.info("History stats: %d entries, %d chars", entry_count, char_len)

    if char_len <= HISTORY_CHAR_CAP and entry_count <= HISTORY_ENTRY_CAP:
        log.info("History within caps — no condensation needed.")
        return history

    log.info("History over cap (entries=%d/%d, chars=%d/%d) — condensing…",
             entry_count, HISTORY_ENTRY_CAP, char_len, HISTORY_CHAR_CAP)

    user_payload = (
        f"Today's date: {_today()}\n"
        f"Stale threshold: {STALE_DAYS} days, min_count for retention: {STALE_MIN_COUNT}\n\n"
        f"Entries:\n{json.dumps(history.get('entries', []), indent=2)}"
    )
    raw = chat_completion(
        client,
        [
            {"role": "system", "content": _CONDENSE_SYSTEM},
            {"role": "user",   "content": user_payload},
        ],
        temperature=0.05,
        max_tokens=2048,
    )
    try:
        condensed = json.loads(raw)
        if not isinstance(condensed, list):
            raise ValueError("Expected a JSON array")
        old_count = len(history["entries"])
        history["entries"] = condensed
        log.info("Condensed: %d → %d entries", old_count, len(condensed))
    except Exception as exc:
        log.warning("Condensation parse failed (%s). Keeping original entries.", exc)

    return history


def save_and_sync_session(client, messages: list[dict], history: dict) -> tuple[dict, str]:
    """
    Full end-of-session pipeline:
      1. Extract new facts from the conversation.
      2. Merge into existing history.
      3. Condense if over caps.
      4. Persist to disk.
    Returns (updated_history, status_message).
    """
    if not any(m["role"] == "user" for m in messages):
        return history, "Nothing to sync — no user messages in this session."

    new_facts = extract_new_facts(client, messages, history)
    history   = merge_facts_into_history(history, new_facts)
    history   = condense_history_if_needed(client, history)
    save_history(history)

    added = len(new_facts)
    total = len(history.get("entries", []))
    return history, f"Synced — {added} fact(s) added, {total} total stored."


# ═════════════════════════════════════════════════════════════════════════════
#  System prompt
# ═════════════════════════════════════════════════════════════════════════════

_BASE_SYSTEM = """\
You are an intelligent in-car AI assistant embedded in a Software-Defined Vehicle (SDV) demo car.
Your role: help the driver with navigation hints, music, nearby food, vehicle settings,
weather, and general conversation — concisely and safely.

Safety rule: keep all responses short. The driver may be operating a vehicle.
Prefer 1–3 sentence answers. Never produce long bulleted lists unprompted.
If the driver asks a complex question, give the key answer first, then offer to elaborate.
"""


def build_system_prompt(history: dict) -> str:
    snippet = history_to_prompt_snippet(history)
    prompt  = _BASE_SYSTEM + "\n\n" + snippet
    if DRIVER_NAME:
        prompt += (
            f"\n\nThe driver has been identified by face recognition as {DRIVER_NAME}. "
            f"Always address them by name naturally. "
            f"Keep a warm, personal tone throughout the conversation."
        )
    log.info("System prompt assembled — history snippet: %d chars", len(snippet))
    return prompt


# ═════════════════════════════════════════════════════════════════════════════
#  Streamlit UI
# ═════════════════════════════════════════════════════════════════════════════

def _init_session_state():
    if "messages"      not in st.session_state:
        st.session_state.messages      = []   # user+assistant turns only
    if "history"       not in st.session_state:
        st.session_state.history       = load_history()
    if "client"        not in st.session_state:
        st.session_state.client        = None
    if "session_saved" not in st.session_state:
        st.session_state.session_saved = False
    if "status_msg"    not in st.session_state:
        st.session_state.status_msg    = ""
    if "greeted"       not in st.session_state:
        st.session_state.greeted       = False


def _resolve_client(api_key: str):
    """Cache the Groq client in session state; recreate only if key changes."""
    cached = st.session_state.client
    # Store key hash so we detect changes without exposing the key
    current_hash = hash(api_key)
    if cached is None or st.session_state.get("_api_key_hash") != current_hash:
        try:
            st.session_state.client        = make_client(api_key)
            st.session_state._api_key_hash = current_hash
            log.info("Groq client (re)initialised.")
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()
    return st.session_state.client


def main():
    page_title = f"SDV Assistant — {DRIVER_NAME}" if DRIVER_NAME else "SDV Car Assistant"
    st.set_page_config(
        page_title=page_title,
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_session_state()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🚗 SDV Assistant")

        # ── API key resolution ─────────────────────────────────────────────
        st.subheader("API Configuration")
        api_key: str = ENV_API_KEY
        if api_key:
            st.success("API key loaded from `.env`", icon="✅")
        else:
            api_key = st.text_input(
                "Groq API Key",
                type="password",
                placeholder="gsk_…",
                help="Paste your Groq key here, or set GROQ_API_KEY in a .env file.",
            )
            if not api_key:
                st.info("Enter your API key above to enable chat.", icon="ℹ️")

        st.divider()

        # ── Driver memory panel ────────────────────────────────────────────
        st.subheader("Driver Memory")
        history     = st.session_state.history
        entry_count = len(history.get("entries", []))
        last_upd    = history.get("last_updated", "never")

        col1, col2 = st.columns(2)
        col1.metric("Stored facts", entry_count)
        col2.metric("Last sync", last_upd[11:16] if "T" in last_upd else "—")

        if entry_count:
            with st.expander("View stored facts", expanded=False):
                for e in sorted(
                    history["entries"],
                    key=lambda x: x.get("last_mentioned", ""),
                    reverse=True,
                ):
                    count = e.get("mention_count", 1)
                    last  = e.get("last_mentioned", "?")
                    st.markdown(
                        f"**{e.get('category','?')}** · {e['fact']}  \n"
                        f"<sub>×{count} · last {last} · {e.get('confidence','?')} confidence</sub>",
                        unsafe_allow_html=True,
                    )
        else:
            st.caption("No facts stored yet — chat first, then sync.")

        st.divider()

        # ── Session controls ───────────────────────────────────────────────
        st.subheader("Session Controls")

        if st.button("💾 Save & Sync Session", use_container_width=True,
                     help="Extract new memories from this session and save to history"):
            if not api_key:
                st.error("API key required to sync.")
            elif not st.session_state.messages:
                st.warning("No conversation to sync yet.")
            else:
                client = _resolve_client(api_key)
                with st.spinner("Syncing memories…"):
                    updated, msg = save_and_sync_session(
                        client,
                        st.session_state.messages,
                        st.session_state.history,
                    )
                st.session_state.history       = updated
                st.session_state.session_saved = True
                st.session_state.status_msg    = msg
                st.rerun()

        if st.session_state.session_saved:
            st.success(st.session_state.status_msg or "Session synced.", icon="✅")

        if st.button("🗑️ Clear Chat (keep memory)", use_container_width=True):
            st.session_state.messages      = []
            st.session_state.session_saved = False
            st.session_state.status_msg    = ""
            st.rerun()

        st.divider()

        # ── End Session (only shown when launched by the orchestrator) ─────
        if DRIVER_NAME:
            st.subheader("Drive Mode")
            st.caption(
                "Click below when you're done chatting. "
                "The system will start drowsiness monitoring."
            )
            if st.button("🚗 End Session & Start Driving",
                         use_container_width=True, type="primary"):
                # Sync unsaved conversation before exiting
                if api_key and st.session_state.messages and not st.session_state.session_saved:
                    client = _resolve_client(api_key)
                    with st.spinner("Saving your session…"):
                        updated, _ = save_and_sync_session(
                            client,
                            st.session_state.messages,
                            st.session_state.history,
                        )
                    st.session_state.history = updated
                st.info("Starting drowsiness monitor… you can close this tab.")
                # os._exit terminates the Streamlit server process immediately,
                # which unblocks proc.wait() in main.py and triggers MONITOR phase.
                import os as _os
                _os._exit(0)
            st.divider()

        st.caption(f"Model: `{MODEL_ID}`  |  Timeout: {API_TIMEOUT}s")

    # ── Main chat area ─────────────────────────────────────────────────────
    st.title("SDV In-Car AI Assistant")

    if not api_key:
        st.warning("Enter your Groq API key in the sidebar to start chatting.")
        st.stop()

    client = _resolve_client(api_key)

    # ── Opening greeting (only when driver is known and chat is fresh) ─────────
    # Generates one assistant message that addresses the driver by name, using
    # their stored history as context.  Runs once per session, not on every rerun.
    if DRIVER_NAME and not st.session_state.greeted and not st.session_state.messages:
        with st.spinner(f"Welcome back, {DRIVER_NAME}…"):
            greeting_prompt = build_system_prompt(st.session_state.history) + (
                f"\n\nGreet {DRIVER_NAME} warmly and by name. "
                f"Keep it to 1–2 sentences. Do not ask how you can help yet."
            )
            greeting = chat_completion(
                client,
                [{"role": "system", "content": greeting_prompt}],
                temperature=0.8,
                max_tokens=80,
            )
        st.session_state.messages.append({"role": "assistant", "content": greeting})
        st.session_state.greeted = True
        st.rerun()

    # Render message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Prompt new input
    user_input = st.chat_input("Ask your car assistant anything…")
    if user_input:
        # Show and record user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Build context: system prompt (with injected history) + full turn history
        system_prompt = build_system_prompt(st.session_state.history)
        full_context  = [{"role": "system", "content": system_prompt}] + st.session_state.messages

        # Get and show assistant reply
        with st.chat_message("assistant"):
            with st.spinner(""):
                reply = chat_completion(client, full_context)
            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.session_state.session_saved = False  # mark as dirty until next sync


if __name__ == "__main__":
    main()
