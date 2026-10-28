"""
Dev only. Fill a running record service with synthetic turns.

Posts to the service's own /turn endpoint, so it needs no access to
the store and can run while the UI is open (the ledger grows live).

    .venv/bin/python seed.py --turns 12000 --days 120

The archive it leaves behind has what the UI needs to be exercised
properly: several agents at three depths, bursts and silences,
weekend lulls, a multi-day outage, tool calls with failures, and
turns all the way up to the present so the live edge is real.
"""

from __future__ import annotations

import argparse
import random
import time

import httpx

# id, parent, depth, weight, joined (days ago)
AGENTS = [
    ("r0", None, 0, 0.10, 120),
    ("planner", "r0", 1, 0.12, 118),
    ("coder", "r0", 1, 0.22, 117),
    ("explorer", "r0", 1, 0.16, 116),
    ("reviewer", "r0", 1, 0.11, 110),
    ("writer", "r0", 1, 0.08, 96),
    ("scraper", "explorer", 2, 0.09, 115),
    ("puller", "explorer", 2, 0.06, 100),
    ("tester", "coder", 2, 0.07, 84),
    ("archivist", "r0", 1, 0.05, 52),
    ("migrator", "r0", 1, 0.04, 34),
]

# The persona is one global file, re-read before every turn, and a turn
# carries its text only when the file changed since that agent's last
# turn -- it is the epoch marker, not a per-agent property. Two rewrites
# over the archive make the marker visible; the rest of the turns leave
# it empty, exactly like the real engine.
PERSONA_EPOCHS = [
    "You keep the session small and the changes verifiable.",
    "You keep the session small and the changes verifiable. When a "
    "trade-off is real, say which side you chose and why.",
    "You keep the session small and the changes verifiable. Answer with "
    "the smallest thing that is true; leave the rest out.",
]

MODELS = ["gpt-5", "claude-sonnet-4.5", "qwen3-coder", "deepseek-v3.2"]

TASKS = [
    "the ledger pager skips a row when a turn lands mid-page",
    "why did the histogram go flat after the retention prune",
    "rename the window strip to something honest",
    "add a test for the exit-turn rewrite",
    "the drawer shows a stale reply after a fast second turn",
    "trace a dropped injection through the hub",
    "the archive brush cannot widen once it is narrow",
    "check that paging never re-reads the same X key",
    "make idle gaps readable without adding a second scale",
    "the cursor jumps when the window slides under it",
    "reduce the bytes per turn without losing the calls",
    "does the rollup survive a restart in the middle of a batch",
    "write down why record never speaks to the agent",
    "find every place that assumes seq starts at zero",
    "the strip looks barcode-ish when the window is one day",
    "compare two accounts of the same turn",
    "the footer counts turns that were pruned",
    "make the first paint keep the previous scroll position",
    "the parent chain highlight misses a depth-2 agent",
    "wrap the collect payload so a retry is idempotent",
]

CALLS = ["read", "grep", "shell", "edit", "write", "web_search", "test"]

# The provider-neutral ToolCall shape: an id, a name, the arguments the
# model produced. The id is what pairs a call with its result.
ARGS = {
    "read": {"path": "builtin/services/record/store.py", "limit": 120},
    "grep": {"pattern": "cursor", "path": "builtin/services/record"},
    "shell": {"command": "uv run pytest -q"},
    "edit": {"path": "builtin/services/record/web/app.js"},
    "write": {"path": "docs/notes.md", "bytes": 1800},
    "web_search": {"query": "indexeddb cursor pagination"},
    "test": {"target": "builtin/services/record/tests"},
}

OUTPUTS = {
    "read": "42 lines, the pager starts at the cursor key",
    "grep": "7 matches across 3 files",
    "shell": "exit code: 0\n8 passed in 2.41s",
    "edit": "1 file changed, +12 -4",
    "write": "wrote 1.8 KB",
    "web_search": "9 results, 3 worth opening",
    "test": "8 passed in 2.41s",
}

# There is no failure flag on a result: what went wrong is in the text,
# and shell is the one verb that leads with its status line.
FAILURES = [
    "exit code: 1\nno such key: S\\x00first",
    "exit code: 124\ntimed out after 30s",
    "exit code: 1\n403 from the upstream service",
]

DELTAS = [
    ("service", "+memory online"),
    ("service", "-vision offline"),
    ("subagent", "spawned a3: scraper"),
    ("subagent", "-a3 died: killed by holder r0"),
    ("skill", "loaded pdf"),
]

SERVICES = ["memory", "network", "system", "record"]

CONTRIBUTIONS = [
    "3 related turns found for this task",
    "workspace idle 12 days; last writer was coder",
    "upstream reachable, 41ms",
]


def observation(delta, injections, inbox):
    """
    The unified observation schema, rendered the way the core does it:
    three ordered sections, each dropped entirely when empty.
    """
    parts = []

    grouped: dict[str, list[str]] = {}

    for event in delta:
        grouped.setdefault(event["section"], []).append(event["line"])

    if grouped:
        inner = []

        for section, lines in grouped.items():
            body = "\n".join(lines)
            inner.append(f"<{section}>{body}</{section}>")

        parts.append("<delta>\n" + "\n".join(inner) + "\n</delta>")

    if injections:
        sections = []

        for injection in injections:
            body = injection["text"] if injection["ok"] else f"[service error] {injection['text']}"
            sections.append(f'<service id="{injection["service_id"]}">\n{body}\n</service>')

        parts.append("<injection>\n" + "\n\n".join(sections) + "\n</injection>")

    if inbox:
        sections = []

        for message in inbox:
            attrs = f' id="{message["source"]}"' if message["kind"] in ("subagent", "unknown") else ""

            if message["event"] != "send":
                attrs += f' event="{message["event"]}"'

            sections.append(f'<{message["kind"]}{attrs}>\n{message["text"]}\n</{message["kind"]}>')

        parts.append("<inbox>\n" + "\n\n".join(sections) + "\n</inbox>")

    return "\n\n".join(parts) or "(no input this turn)"


def turn_payload(rng, agent, parent, depth, ts, uniq, persona=None):
    dur = rng.uniform(0.6, 120.0)
    calls_n = rng.choices([0, 1, 2, 3, 4], [0.24, 0.31, 0.25, 0.13, 0.07])[0]

    calls = []
    results = []

    for index in range(calls_n):
        name = rng.choice(CALLS)
        call_id = f"t{index + 1}"
        start = ts + dur * (0.15 + 0.5 * index / max(1, calls_n))
        failed = rng.random() < 0.06

        calls.append({"id": call_id, "name": name, "arguments": ARGS[name]})
        results.append(
            {
                "call_id": call_id,
                "name": name,
                "output": rng.choice(FAILURES) if failed else OUTPUTS[name],
                "started_at": start,
                "ended_at": start + rng.uniform(0.1, 24.0),
            }
        )

    task = rng.choice(TASKS)
    reply = f"{task.capitalize()}. " + rng.choice(
        [
            "The cause is the cursor, not the window.",
            "Fixed in the pager; the page boundary now resumes from the key.",
            "Confirmed against the store tests.",
            "Left it alone, the trade-off is not worth it yet.",
            "Found two more places with the same assumption.",
        ]
    )

    inbox = (
        [
            {
                "kind": "user",
                "source": "user",
                "text": task,
                "event": "send",
                "received_at": ts - rng.uniform(0.5, 40.0),
            }
        ]
        if rng.random() < 0.8
        else []
    )

    delta = [
        {"section": section, "line": line}
        for section, line in rng.sample(DELTAS, rng.randint(0, 3))
    ]

    injections = []

    if rng.random() < 0.55:
        for _ in range(rng.randint(1, 3)):
            start = ts + rng.uniform(0.0, 0.4)
            ok = rng.random() > 0.08

            injections.append(
                {
                    "service_id": rng.choice(SERVICES),
                    "text": rng.choice(CONTRIBUTIONS) if ok else "no reply within 2s",
                    "ok": ok,
                    "started_at": start,
                    "ended_at": start + rng.uniform(0.02, 0.6),
                }
            )

    llm_start = ts + min(rng.uniform(0.05, 0.6), dur * 0.4)
    llm_end = min(ts + dur, llm_start + dur * rng.uniform(0.4, 0.9))

    # The frozen run: the observation, the assistant's reply and calls,
    # then one tool message per result, keyed by the call it answers.
    rendered = [
        {"role": "user", "content": observation(delta, injections, inbox)},
        {"role": "assistant", "content": reply, "tool_calls": calls},
    ]

    for result in results:
        rendered.append(
            {
                "role": "tool",
                "tool_call_id": result["call_id"],
                "content": result["output"],
            }
        )

    return {
        "agent_id": agent,
        "parent_id": parent,
        "depth": depth,
        "state": rng.choices(["end", "running", "start"], [0.7, 0.25, 0.05])[0],
        "persona": persona,
        "inbox": inbox,
        "injections": injections,
        "delta": delta,
        "reply": reply,
        "calls": calls,
        "results": results,
        "llm": {
            "model": rng.choice(MODELS),
            "started_at": llm_start,
            "ended_at": llm_end,
            "usage": {
                "prompt_tokens": rng.randint(1_800, 62_000),
                "completion_tokens": rng.randint(80, 4_200),
            },
            "finish_reason": "stop",
            "retries": rng.choices([0, 1, 2], [0.88, 0.09, 0.03])[0],
        },
        "rendered": rendered,
        "started_at": ts + uniq * 1e-6,
        "ended_at": ts + dur,
    }


def day_times(rng, day_start, weekday):
    """Turn times for one day: a few sessions, not a uniform smear."""
    turns = int(rng.gauss(120 if weekday else 34, 26 if weekday else 12))
    turns = max(4, turns)

    times = []
    for _ in range(rng.randint(1, 3)):
        centre = rng.uniform(8.6, 21.4)
        spread = rng.uniform(0.35, 2.2)
        for _ in range(turns // rng.randint(1, 3) + 1):
            hour = min(23.4, max(7.0, rng.gauss(centre, spread)))
            times.append(day_start + hour * 3600)
    return sorted(times)


def generate(turns, days, seed):
    if turns <= 0:
        return []

    rng = random.Random(seed)
    now = time.time()
    oldest = now - days * 86_400

    # Two silences: a long one and a short one, so the ledger has
    # gaps wide enough to earn their dividers.
    silent = set(range(days - 72, days - 60)) | set(range(days - 21, days - 19))

    out = []
    for offset in range(days):
        if offset in silent:
            continue
        day_start = oldest + offset * 86_400
        weekday = time.localtime(day_start).tm_wday < 5
        ages = days - offset
        active = [a for a in AGENTS if ages <= a[4]]
        weights = [a[3] for a in active]

        for ts in day_times(rng, day_start, weekday):
            agent, parent, depth, _, _ = rng.choices(active, weights)[0]
            out.append((ts, agent, parent, depth))
            if len(out) >= turns:
                break
        if len(out) >= turns:
            break

    # The newest turns sit just behind the present, so the live edge
    # of the archive is honest.
    live = now - rng.uniform(30, 120)
    out.append((live, "r0", None, 0))
    out.append((live + 4, "coder", "r0", 1))
    out.append((live + 19, "explorer", "r0", 1))

    out.sort(key=lambda item: item[0])

    # Each agent notices a rewrite on its own next turn, so one change of
    # the file shows up as one marker per agent, not one globally.
    rewrites = [oldest + (now - oldest) * at for at in (0.34, 0.78)]
    seen = {}

    payloads = []
    for index, (ts, agent, parent, depth) in enumerate(out):
        epoch = PERSONA_EPOCHS[sum(1 for at in rewrites if at <= ts)]
        persona = None

        if agent not in seen:
            seen[agent] = epoch  # boot: already loaded, so nothing to carry
        elif seen[agent] != epoch:
            seen[agent] = epoch
            persona = epoch

        payloads.append(turn_payload(rng, agent, parent, depth, ts, index, persona))
    return payloads


def live_burst(count, seed):
    """Extra turns in the last half hour, so the default window (which
    opens on the live edge) is not an empty screen."""
    rng = random.Random(seed)
    now = time.time()
    weights = [a[3] for a in AGENTS]
    out = []
    for index in range(count):
        agent, parent, depth, _, _ = rng.choices(AGENTS, weights)[0]
        ts = now - rng.uniform(40, 1_700)
        out.append(turn_payload(rng, agent, parent, depth, ts, 9_000 + index))
    return out


def main():
    parser = argparse.ArgumentParser(description="seed a running record service")
    parser.add_argument("--turns", type=int, default=12_000)
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--live", type=int, default=0, help="extra turns in the last 30 minutes")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--url", default="http://127.0.0.1:9003")
    args = parser.parse_args()

    payloads = generate(args.turns, args.days, args.seed)
    if args.live:
        payloads += live_burst(args.live, args.seed + 1)
        payloads.sort(key=lambda payload: payload["started_at"])
    print(f"{len(payloads)} turns over {args.days} days -> {args.url}")

    started = time.time()
    sent = 0
    with httpx.Client(timeout=10.0) as client:
        for payload in payloads:
            response = client.post(f"{args.url}/turn", json={"prev_turn": payload})
            response.raise_for_status()
            sent += 1
            if sent % 1_000 == 0:
                print(f"  {sent} sent ({sent / (time.time() - started):.0f}/s)")

    print(f"done: {sent} turns in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
