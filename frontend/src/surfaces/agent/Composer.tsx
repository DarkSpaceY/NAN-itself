// Input box: the protocol has no addressed input; everything sent goes
// to the root.
//
// Drilling into a subagent only annotates the target beside the box;
// the input path does not change.

import { useState } from "react";

import type { InputPart } from "../../kernel/history/parts";
import type { Transport } from "../../transport/ws";

interface ComposerProps {
  transport: Transport;
  paused: boolean;
  /** Hint text when focus is not root, e.g. "root · viewing a3". */
  targetLabel: string | null;
  /** Messages sent while the round is still running: queued until the round boundary, not in the timeline. */
  queued: readonly InputPart[];
}

export function Composer({
  transport,
  paused,
  targetLabel,
  queued,
}: ComposerProps) {
  const [text, setText] = useState("");

  const submit = () => {
    const value = text.trim();
    if (!value) return;
    transport.send({ t: "input", text: value, mid: crypto.randomUUID() });
    setText("");
  };

  const rows = Math.min(6, Math.max(1, text.split("\n").length));

  return (
    <form
      className="composer"
      onSubmit={(event) => {
        event.preventDefault();
        submit();
      }}
    >
      {queued.length > 0 && (
        <div className="composer-queue">
          {queued.map((part) => (
            <div className="queued-msg" key={part.id}>
              <span className="queued-mark">queued</span>
              <span className="queued-text">{part.text}</span>
            </div>
          ))}
        </div>
      )}

      <textarea
        className="composer-input"
        rows={rows}
        value={text}
        placeholder="send to root…"
        onChange={(event) => setText(event.target.value)}
        onKeyDown={(event) => {
          if (
            event.key === "Enter" &&
            !event.shiftKey &&
            !event.nativeEvent.isComposing
          ) {
            event.preventDefault();
            submit();
          }
        }}
      />

      <div className="composer-bar">
        <button
          type="button"
          className="ghost"
          onClick={() => transport.send({ t: paused ? "resume" : "pause" })}
        >
          {paused ? "resume" : "pause"}
        </button>

        {targetLabel && <span className="target">→ {targetLabel}</span>}

        <span className="spacer" />
        <span className="hint">enter send · shift+enter newline</span>
        <button type="submit" className="send" disabled={!text.trim()}>
          send
        </button>
      </div>
    </form>
  );
}
