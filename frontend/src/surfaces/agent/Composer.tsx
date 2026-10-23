// 输入框：协议没有寻址输入，发出的永远进 root。
//
// 钻到子 agent 时只在旁边标注目标，输入路径不变。

import { useState } from "react";

import type { InputPart } from "../../kernel/history/parts";
import type { Transport } from "../../transport/ws";

interface ComposerProps {
  transport: Transport;
  paused: boolean;
  /** 非 root 焦点时的提示文本，例如 "root · viewing a3"。 */
  targetLabel: string | null;
  /** 本轮还在跑时发出的消息：排队等轮边界，不进时间线。 */
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
