import ReactMarkdown from 'react-markdown';
import type { Item } from '../state/store';

type Nano = Extract<Item, { k: 'nano' }>;

// 原语:Message(NAN 侧)—— 无头输出;流式期间纯文本+光标,完成后渲染 markdown

// Render edge only: turn a numeric timestamp into a human-readable format (the
// transport stays epoch), and a numeric duration (seconds) into a label.
const fmtTime = (ts: number) =>
  new Date(ts * 1000).toLocaleTimeString('zh-CN', { hour12: false });

const fmtDuration = (s: number) => `${s.toFixed(1)}s`;

export function NanoMessage({ item }: { item: Nano }) {
  const streaming = item.ts === undefined;

  return (
    <div className="row">
      <div className={`prose${streaming ? ' streaming' : ' md'}`}>
        {streaming ? (
          <>
            {item.text}
            <span className="caret" />
          </>
        ) : (
          <ReactMarkdown>{item.text}</ReactMarkdown>
        )}
      </div>
      {!streaming && (
        <div className="tsnote">{`✓ ${fmtTime(item.ts ?? 0)}${item.duration !== undefined ? ` · ${fmtDuration(item.duration)}` : ''}`}</div>
      )}
    </div>
  );
}

export function UserMessage({ item }: { item: Extract<Item, { k: 'user' }> }) {
  return (
    <div className="row user">
      <div className={`bubble${item.queued ? ' queued' : ''}`}>{item.text}</div>
    </div>
  );
}