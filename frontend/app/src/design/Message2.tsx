import ReactMarkdown from 'react-markdown';
import type { Item } from '../state/store';

type Nano = Extract<Item, { k: 'nano' }>;

// 原语:Message(NAN 侧)—— 无头输出;流式期间纯文本+光标,完成后渲染 markdown

// 渲染边缘才把数值时间戳变成人类可读格式(协议契约:传输层一律 epoch)
const fmtTime = (ts: number) =>
  new Date(ts * 1000).toLocaleTimeString('zh-CN', { hour12: false });

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
        <div className="tsnote">{`✓ ${fmtTime(item.ts ?? 0)}${item.duration ? ` · ${item.duration}` : ''}`}</div>
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