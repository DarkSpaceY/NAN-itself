import ReactMarkdown from 'react-markdown';
import type { Item } from '../state/store';

type Nano = Extract<Item, { k: 'nano' }>;

// 原语:Message(NAN 侧)—— 无头输出;流式期间纯文本+光标,完成后渲染 markdown
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
        <div className="tsnote">{`✓ ${item.ts}${item.duration ? ` · ${item.duration}` : ''}`}</div>
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