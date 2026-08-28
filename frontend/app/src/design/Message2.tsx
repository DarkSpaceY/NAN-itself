import ReactMarkdown from 'react-markdown';
import { useEffect, useRef, useState } from 'react';
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

export function UserMessage({ item }: Extract<Item, { k: 'user' }> extends never ? never : { item: Extract<Item, { k: 'user' }> }) {
  return <UserMessageInner item={item} />;
}

function UserMessageInner({ item }: { item: Extract<Item, { k: 'user' }> }) {
  const ref = useRef<HTMLDivElement>(null);
  const first = useRef(true);

  // 入场轻微高亮一次
  useEffect(() => {
    if (!first.current) return;
    first.current = false;
    ref.current?.classList.add('fresh');
    const t = setTimeout(() => ref.current?.classList.remove('fresh'), 1200);
    return () => clearTimeout(t);
  }, []);

  return (
    <div className="row user">
      <div ref={ref} className="bubble">
        {item.text}
      </div>
    </div>
  );
}