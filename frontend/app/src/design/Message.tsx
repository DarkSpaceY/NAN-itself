import type { Item } from '../state/store';

// 原语:Message(人话)—— 用户气泡 / NAN 裸段输出
export function UserMessage({ item }: { item: Extract<Item, { k: 'user' }> }) {
  return (
    <div className="row user">
      <div className={`bubble${item.queued ? ' queued' : ''}`}>{item.text}</div>
    </div>
  );
}

export function NanoMessage({ item }: { item: Extract<Item, { k: 'nano' }> }) {
  const streaming = item.ts === undefined;
  return (
    <div className="row">
      <div className="prose">
        {item.text}
        {streaming && <span className="caret" />}
      </div>
      {!streaming && <div className="tsnote">{`✓ ${item.ts}${item.duration ? ` · ${item.duration}` : ''}`}</div>}
    </div>
  );
}