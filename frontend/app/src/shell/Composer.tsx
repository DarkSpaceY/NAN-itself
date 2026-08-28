import { type KeyboardEvent, useRef } from 'react';

// 原语:Composer —— 底部固定输入
export function Composer({
  onSend,
  seq,
  nextHop,
}: {
  onSend: (text: string) => void;
  seq: number;
  nextHop?: string;
}) {
  const ref = useRef<HTMLInputElement>(null);

  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key !== 'Enter' || e.shiftKey) return;
    e.preventDefault();
    const v = ref.current?.value.trim();
    if (!v) return;
    if (ref.current) ref.current.value = '';
    onSend(v);
  };

  return (
    <div className="composer">
      <div className="cp-in">
        <span className="ps">❯</span>
        <input
          ref={ref}
          placeholder="输入消息…"
          autoComplete="off"
          onKeyDown={onKey}
          aria-label="输入消息"
        />
        <kbd>⏎</kbd>
        <span className="cp-status">
          {seq > 0 ? `seq ${seq}${nextHop ? ` · 下一跳 ${nextHop}` : ''}` : ''}
        </span>
      </div>
    </div>
  );
}