import { useEffect, useState } from 'react';
import type { Item } from '../state/store';

const BRAILLE = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

type Rec = Extract<Item, { k: 'record' }>;

// 原语:Record —— 一切机器事件的统一形态
export function RecordItem({ item }: { item: Rec }) {
  const [open, setOpen] = useState(item.open);
  const [spin, setSpin] = useState(0);
  const running = item.state === 'running';

  useEffect(() => {
    setOpen(item.open);
  }, [item.open, item.state]);

  useEffect(() => {
    if (!running) return;
    let i = 0;
    const t = setInterval(() => setSpin(i++), 110);
    return () => clearInterval(t);
  }, [running]);

  return (
    <div className={`rec ${item.state}${open ? ' open' : ''}`} onClick={() => setOpen(!open)}>
      <div className="rrow">
        <span className="glyph">{running ? BRAILLE[spin % BRAILLE.length] : item.glyph}</span>
        <span className="rname">{item.name}</span>
        {item.summary ? <span className="rsum">{highlight(item.summary)}</span> : null}
        <span className="note">{item.note ?? ''}</span>
        <span className="chev">▸</span>
      </div>
      {item.detail.length > 0 && (
        <div className="rdetail">
          {item.detail.map((line, i) => (
            <div key={i}>
              {line.startsWith('✗') ? (
                <span className="e">{line}</span>
              ) : line.startsWith('· ') ? (
                <>
                  · <span className="f">{line.slice(2)}</span>
                </>
              ) : (
                highlight(line)
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// 数字/耗时轻量着色:纯文本切分,无注入
function highlight(text: string) {
  const parts = text.split(/(\+\d+|\d+ files|\d+\.\d+s|\d+m\d+s|\d+s\b)/g);
  return parts.map((p, i) =>
    /^\+?\d/.test(p) && p.length <= 12 ? (
      <b key={i} className="hl">
        {p}
      </b>
    ) : (
      p
    ),
  );
}
