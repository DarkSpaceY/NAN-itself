import { useEffect, useState } from 'react';
import type { Item } from '../state/store';
import type { RecordEntry, RecordStarted } from '../protocol';

const BRAILLE = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

type Rec = Extract<Item, { k: 'record' }>;

// Render edge: durations arrive as a number of seconds from the wire.
const fmtDuration = (s: number) => `${s.toFixed(1)}s`;

// Prose carried in the typed payload (the backend never pre-renders it):
// a subagent's delegation instruction / report body.
function payloadProse(s: RecordStarted): string | null {
  switch (s.category) {
    case 'subagent_spawn':
      return s.payload.task;
    case 'subagent_report':
      return s.payload.body;
    default:
      return null;
  }
}

// Primitive: Record — the uniform shape of every machine event.
export function RecordItem({ item }: { item: Rec }) {
  const [open, setOpen] = useState(item.open);
  const [spin, setSpin] = useState(0);
  const running = item.state === 'running';
  const prose = payloadProse(item.started);

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
        <span className="rname">{item.label}</span>
        <span className="note">{item.duration !== undefined ? fmtDuration(item.duration) : ''}</span>
        <span className="chev">▸</span>
      </div>
      {(item.detail.length > 0 || prose || item.error) && (
        <div className="rdetail">
          {item.detail.map((entry, i) => (
            <div key={i}>{renderEntry(entry)}</div>
          ))}
          {prose && <div>{prose}</div>}
          {item.error && (
            <div>
              <span className="e">
                {item.error.type}: {item.error.message}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// Entries are typed by the backend; render by `kind`, never by sniffing text.
function renderEntry(entry: RecordEntry) {
  switch (entry.kind) {
    case 'text':
      return entry.text;
    case 'item':
      return (
        <>
          · <span className="f">{entry.text}</span>
        </>
      );
    case 'field':
      return (
        <>
          <span className="k">{entry.label}</span> {entry.value}
        </>
      );
    case 'code':
      return <pre className="code">{entry.text}</pre>;
  }
}
