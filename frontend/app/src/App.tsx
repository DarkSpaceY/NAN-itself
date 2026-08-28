import { memo, useEffect, useRef } from 'react';
import { useAgentStream } from './state/useAgentStream2';
import { TopBar } from './shell/TopBar';
import { Composer } from './shell/Composer';
import { RecordItem } from './design/RecordItem';
import { UserMessage } from './design/Message';
import { NanoMessage } from './design/Message2';
import type { Item } from './state/store';

// memo:流式增量每秒触发几十次 fold,未受影响的行靠引用相等跳过重渲染
const Row = memo(function Row({ item }: { item: Item }) {
  switch (item.k) {
    case 'divider':
      return <div className="divider">{item.label}</div>;
    case 'user':
      return <UserMessage item={item} />;
    case 'nano':
      return <NanoMessage item={item} />;
    case 'record':
      return <RecordItem item={item} />;
  }
});

export default function App() {
  const { snap, send, conn } = useAgentStream();
  const mainRef = useRef<HTMLElement>(null);
  const pinnedRef = useRef(true);

  // 吸底:仅当用户本就停留在底部附近时才自动滚底;
  // 一旦上滚离开底部,流式增量不再拽动视口,回到底部即恢复。
  const onScroll = () => {
    const el = mainRef.current;
    if (!el) return;
    pinnedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  };

  useEffect(() => {
    if (mainRef.current && pinnedRef.current) {
      mainRef.current.scrollTop = mainRef.current.scrollHeight;
    }
  }, [snap.items.length, snap.items[snap.items.length - 1]]);

  return (
    <>
      <TopBar status={snap.status} model={snap.model} baseUrl={snap.baseUrl} conn={conn} />
      <main ref={mainRef} onScroll={onScroll}>
        <div className="col">
          {snap.items.map((it) => (
            <Row key={it.key} item={it} />
          ))}
        </div>
      </main>
      <Composer onSend={send} seq={snap.seq} nextHop={snap.status?.next_hop} />
    </>
  );
}