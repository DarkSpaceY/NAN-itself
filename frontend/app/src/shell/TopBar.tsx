import type { Status } from '../protocol';
import { UI_VERSION, type ConnState } from '../state/useAgentStream2';

// 骨架:Topbar = 循环状态机 + 连接可见性
export function TopBar({
  status,
  model,
  baseUrl,
  conn,
}: {
  status: Status;
  model: string;
  baseUrl: string;
  conn: ConnState;
}) {
  return (
    <div className="topbar">
      <span className="brand">NAN</span>
      <Pulse state={status.state} />
      <StatusText status={status} />
      {conn === 'reconnecting' && (
        <span className="status bad">⚠ 连接断开,重连中…</span>
      )}
      <span className="sp" />
      <span className="env">
        {model}
        {baseUrl ? ` · ${baseUrl.replace(/^https?:\/\//, '')}` : ''} · ui@{UI_VERSION}
      </span>
    </div>
  );
}

export function Pulse({ state }: { state: Status['state'] }) {
  return <span className={`dot${state === 'working' ? ' working' : state === 'error' ? ' err' : ''}`} />;
}

export function StatusText({ status }: { status: Status }) {
  if (status.state === 'working') {
    return <span className="status"><span className="run">工作中</span></span>;
  }
  if (status.state === 'error') {
    return <span className="status"><span className="bad">异常</span></span>;
  }
  return <span className="status"><span className="idle">空闲</span></span>;
}