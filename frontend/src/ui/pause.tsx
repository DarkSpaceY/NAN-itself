// 暂停控制：状态在 App（store 驱动），按钮在 Thread 的 composer 操作行渲染。
//
// 三态，而不是两态——因为暂停是协作式的，点击到后端确认之间有一段真实存在的等待：
//   running  → 点击 = 请求暂停
//   pending  → 已请求，等当前步骤结束；点击 = 撤回请求
//   paused   → 后端已确认；点击 = 继续

import { createContext, useContext } from "react";
import { LoaderIcon, PauseIcon, PlayIcon } from "lucide-react";
import { cn } from "@/lib/utils";

export type PauseState = "running" | "pending" | "paused";

export interface PauseControl {
  state: PauseState;
  /** 已确认的暂停态（后端 status 回传）。与 state === "paused" 等价，保留以兼容旧调用。 */
  paused: boolean;
  /** 已请求暂停、等待后端确认。与 state === "pending" 等价，保留以兼容旧调用。 */
  pending: boolean;
  toggle: () => void;
}

export const PauseControlContext = createContext<PauseControl | null>(null);

export function usePauseControl(): PauseControl | null {
  return useContext(PauseControlContext);
}

const COPY: Record<PauseState, { label: string; hint: string }> = {
  running: { label: "暂停", hint: "当前步骤结束后暂停" },
  pending: { label: "暂停中", hint: "正在等待当前步骤结束，点击取消" },
  paused: { label: "继续", hint: "继续运行" },
};

/**
 * 直接放进 composer 操作行即可：<PauseButton />
 * 无 Provider 时渲染 null，不会在孤立预览里报错。
 */
export function PauseButton({ className }: { className?: string }) {
  const ctl = usePauseControl();
  if (!ctl) return null;

  const { label, hint } = COPY[ctl.state];
  const Icon = ctl.state === "paused" ? PlayIcon : PauseIcon;

  return (
    <button
      type="button"
      onClick={ctl.toggle}
      title={hint}
      aria-label={hint}
      aria-busy={ctl.state === "pending"}
      data-state={ctl.state}
      className={cn("nan-pause", className)}
    >
      {ctl.state === "pending" ? (
        <LoaderIcon className="size-3.5 animate-spin [animation-duration:0.9s]" />
      ) : (
        <Icon className="size-3.5 fill-current" />
      )}
      <span>{label}</span>
    </button>
  );
}