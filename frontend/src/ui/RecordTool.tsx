// NAN 的活动时间线：一个 step = 感知（module_query）→ 言语（正文）→ 行动（工具）。
// 这里负责感知与行动两端。数据来自 convert.ts 塞进 tool-call result 的结构化字段。
//
// 视觉语言：一条脊线 + 脊线上的点。没有图标芯片、没有容器底色。
//   ○ 空心环   感知（module_query）
//   ● 实心点   行动（invoke_tool / invoke_skill / spawn）
//   · 小圆点   轻量动作（list_* / show_* / sleep）
//   ◆ 菱形     收尾（finish）
//   进行中的点带呼吸光晕；失败的点变红。
// 「现在」最亮，已完成的行自然退后。
//
// 展开：行动类默认可展开；轻量动作仅在失败且有内容时可展开。
// 失败信息不藏在折叠里：收起时直接铺在行下（最多两行）。

import { memo, useState, type ReactNode } from "react";
import type {
  ToolCallMessagePartComponent,
  ToolCallMessagePartStatus,
} from "@assistant-ui/react";
import { ChevronRightIcon } from "lucide-react";

// ------------------------------------------------------------------
// 数据
// ------------------------------------------------------------------

interface RecordEntry {
  kind: string;
  label?: string;
  text?: string;
  value?: string;
}

interface RecordResult {
  state?: "running" | "done" | "failed" | "void";
  title?: string;
  entries?: RecordEntry[];
  error?: { type: string; message: string };
  durationS?: number;
}

type RunState = "running" | "done" | "failed";

/**
 * convert.ts 的约定：进行中的记录没有 result，完成后才带 result 出现。
 * 所以「没有 result」= 正在进行；有 result 则以 result.state 为准。
 * status 只用来识别被取消/出错、永远等不到 result 的记录。
 */
function runStateOf(result?: RecordResult, status?: ToolCallMessagePartStatus): RunState {
  if (!result) {
    if (status?.type === "incomplete") return status.reason === "error" ? "failed" : "done";
    return "running";
  }
  if (result.state === "failed") return "failed";
  if (result.state === "running") return "running";
  return "done";
}

/** ≥1s 才显示——无限运行的流里，毫秒级耗时只是噪音。 */
function formatDuration(s?: number): string | null {
  if (s == null || s < 1) return null;
  if (s < 10) return `${s.toFixed(1)}s`;
  if (s < 60) return `${Math.round(s)}s`;
  const m = Math.floor(s / 60);
  return `${m}m${String(Math.round(s % 60)).padStart(2, "0")}s`;
}

// ------------------------------------------------------------------
// 分类 → 呈现
// ------------------------------------------------------------------

type Shape = "ring" | "dot" | "pip" | "diamond";
type Weight = "sense" | "act" | "light";

interface CategoryStyle {
  /** 行首动词；title 跟在后面。 */
  verb: string;
  shape: Shape;
  weight: Weight;
  /** 有内容时是否可展开（轻量动作失败时另行升级）。 */
  expandable: boolean;
}

const CATEGORY_STYLES: Record<string, CategoryStyle> = {
  module_query: { verb: "观察", shape: "ring", weight: "sense", expandable: true },
  invoke_tool: { verb: "使用", shape: "dot", weight: "act", expandable: true },
  invoke_skill: { verb: "运用技能", shape: "dot", weight: "act", expandable: true },
  spawn: { verb: "派出", shape: "dot", weight: "act", expandable: true },
  list_tools: { verb: "盘点工具", shape: "pip", weight: "light", expandable: false },
  list_skills: { verb: "盘点技能", shape: "pip", weight: "light", expandable: false },
  list_channels: { verb: "盘点通道", shape: "pip", weight: "light", expandable: false },
  show_tool: { verb: "查看工具", shape: "pip", weight: "light", expandable: false },
  show_skill: { verb: "查看技能", shape: "pip", weight: "light", expandable: false },
  show_channels: { verb: "查看通道", shape: "pip", weight: "light", expandable: false },
  sleep: { verb: "休息", shape: "pip", weight: "light", expandable: false },
  finish: { verb: "收尾", shape: "diamond", weight: "light", expandable: false },
};

const FALLBACK_STYLE: CategoryStyle = {
  verb: "使用",
  shape: "dot",
  weight: "act",
  expandable: true,
};

const styleOf = (category: string): CategoryStyle =>
  CATEGORY_STYLES[category] ?? FALLBACK_STYLE;

// ------------------------------------------------------------------
// 原子
// ------------------------------------------------------------------

const STATE_LABEL: Record<RunState, string> = {
  running: "进行中",
  done: "",
  failed: "失败",
};

function Dot({ shape, state }: { shape: Shape; state: RunState }) {
  return <span className="nan-dot" data-shape={shape} data-state={state} aria-hidden />;
}

/** 折叠容器：grid-rows 0fr→1fr，收起时 visibility:hidden 让内容退出 Tab 序与读屏。 */
function Collapse({ open, children }: { open: boolean; children: ReactNode }) {
  return (
    <div className="nan-collapse" data-open={open} aria-hidden={!open}>
      <div>{children}</div>
    </div>
  );
}

interface RowProps {
  weight: Weight;
  shape: Shape;
  state: RunState;
  expandable: boolean;
  open: boolean;
  onToggle: () => void;
  meta?: ReactNode;
  children: ReactNode;
}

/** 所有时间线行的唯一骨架：点 · 标签 · 右侧元信息。 */
function Row({ weight, shape, state, expandable, open, onToggle, meta, children }: RowProps) {
  const inner = (
    <>
      <Dot shape={shape} state={state} />
      <span className="nan-row-label">
        {STATE_LABEL[state] && <span className="sr-only">{STATE_LABEL[state]}：</span>}
        {children}
      </span>
      {(meta || expandable) && (
        <span className="nan-row-meta">
          {meta}
          {expandable && <ChevronRightIcon className="nan-chevron" data-open={open} />}
        </span>
      )}
    </>
  );

  return expandable ? (
    <button
      type="button"
      className="nan-row"
      data-weight={weight}
      data-state={state}
      aria-expanded={open}
      onClick={onToggle}
    >
      {inner}
    </button>
  ) : (
    <div className="nan-row" data-weight={weight} data-state={state}>
      {inner}
    </div>
  );
}

function Entry({ label, tone, children }: { label?: string; tone?: "error"; children: ReactNode }) {
  return (
    <div className="nan-entry" data-tone={tone}>
      <span className="nan-entry-label">{label}</span>
      <span className="nan-entry-value">{children}</span>
    </div>
  );
}

function Detail({ open, result }: { open: boolean; result: RecordResult }) {
  return (
    <Collapse open={open}>
      <div className="nan-detail-body">
        {result.error && (
          <Entry label="出错" tone="error">
            {result.error.message}
          </Entry>
        )}
        {result.entries?.map((e, i) =>
          e.kind === "text" ? (
            <pre key={i} className="nan-entry-text">
              {e.text}
            </pre>
          ) : (
            <Entry key={i} label={e.label}>
              {e.value}
            </Entry>
          ),
        )}
      </div>
    </Collapse>
  );
}

function InlineError({ message }: { message: string }) {
  return <div className="nan-inline-error">{message}</div>;
}

// ------------------------------------------------------------------
// 行动行
// ------------------------------------------------------------------

function ActivityRow({
  toolName,
  result,
  status,
}: {
  toolName: string;
  result?: RecordResult;
  status?: ToolCallMessagePartStatus;
}) {
  const r = result ?? {};
  const state = runStateOf(result, status);
  const cat = styleOf(toolName);
  const [open, setOpen] = useState(false);

  // 已知类别没有 title 时只显示动词；未知类别回退显示原始 toolName
  const title = r.title ?? (toolName in CATEGORY_STYLES ? "" : toolName);
  const hasBody = Boolean(r.entries?.length || r.error);
  const expandable = hasBody && (cat.expandable || state === "failed");
  // 失败信息始终可见：收起、或根本不可展开时，直接铺在行下
  const showInlineError = Boolean(r.error) && (!expandable || !open);

  return (
    <div data-tool-category={toolName}>
      <Row
        weight={cat.weight}
        shape={cat.shape}
        state={state}
        expandable={expandable}
        open={open}
        onToggle={() => setOpen((v) => !v)}
        meta={state === "running" ? null : formatDuration(r.durationS)}
      >
        <span className="nan-verb">{cat.verb}</span>
        {title && title !== cat.verb && <span className="nan-title">{title}</span>}
      </Row>
      {showInlineError && r.error && <InlineError message={r.error.message} />}
      {expandable && <Detail open={open} result={r} />}
    </div>
  );
}

// ------------------------------------------------------------------
// 感知行：module_query 不是「动作」，是「看见」。
// 同一副骨架，空心环 + 更轻的字重；最多内联两个字段，其余展开看。
// ------------------------------------------------------------------

function QueryRow({
  toolName,
  result,
  status,
}: {
  toolName: string;
  result?: RecordResult;
  status?: ToolCallMessagePartStatus;
}) {
  const r = result ?? {};
  const state = runStateOf(result, status);
  const [open, setOpen] = useState(false);

  // 失败的观察回到普通行语义：得能看到错误
  if (state === "failed") return <ActivityRow toolName={toolName} result={r} status={status} />;

  const entries = r.entries ?? [];
  const inline = entries.filter((e) => e.kind === "field").slice(0, 2);
  const more = entries.filter((e) => e.kind === "field").length - inline.length;
  const expandable = entries.length > inline.length;

  return (
    <div data-tool-category={toolName}>
      <Row
        weight="sense"
        shape="ring"
        state={state}
        expandable={expandable}
        open={open}
        onToggle={() => setOpen((v) => !v)}
        meta={state === "running" ? null : formatDuration(r.durationS)}
      >
        <span className="nan-title">{r.title || toolName}</span>
        {inline.map((f, i) => (
          <span key={i} className="nan-field">
            {f.label}
            <b>{f.value}</b>
          </span>
        ))}
        {more > 0 && <span className="nan-field">+{more}</span>}
      </Row>
      {expandable && <Detail open={open} result={r} />}
    </div>
  );
}

// ------------------------------------------------------------------
// 入口
// ------------------------------------------------------------------

function RecordLine({
  toolName,
  result,
  status,
}: {
  toolName: string;
  result?: RecordResult;
  status?: ToolCallMessagePartStatus;
}) {
  return toolName === "module_query" ? (
    <QueryRow toolName={toolName} result={result} status={status} />
  ) : (
    <ActivityRow toolName={toolName} result={result} status={status} />
  );
}

const RecordToolImpl: ToolCallMessagePartComponent = (props) => {
  // __group__ / __sense__ 是 convert.ts 合成的折叠 part，不是真实工具名
  if (props.toolName === "__group__") {
    return <RecordGroupImpl result={props.result} toolCallId={props.toolCallId} />;
  }
  if (props.toolName === "__sense__") {
    return <RecordSenseGroupImpl result={props.result} />;
  }
  return (
    <RecordLine
      toolName={props.toolName}
      result={props.result as RecordResult | undefined}
      status={props.status}
    />
  );
};

export const RecordTool = memo(RecordToolImpl);

// ------------------------------------------------------------------
// 动作组：连续的动作折叠成一行摘要，点开是完整时间线。
// 运行中默认展开（看得到实时进展）；用户点过就以用户为准。
// 子行与摘要行共用同一条脊线——组是时间线的一段，不是嵌套的盒子。
// ------------------------------------------------------------------

interface GroupResult {
  turnOpen?: boolean;
  records?: Array<{ toolName: string; result?: RecordResult }>;
}

/**
 * 用户的展开/收起选择，按组 id 存在模块级：消息部件序列变化会让
 * aui 重建 part 组件（useState 被重置），用户意图不能跟着丢。
 * 页面刷新即清空——这正是「回到默认行为」的自然语义。
 */
const userToggled = new Map<string, boolean>();

const RecordGroupImpl = ({
  result,
  toolCallId = "group",
}: {
  result?: unknown;
  toolCallId?: string;
}) => {
  const r = ((result ?? {}) as GroupResult);
  const records = r.records ?? [];
  const [toggled, setToggled] = useState<boolean | null>(
    () => userToggled.get(toolCallId) ?? null,
  );

  const toggle = () => {
    const open = toggled ?? r.turnOpen === true;
    const next = !open;
    userToggled.set(toolCallId, next);
    setToggled(next);
  };

  // 展开跟随轮生命周期（convert 塞进 result 的 turnOpen），不跟着
  // 单条 record 的完成抖动；用户点过就以用户为准（跨重挂载生效）。
  const open = toggled ?? r.turnOpen === true;
  const running = records.some((x) => !x.result || x.result.state === "running");
  const failedCount = records.filter((r) => r.result?.state === "failed").length;
  const durationS = records.reduce((a, r) => a + (r.result?.durationS ?? 0), 0);

  // 收起时仍要让人知道「现在在做什么」：露出最新一条进行中的标题
  let liveTitle = "";
  for (let i = records.length - 1; i >= 0; i--) {
    const rec = records[i];
    if (!rec.result || rec.result.state === "running") {
      liveTitle = rec.result?.title ?? styleOf(rec.toolName).verb;
      break;
    }
  }

  const state: RunState = running ? "running" : failedCount > 0 ? "failed" : "done";

  return (
    <div className="nan-activity-group" data-open={open}>
      <Row
        weight="act"
        shape="dot"
        state={state}
        expandable
        open={open}
        onToggle={toggle}
        meta={running ? null : formatDuration(durationS)}
      >
        <span className="nan-title">{records.length} 个动作</span>
        {failedCount > 0 && (
          <span className="nan-field" data-tone="error">
            {failedCount} 个失败
          </span>
        )}
        {!open && liveTitle && <span className="nan-field">{liveTitle}</span>}
      </Row>
      <Collapse open={open}>
        <div className="nan-detail-body" data-flush>
          {records.map((r, i) => (
            <RecordLine key={i} toolName={r.toolName} result={r.result} />
          ))}
        </div>
      </Collapse>
    </div>
  );
};

export const RecordGroup = memo(RecordGroupImpl);

// ------------------------------------------------------------------
// 感知组：一个 step 开头往往连续读取多个模块（module_query）。
// 折叠成一行「感知 记忆 收件箱 时间」——模块名就是摘要；点开才是各模块的字段。
// 默认收起；有模块失败时默认展开，错误不能被折叠藏起来。
// ------------------------------------------------------------------

const RecordSenseGroupImpl = ({ result }: { result?: unknown }) => {
  const records = ((result ?? {}) as GroupResult).records ?? [];
  const [toggled, setToggled] = useState<boolean | null>(null);

  const running = records.some((r) => !r.result);
  const failedCount = records.filter((r) => r.result?.state === "failed").length;
  const durationS = records.reduce((a, r) => a + (r.result?.durationS ?? 0), 0);
  // 感知组默认收起（摘要行本身就是内容）；有失败时展开，错误不能被藏住
  const open = toggled ?? failedCount > 0;
  const state: RunState = running ? "running" : failedCount > 0 ? "failed" : "done";

  return (
    <div className="nan-activity-group" data-open={open}>
      <Row
        weight="sense"
        shape="ring"
        state={state}
        expandable
        open={open}
        onToggle={() => setToggled(!open)}
        meta={running ? null : formatDuration(durationS)}
      >
        <span className="nan-title">感知</span>
        {records.map((r, i) => (
          <span key={i} className="nan-field">
            {r.result?.title ?? r.toolName}
          </span>
        ))}
        {failedCount > 0 && (
          <span className="nan-field" data-tone="error">
            {failedCount} 个失败
          </span>
        )}
      </Row>
      <Collapse open={open}>
        <div className="nan-detail-body" data-flush>
          {records.map((r, i) => (
            <RecordLine key={i} toolName={r.toolName} result={r.result} />
          ))}
        </div>
      </Collapse>
    </div>
  );
};

export const RecordSenseGroup = memo(RecordSenseGroupImpl);