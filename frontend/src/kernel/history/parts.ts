// 一个 agent 的时间线模型。
//
// 与「一问一答」不同：agent 连续跑 turn，每条 assistant 输出是一段
// 流，verb 调用是同一段流里的活动条目。轮以 turn_started 为锚点，
// 用户的输入（user_input 回执）开启该轮。

export type VerbState = "running" | "done" | "failed";

/** 模型输出的可见文本；同一 turn 内多次输出合并为一段流。 */
export interface TextPart {
  kind: "text";
  id: string;
  text: string;
  /** 流未结束（output_done 之前）。 */
  streaming?: boolean;
}

/**
 * 用户输入回执。只出现在 root 的时间线里。
 *
 * echo     回执刚发出，还没有轮开始跑（空闲时发送，通常一闪而过）
 * queued   本轮还在跑的时候插话：先落在当前轮里，等下一个轮边界才被取走
 * consumed 已被某一轮取走
 */
export type InputStatus = "echo" | "queued" | "consumed";

export interface InputPart {
  kind: "input";
  id: string;
  text: string;
  mid?: string;
  status: InputStatus;
}

/** 一次 verb 调用。text 是模型收到的原文，detail 是给界面的摘要。 */
export interface VerbPart {
  kind: "verb";
  id: string;
  verb: string;
  args: Record<string, unknown>;
  state: VerbState;
  text?: string;
  detail?: Record<string, unknown> | null;
  error?: { type: string; message: string };
  durationS?: number;
}

export type Part = TextPart | InputPart | VerbPart;

export interface Round {
  /** 后端 turn_index；在 turn_started 之前创建的轮先用本地计数。 */
  index: number;
  ts: number;
  /** 该轮最后应用的事件 seq，持久化后用于跳过重放。 */
  appliedSeq: number;
  closed: boolean;
  /** 后端对整轮的处置；null 表示被 agent_exited 直接截断。 */
  outcome: "done" | "failed" | "cancelled" | null;
  /** 失败原因或重排队的输入数；折叠行里显示。 */
  detail: string | null;
  parts: Part[];
}
