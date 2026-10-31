// Timeline model for one agent.
//
// Unlike a one-question-one-answer chat: an agent runs turns back to
// back, each assistant output is a stream, and verb calls are activity
// entries within the same stream. A round anchors on turn_started and
// is opened by the user's input (the user_input echo).

export type VerbState = "running" | "done" | "failed";

/** Visible text of model output; multiple outputs within one turn merge into one stream. */
export interface TextPart {
  kind: "text";
  id: string;
  text: string;
  /** Stream not finished (before output_done). */
  streaming?: boolean;
}

/** Model reasoning streamed before the answer; rendered dim and collapsible. */
export interface ThinkingPart {
  kind: "thinking";
  id: string;
  text: string;
  /** Stream not finished (before thinking_done). */
  streaming?: boolean;
}

/**
 * User input echo. Appears only in the root's timeline.
 *
 * echo     just sent, no round running yet (sent while idle, usually fleeting)
 * queued   interjection while the round is still running: lands in the
 *          current round, taken away at the next round boundary
 * consumed taken by some round
 */
export type InputStatus = "echo" | "queued" | "consumed";

export interface InputPart {
  kind: "input";
  id: string;
  text: string;
  mid?: string;
  status: InputStatus;
}

/** One verb call. text is the raw text the model received; detail is the summary for the UI. */
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

export type Part = TextPart | ThinkingPart | InputPart | VerbPart;

export interface Round {
  /** Backend turn_index; rounds created before turn_started use a local counter first. */
  index: number;
  ts: number;
  /** seq of the last event applied to this round; used to skip replay after persistence. */
  appliedSeq: number;
  closed: boolean;
  /** The backend's disposition of the whole round; null means truncated outright by agent_exited. */
  outcome: "done" | "failed" | "cancelled" | null;
  /** Failure reason or requeued input count; shown in the collapsed line. */
  detail: string | null;
  parts: Part[];
}
