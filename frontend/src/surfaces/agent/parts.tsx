// Part rendering: text goes through markdown, verbs collapse into a
// one-line activity, input echoes get their own block.
//
// Recognizes part shapes only, not events; the event-to-part folding
// happens in the mapper.

import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import remarkGfm from "remark-gfm";

import type {
  InputPart,
  Part,
  TextPart,
  ThinkingPart,
  VerbPart,
} from "../../kernel/history/parts";

function truncate(value: string, max: number): string {
  return value.length > max ? `${value.slice(0, max)}…` : value;
}

function argSummary(args: Record<string, unknown>): string {
  const first = Object.entries(args)[0];
  if (!first) return "";
  const [key, raw] = first;
  const value = typeof raw === "string" ? raw : JSON.stringify(raw);
  return `${key}: ${truncate(value ?? "null", 96)}`;
}

function argValue(value: unknown): string {
  if (value === null) return "null";
  if (Array.isArray(value)) {
    const nested = value.some((item) => item !== null && typeof item === "object");
    return nested
      ? JSON.stringify(value, null, 2)
      : value.map((item) => String(item)).join("\n");
  }
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    const nested = entries.some(([, item]) => item !== null && typeof item === "object");
    return nested
      ? JSON.stringify(value, null, 2)
      : entries.map(([key, item]) => `${key}=${String(item)}`).join("\n");
  }
  return String(value);
}

/** The full arguments behind the one-line summary, never truncated. */
function VerbArgs({
  verb,
  args,
}: {
  verb: string;
  args: Record<string, unknown>;
}) {
  const entries = Object.entries(args);
  if (entries.length === 0) return null;

  const command =
    verb === "shell" && typeof args.command === "string" ? args.command : null;
  const rest =
    command === null ? entries : entries.filter(([key]) => key !== "command");

  return (
    <div className="verb-input">
      {command !== null && <pre className="verb-arg-command">{command}</pre>}
      {rest.map(([key, value]) => (
        <div className="verb-arg" key={key}>
          <span className="verb-arg-key">{key}</span>
          <span className="verb-arg-value">{argValue(value)}</span>
        </div>
      ))}
    </div>
  );
}

function Text({ part }: { part: TextPart }) {
  return (
    <div className={`text-part${part.streaming ? " streaming" : ""}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
      >
        {part.text}
      </ReactMarkdown>
    </div>
  );
}

function Input({ part }: { part: InputPart }) {
  return (
    <div className="input-part">
      <div className="input-text">{part.text}</div>
    </div>
  );
}

function Thinking({ part }: { part: ThinkingPart }) {
  return (
    <details
      className={`thinking-part${part.streaming ? " streaming" : ""}`}
      open={part.streaming}
    >
      <summary>
        <span className="caret">›</span>
        <span className="thinking-label">thinking</span>
        {part.streaming && <span className="thinking-state">…</span>}
      </summary>
      <div className="thinking-body">{part.text}</div>
    </details>
  );
}

function firstLine(value: unknown, max = 120): string {
  if (typeof value !== "string") return "";
  const line = value.trim().split("\n")[0] ?? "";
  return truncate(line, max);
}

/** One line per verb: the argument the reader actually needs. */
function verbSummary(part: VerbPart): { text: string; tags: string[] } {
  const depth = part.detail?.depth;

  switch (part.verb) {
    case "shell": {
      const command =
        typeof part.args.command === "string" ? part.args.command : "";
      const lines = command.trim().split("\n").filter(Boolean);
      return {
        text: `$ ${truncate(lines[0] ?? "", 120)}`,
        tags: lines.length > 1 ? [`${lines.length} 行`] : [],
      };
    }
    case "spawn":
      return {
        text: firstLine(part.args.task),
        tags: typeof depth === "number" ? [`depth ${depth}`] : [],
      };
    case "send": {
      const down = part.args.direction === "down";
      const who =
        typeof part.args.child_id === "string"
          ? part.args.child_id
          : down
            ? "child"
            : "parent";
      return { text: `${down ? "↓" : "↑"} ${who}`, tags: [] };
    }
    case "kill":
      return { text: String(part.args.child_id ?? ""), tags: [] };
    case "sleep":
      return { text: `${String(part.args.seconds ?? "")}s`, tags: [] };
    case "finish":
      return { text: firstLine(part.args.report), tags: [] };
    default:
      return { text: argSummary(part.args), tags: [] };
  }
}

function ShellBody({ part }: { part: VerbPart }) {
  const command =
    typeof part.args.command === "string" ? part.args.command : "";
  const exit =
    typeof part.detail?.exit_code === "number" ? part.detail.exit_code : null;
  const job =
    typeof part.detail?.job_id === "string" ? part.detail.job_id : null;
  const extras = [
    typeof part.args.cwd === "string" ? `cwd ${part.args.cwd}` : null,
    part.args.background === true ? "background" : null,
  ].filter((item): item is string => item !== null);
  const foot = [
    exit !== null ? `exit ${exit}` : null,
    part.durationS !== undefined ? `${part.durationS.toFixed(2)}s` : null,
    job ? `job ${job}` : null,
  ].filter((item): item is string => item !== null);

  return (
    <div className="verb-term">
      <pre className="verb-term-cmd">$ {command}</pre>
      {extras.length > 0 && (
        <div className="verb-term-meta">{extras.join(" · ")}</div>
      )}
      {part.error ? (
        <pre className="verb-term-out error">
          {part.error.type}: {part.error.message}
        </pre>
      ) : (
        part.text && <pre className="verb-term-out">{truncate(part.text, 4000)}</pre>
      )}
      {foot.length > 0 && <div className="verb-term-foot">{foot.join(" · ")}</div>}
    </div>
  );
}

function SpawnBody({
  part,
  onDrill,
}: {
  part: VerbPart;
  onDrill?: (agentId: string) => void;
}) {
  const task = typeof part.args.task === "string" ? part.args.task : "";
  const childId =
    typeof part.detail?.child_id === "string" ? part.detail.child_id : null;
  const depth = part.detail?.depth;

  return (
    <>
      {task && <p className="verb-quote">{task}</p>}
      <div className="verb-child">
        <span className="verb-dot" data-state={part.state} />
        <span className="verb-child-id">{childId ?? "子代理"}</span>
        {typeof depth === "number" && (
          <span className="verb-meta">depth {depth}</span>
        )}
        {childId && onDrill && (
          <button type="button" className="drill" onClick={() => onDrill(childId)}>
            open →
          </button>
        )}
      </div>
    </>
  );
}

function SendBody({ part }: { part: VerbPart }) {
  const text = typeof part.args.text === "string" ? part.args.text : "";
  return text ? <blockquote className="verb-quote">{text}</blockquote> : null;
}

function SleepBody({ part }: { part: VerbPart }) {
  const seconds =
    typeof part.args.seconds === "number" ? part.args.seconds : null;
  const running = part.state === "running";

  return (
    <div className="verb-sleep">
      <div className="verb-track">
        <div
          className="verb-fill"
          data-state={part.state}
          style={
            running && seconds !== null
              ? { animationDuration: `${seconds}s` }
              : undefined
          }
        />
      </div>
      <span className="verb-meta">
        {running ? `计时中 · ${seconds ?? "?"}s` : (part.text ?? "已等待")}
      </span>
    </div>
  );
}

function FinishBody({ part }: { part: VerbPart }) {
  const report = typeof part.args.report === "string" ? part.args.report : "";
  if (!report) return null;

  return (
    <div className="verb-report">
      <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
        {report}
      </ReactMarkdown>
    </div>
  );
}

function GenericBody({ part }: { part: VerbPart }) {
  return (
    <>
      <VerbArgs verb={part.verb} args={part.args} />
      {part.text && <pre className="verb-body">{truncate(part.text, 4000)}</pre>}
      {part.detail && part.detail.refused !== true && (
        <pre className="verb-body detail">
          {JSON.stringify(part.detail, null, 2)}
        </pre>
      )}
    </>
  );
}

function VerbBody({
  part,
  onDrill,
}: {
  part: VerbPart;
  onDrill?: (agentId: string) => void;
}) {
  if (part.error && part.verb !== "shell") {
    const { type, message } = part.error;
    return (
      <pre className="verb-body error">
        {type}: {message}
      </pre>
    );
  }

  switch (part.verb) {
    case "shell":
      return <ShellBody part={part} />;
    case "spawn":
      return <SpawnBody part={part} onDrill={onDrill} />;
    case "send":
      return <SendBody part={part} />;
    case "kill":
      return <p className="verb-note">{part.text ?? ""}</p>;
    case "sleep":
      return <SleepBody part={part} />;
    case "finish":
      return <FinishBody part={part} />;
    default:
      return <GenericBody part={part} />;
  }
}

function Verb({
  part,
  onDrill,
}: {
  part: VerbPart;
  onDrill?: (agentId: string) => void;
}) {
  const summary = verbSummary(part);

  return (
    <details className="verb-part" data-state={part.state}>
      <summary>
        <span className="caret">›</span>
        <span className="verb-name">{part.verb}</span>
        <span className="verb-args">{summary.text}</span>
        {summary.tags.map((tag) => (
          <span className="verb-tag" key={tag}>
            {tag}
          </span>
        ))}
        {part.state === "running" && (
          <span className="verb-state" data-kind="running">
            running
          </span>
        )}
        {part.state === "failed" && (
          <span className="verb-state" data-kind="failed">
            failed
          </span>
        )}
        {part.state === "refused" && (
          <span className="verb-state" data-kind="refused">
            refused
          </span>
        )}
        {part.durationS !== undefined && (
          <span className="verb-duration">{part.durationS.toFixed(2)}s</span>
        )}
      </summary>

      <VerbBody part={part} onDrill={onDrill} />
    </details>
  );
}

export function PartView({
  part,
  onDrill,
}: {
  part: Part;
  onDrill?: (agentId: string) => void;
}) {
  switch (part.kind) {
    case "text":
      return <Text part={part} />;
    case "thinking":
      return <Thinking part={part} />;
    case "input":
      return <Input part={part} />;
    case "verb":
      return <Verb part={part} onDrill={onDrill} />;
  }
}
