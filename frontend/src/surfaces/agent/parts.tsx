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
  if (Array.isArray(value)) return value.map((item) => String(item)).join("\n");
  if (typeof value === "object") {
    return Object.entries(value as Record<string, unknown>)
      .map(([key, item]) => `${key}=${String(item)}`)
      .join("\n");
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

function Verb({
  part,
  onDrill,
}: {
  part: VerbPart;
  onDrill?: (agentId: string) => void;
}) {
  const childId =
    typeof part.detail?.child_id === "string" ? part.detail.child_id : null;

  return (
    <details className="verb-part" data-state={part.state}>
      <summary>
        <span className="caret">›</span>
        <span className="verb-name">{part.verb}</span>
        <span className="verb-args">{argSummary(part.args)}</span>
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

      {childId && onDrill && (
        <button
          type="button"
          className="drill"
          onClick={() => onDrill(childId)}
        >
          open {childId} →
        </button>
      )}

      <VerbArgs verb={part.verb} args={part.args} />

      {part.error ? (
        <pre className="verb-body error">
          {part.error.type}: {part.error.message}
        </pre>
      ) : (
        <>
          {part.text && (
            <pre className="verb-body">{truncate(part.text, 4000)}</pre>
          )}
          {part.detail && part.detail.refused !== true && (
            <pre className="verb-body detail">
              {JSON.stringify(part.detail, null, 2)}
            </pre>
          )}
        </>
      )}
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
