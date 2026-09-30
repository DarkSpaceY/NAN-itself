// 助手回复的 markdown 渲染。历史流与舞台面共用同一套插件与高亮器，
// 保证同一段文字在两处长得一样。
//
// 两个入口：
// - MarkdownText：历史流里的 text part，走 assistant-ui 的 primitive
//   （它自己取当前 part 的文本与状态，并按 data-status 驱动流式光标）
// - Markdown：舞台面的文本（那里没有 part 上下文，直接喂字符串）

import {
  isValidElement,
  useEffect,
  useMemo,
  useState,
  type ComponentPropsWithoutRef,
  type ReactElement,
  type ReactNode,
} from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { MarkdownTextPrimitive } from "@assistant-ui/react-markdown";
import { makeLightSyntaxHighlighter } from "@assistant-ui/react-syntax-highlighter";

import "katex/dist/katex.min.css";
import "@assistant-ui/react-markdown/styles/dot.css";

import bash from "react-syntax-highlighter/dist/esm/languages/hljs/bash";
import css from "react-syntax-highlighter/dist/esm/languages/hljs/css";
import go from "react-syntax-highlighter/dist/esm/languages/hljs/go";
import javascript from "react-syntax-highlighter/dist/esm/languages/hljs/javascript";
import json from "react-syntax-highlighter/dist/esm/languages/hljs/json";
import markdown from "react-syntax-highlighter/dist/esm/languages/hljs/markdown";
import python from "react-syntax-highlighter/dist/esm/languages/hljs/python";
import rust from "react-syntax-highlighter/dist/esm/languages/hljs/rust";
import sql from "react-syntax-highlighter/dist/esm/languages/hljs/sql";
import typescript from "react-syntax-highlighter/dist/esm/languages/hljs/typescript";
import xml from "react-syntax-highlighter/dist/esm/languages/hljs/xml";
import yaml from "react-syntax-highlighter/dist/esm/languages/hljs/yaml";
import atomOneLight from "react-syntax-highlighter/dist/esm/styles/hljs/atom-one-light";
import atomOneDark from "react-syntax-highlighter/dist/esm/styles/hljs/atom-one-dark";

/** 代码块语言：hljs 语法 + 常见别名。 */
const LANGUAGES = {
  bash,
  css,
  go,
  javascript,
  json,
  markdown,
  python,
  rust,
  sql,
  typescript,
  xml,
  yaml,
  // 别名
  sh: bash,
  shell: bash,
  zsh: bash,
  js: javascript,
  ts: typescript,
  html: xml,
  yml: yaml,
  py: python,
  md: markdown,
};

const REMARK_PLUGINS = [remarkGfm, remarkMath];
const REHYPE_PLUGINS = [rehypeKatex];

/** 代码块外层的 Pre/Code 壳：高亮器只负责内容。 */
const PreTag = (props: ComponentPropsWithoutRef<"pre">) => <pre {...props} />;
const CodeTag = (props: ComponentPropsWithoutRef<"code">) => <code {...props} />;
const TAGS = { Pre: PreTag, Code: CodeTag };

type Highlighter = ReturnType<typeof makeLightSyntaxHighlighter>;

/** 跟随系统深浅色切换代码主题。 */
function useDarkMode(): boolean {
  const [dark, setDark] = useState(
    () =>
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-color-scheme: dark)").matches,
  );

  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => setDark(mq.matches);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  return dark;
}

function useHighlighter(): Highlighter {
  const dark = useDarkMode();
  return useMemo(
    () =>
      makeLightSyntaxHighlighter({
        style: dark ? atomOneDark : atomOneLight,
        languages: LANGUAGES,
      }),
    [dark],
  );
}

/** 历史流的 text part。 */
export function MarkdownText() {
  const SyntaxHighlighter = useHighlighter();
  return (
    <MarkdownTextPrimitive
      className="aui-md"
      remarkPlugins={REMARK_PLUGINS}
      rehypePlugins={REHYPE_PLUGINS}
      components={{ SyntaxHighlighter }}
    />
  );
}

/** 任意字符串的 markdown 渲染（舞台面、无 part 上下文处）。 */
export function Markdown({ text }: { text: string }) {
  const SyntaxHighlighter = useHighlighter();

  const components = useMemo(
    () => ({
      pre: (props: { children?: ReactNode }) => (
        <PreBlock highlighter={SyntaxHighlighter} {...props} />
      ),
    }),
    [SyntaxHighlighter],
  );

  return (
    <div className="aui-md">
      <ReactMarkdown
        remarkPlugins={REMARK_PLUGINS}
        rehypePlugins={REHYPE_PLUGINS}
        components={components}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}

function PreBlock({
  children,
  highlighter: SyntaxHighlighter,
}: {
  children?: ReactNode;
  highlighter: Highlighter;
}) {
  const code = isValidElement(children)
    ? (children as ReactElement<{ className?: string; children?: ReactNode }>)
    : null;
  const language =
    /language-([\w-]+)/.exec(code?.props.className ?? "")?.[1] ?? "text";

  return (
    <SyntaxHighlighter
      language={language}
      code={String(code?.props.children ?? "").replace(/\n$/, "")}
      components={TAGS}
    />
  );
}
