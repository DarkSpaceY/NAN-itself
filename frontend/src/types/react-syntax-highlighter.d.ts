// react-syntax-highlighter 的语言语法与主题是深层导入，包内没有随附类型，
// 这里补最小声明。语法模块导出 hljs 语法对象，主题模块导出样式映射。
declare module "react-syntax-highlighter/dist/esm/languages/hljs/*" {
  const language: unknown;
  export default language;
}

declare module "react-syntax-highlighter/dist/esm/styles/hljs/*" {
  const style: Record<string, React.CSSProperties>;
  export default style;
}
