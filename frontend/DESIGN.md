# NAN GUI · 设计规范 v1

状态:β 骨架冻结,原语系统 v1。样例:`frontend/samples/f-primitives.html`

---

## 0. 原则

1. 人话用无衬线,机器话用等宽——两种质感,不混排。
2. 流内无卡片。一切机器事件降为 `Record` 行;盒子只属于 chrome(输入框、浮层)。
3. 空闲是常态噪声:心跳只进状态(Pulse),不进流。
4. 极简起步:元素就 6 个,加任何东西前先问能不能用现有原语表达。

## 1. 骨架(β 三明治)

```
Topbar   状态机(40px):品牌 · Pulse ● · 状态文本 · 模型信息
Stream   单列居中,max-width 920px,无限滚动 + 日期分隔
Composer 底部固定(❯ + 输入 + ⏎),右侧状态摘记
```

## 2. Tokens

### 颜色(薄暮实验室)
| token | 值 | 用途 |
|---|---|---|
| --bg | #161514 | 画布 |
| --raise | #262420 | 浮起:输入框、详节 hover |
| --line | rgba(232,227,221,.08) | 发丝线 |
| --ink | #E8E3DD | 正文 |
| --dim | #8A8580 | 次要文字 |
| --faint | #5C5751 | 注脚/停用 |
| --accent | #D97757 | 活动/进行中 |
| --accent-2 | #E8A860 | 高亮/沉淀(v2) |
| --ok | #8FBE6D | 成功 |
| --err | #E5654F | 失败 |

### 字体
| token | 值 | 用途 |
|---|---|---|
| --sans | -apple-system / PingFang SC | 人话:Message、Composer |
| --mono | Sarasa Mono SC / JetBrains Mono | 机器话:Record、Note、状态 |

字号:正文 15 / 机器 12.5 / 注 11。行高:正文 1.85 / 机器 1.9。

### 动效(仅三种)
| 名 | 参数 | 用途 |
|---|---|---|
| rise | 240ms ease-out | 新行入场 |
| breathe | 1.1s 循环 | working 状态(光标/符号) |
| blink | 1.2s steps | idle 光标 |

## 3. 原语

### Message(人话)
- 用户:右对齐气泡(底 rgba 232,227,221,.075;圆角 16/5;sans 15)
- NAN:**无头像、无标题、无时间头**——正文裸段直接跟在 Record 流后;
  时间戳以 Note 嵌入输出尾部:`✓ 22:41:40 · 0.9s · turn #042`
- 流的模型:**过程 → 输出 → 过程**,不断循环(无限 loop)。
  NAN 正文只是记录流中的"人话段",不是带头的"消息单元";
  sleep(15) 也是过程,以 `▸ sleep` Record 收尾,下一跳时间由它携带
- 排队中(输入先于回应):气泡 60% 透明度 + Note「已入队」

### Record(机器话,核心原语)
```
[glyph] name · summary        [note] [chev]
        ↳ detail(mono 12,faint,缩进 24px)
```
- glyph 字母表:`▸ 动作` `◈ 委派` `✓ 完成` `✗ 失败` `↳ 详节` `✦ 沉淀(v2)` `● 活着` `▍ 光标`
- 状态:running(braille 转轮 ⠋⠙⠹… + accent 呼吸)→ done(✓ + 自动折叠)/ failed(✗ + 保持展开,--err)
- 交互:点击行切换详节;hover 发丝底(raise 4.5%)
- 使用者:工具调用、子代理派发/简报、子代理报告、错误/重试、(v2)记忆沉淀

### Note(元数据)
mono 11px --faint。时间戳、耗时、上下文数。永远跟随宿主,不单独成行。

### Pulse(状态)
呼吸点/光标。位置:Topbar 圆点 + Composer 状态摘记。语义:琥珀呼吸=working,绿点=空闲。

### Divider(分隔)
日期粘性分隔:`── 8月27日 ──`(mono 11 faint 居中)。旧回合折叠线同语法:`▸ turn #001–#039 已折叠`。

### Composer(输入)
底部固定,与列同宽。`❯` 提示符 accent;占位符 faint;右端 ⏎ + 状态摘记。

## 4. 交互规则

1. Record 完成即自动折叠;错误保持展开直到被重试消化。
2. 重试叙事:`✗ files.read · 超时` → 新 Record `▸ files.read · 重试 → ✓`(不合并,保留轨迹)。
3. 流式进行中 Composer 永远可用;新输入以排队气泡进入流尾。
4. 状态文本三态:`空闲 · sleep` / `工作中 · 工具 n · 子代理 n` / `工作中 · 等待报告`。

## 5. 宽度与密度

- 列宽 --colw:920px(可调旋钮 720~1080);正文 15px 下约 45 汉字/行。
- 密度开关(v2):详细(D1 全 Record 详节默认可见)/ 安静(D3 Record 折成一行小字)。

## 6. v1 不做
记忆 pill、WORKSTATE 面板、journal 视图、`/` 命令、抽屉、侧栏、多会话。
