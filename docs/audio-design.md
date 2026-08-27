# NAN Hearing System · 设计文档 v0（MVP）

状态：MVP 已实现。特征全景与波次路线见 `docs/audio-features.md`，
本文件只描述系统的形状（架构/契约/配置/失败策略）。

---

## 0. 价值基线

1. 与记忆模块同一范式：**读路径零智能，重活全部在后台采集线程**。
2. `query()` 是纯投影：只读已计算好的环状缓冲与统计量，绝不做 DSP/LLM。
3. 连续流的自尊心：宁可丢帧计数上报，也不阻塞采集回调。
4. 声音是**流**不是事件：重要性由 recency × 类型权重决定，由模块自己裁决。
5. 硬件失败（无麦克风/权限拒绝）永远降级为 `available:false`，绝不拖垮进程。

## 1. 特征全景

听觉的全部可实现特征（第 0–8 层）与跨通道涌现形态（第 9 层）
由 `docs/audio-features.md` 单独维护——新增特征先进那张表登记，
再进代码。本文件只关心系统的形状。

MVP 摘要：L1 能量（RMS/噪声底）+ L4 事件（VAD/静默/瞬态）
+ L7 内容（STT/置信度/语言）+ W2 已并入 L2 特征（ZCR/质心/平坦度）
与 L6 启发式环境三分类（speech/tonal/noisy/quiet）。
其余层按 features 文档的 W3–W6 波次推进。

## 2. MVP 范围（本次实现）

选层理由：投入产出最高——4 层让"何时算一句"成立，7 层给出内容，1 层给环境警觉。

```
麦克风(sounddevice, 16k mono int16, 30ms 帧)
        │ 采集线程 A
        ▼
AudioPipeline(utils/audio.py，纯逻辑可测)
   ├─ FeatureTracker   RMS/噪声底/安静时长
   ├─ VadGate          webrtcvad；缺失时能量门限降级
   ├─ UtteranceSegmenter  预卷 300ms + 尾静 700ms 收句，
   │                      最短 250ms，最长 25s 强制收句
   └─ 瞬态检测         底噪+18dB 触发，冷却 1s
        │ 句子(bytes)
        ▼ queue(32)
转写线程 B → WhisperTranscriber(faster-whisper base, lazy import)
        │ {text, conf, lang}
        ▼
TranscriptRing(deque maxlen=hear_history)
```

- **丢包策略**：mic 队列满→丢弃新帧并计数；utterance 队列满→丢弃新句并计数。只在 DataSpace 上报数字，不打断采集。
- **重要性排序（纯算法）**：Heard 最新优先 + 60s 内规范化文本去重；Ambient 一行常显；「安静超 120s 且无新话语」时 query 返回 None 不刷屏。
- **on_turn(record)**：仅记录回合时间戳（maxlen=20），供未来把"听到的话"归属到对话上下文；不触发任何计算。
- **持久化**：serialize 只存计数器与噪声底初始化值；话语环属于瞬态感知，重启即清空（人重新睁眼也会忘记梦里的话）。

## 3. 渲染契约（领地规则）

`<module>` 是所有模块共享的空间，但**框架不做任何模块标注**——
因此每个模块的全部输出必须收拢在自己的领地头（模块 id）之下，
禁止铸造看似全局的 section（如 `[Ambient]`、`[Heard]`）；
领地内部用普通行表达子结构。这是多模块共存时命名空间的唯一防线。

```
[Audio]
- hearing: speech active / quiet 42s
- ambient: tonal / noisy            # 仅非语音且非安静时出现
- level: -38.2 dBFS (noise floor -55.1)
- sharp sound detected at 14:31     # 仅 60s 内出现过瞬态
- heard 14:32 "嘿 NAN，帮我看下这个报错"
- heard 14:31 (conf 0.42) "……什么东西响了一声"
```

不可用时（仍在本模块领地内，一行）：

```
[Audio]
- input unavailable: <原因>
```

## 4. 配置项

| 参数 | 默认 | 说明 |
|---|---|---|
| NAN_AUDIO_ENABLED | 1 | 组合根据此决定是否注册本模块 |
| NAN_AUDIO_DEVICE | 空 | 输出设备索引（空=系统默认） |
| NAN_AUDIO_SAMPLE_RATE | 16000 | 采集率（whisper 要求 16k） |
| NAN_AUDIO_WHISPER_MODEL | base | faster-whisper 型号 |
| NAN_AUDIO_MODELS_DIR | <repo>/models/whisper | 下载/缓存目录 |
| frame_ms | 30 | 帧长 |
| vad_aggressiveness | 2 | webrtcvad 0–3 |
| preroll_ms | 300 | 句前预卷 |
| trailing_silence_ms | 700 | 尾部静默判句 |
| min_utterance_ms | 250 | 过短丢弃 |
| max_utterance_ms | 25000 | 强制收句 |
| transient_rise_db | 18 | 瞬态触发阈值（相对噪声底） |
| ambient_window_frames | 33 | 环境特征滑动窗口（≈1s，30ms 帧） |
| hear_history | 8 | Heard 环深度 |
| hear_preview_cap | 200 | 单条话语预览字符上限 |
| quiet_report_after_s | 120 | 安静多久后不再输出 query |
| retry_interval | 30 | 设备失败重试周期 |

## 5. 失败与边界

| 场景 | 处理 |
|---|---|
| 无设备/权限拒绝 | available:false，query 输出一行原因，30s 重试 |
| webrtcvad 缺失 | 能量门限降级（floor+10dB 判 speech），功能不中断 |
| faster-whisper 加载失败 | available 保持 true 但转写停摆，loguru CRITICAL；句子仍进队列供未来消费者 |
| 队列溢出 | 丢新留旧，dropped 计数进 DataSpace |
| 进程退出 | stop() 置位 Event → join 线程（≤3s）→ 关闭音频流 |

## 6. 测试契约清单

单元层：
1. Segmenter：预卷/尾静/最短丢弃/最长强制收句四条边界
2. FeatureTracker：噪声底只在持续非语音时下沉
3. 瞬态去抖（1s 冷却内不重复报）
4. query 渲染：Ambient 常显、Heard 去重、空场景返 None
5. 设备不可用降级路径
6. serialize/restore 往返
7. query 耗时上界（纯投影 <50ms）

真机冒烟：
- S1 对空气说一句话 → 数秒内 query 出现该句（含置信度）
- S2 全程安静 → query 在 120s 后静默返 None
- S3 拍桌一声 → Ambient 或瞬态可见
- S4 断开麦克风再插回 → 自动恢复

## 7. 后续波次路线

见 `docs/audio-features.md` 的「实现波次」表；此处不再重复。

调研备忘：RealtimeSTT（现 utils/listen.py）验证过子进程 RT-STT 可行性但隐藏了原始 PCM——W1 由本模块自管采集取而代之；listen.py 保留作参考实现不删除。
