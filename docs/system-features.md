# NAN System Perception · 设备自我感知权威清单

> 三原则：
> 1. Module 只做 push-only 的流；状态可拉的交给 tool（touchpoint 教训）。
> 2. **Module 出事实，agent 出判断**——本模块渲染零启发式、零结论，
>    只有可直接验证的测量与事件。合成推断（会议中/勿扰/位置阶段）
>    属于 agent 层，精度不达 90% 的启发式永远不进本模块。
> 3. 要么做要么不做：✅ 本次实现 / 🚫 明确不做（带理由），无中间态。
>
> 图例：🟢 psutil 跨平台　🟡 平台 API 各一小段（utils 层可注入）
> 信号路径：全部确定性函数，无模型。

## ✅ MVP（本次全做）

### 快照（5s 周期）
| 事实 | 来源 |
|---|---|
| CPU 占用（总体） | 🟢 psutil |
| 内存/swap | 🟢 psutil |
| 磁盘根区用量 | 🟢 psutil |
| 电池电量/充电态/电源来源 | 🟢 psutil（无电池则略行） |
| 网络接口在线态 | 🟢 psutil |
| uptime | 🟢 psutil.boot_time |

### 时间与节律（每回合事实）
| 事实 | 来源 |
|---|---|
| 本地时间/星期/时区偏移 | 标准库 |
| 开机时长 awake | 🟢 |
| 连续工作 work | 空闲信号确定性合成（自上次空闲≥5min 起） |

### 交互流
| 事实 | 来源 |
|---|---|
| 系统空闲时长 idle | 🟡 macOS ioreg / Linux xprintidle / Win GetLastInputInfo |
| 前台应用名（变化流） | 🟡 osascript/xdotool/ctypes |
| 应用启动/退出事件 | 轮询 diff |

### 事件流（push-only 的存在理由）
| 事件 | 检测（确定性） |
|---|---|
| 睡眠/唤醒 | time.time()-monotonic 跳变 >60s |
| WiFi SSID 变化 | 🟡 平台查询 diff |
| 显示器数量变化 | 🟢 mss.monitors diff |
| 磁盘将满 | 🟢 ≥90% 越线（10min 报一次） |
| 失控进程 | 🟢 同 PID CPU>90% 连续 3 样本 |

### 自律
| 事实 | 来源 |
|---|---|
| NAN 自身 CPU/内存 | 🟢 psutil.Process |

## 渲染契约（单领地，全部事实，零结论）
```
[System]
- now: 14:22 周四 (UTC+8)
- focus: Code (NAN-itself — audio.py)
- idle: 12s | work 25m | awake 6.2h
- cpu 34% | mem 62% | disk 71% | battery 78% charging | net up
- events: woke 14:02 | wifi -> "Home-5G" | runaway: Xcode 95%
- self: nan cpu 2.1% mem 310MB
```
安静纪律：无新事件且快照无明显变化时，变化行省略；idle 长期
高于阈值时仅保留一行事实（"idle 3.2h"），判断留给 agent。

## 🚫 明确不做（首版，含理由）
| 项 | 理由 |
|---|---|
| CPU 温度/风扇/GPU/NPU/SMART | 需特权或平台碎片化，价值/成本不达标 |
| 逐键/鼠标事件流 | 强隐私；空闲+前台已覆盖其合成用途 |
| 系统定位坐标 | 授权成本；WiFi SSID 语义先顶 |
| 蓝牙邻近/USB 插拔 | 第二批再说（无当前用例） |
| 剪贴板 | 隐私敏感，无已确认用例 |
| 摄像头/麦克风占用者 | 平台碎片化；其合成用途（会议检测）按启发式纪律已移除 |
| 崩溃报告流/日志风暴/服务状态/新应用安装 | 第二批候选，未承诺 |
| 会议中/勿扰/位置阶段等启发式合成 | **违反启发式纪律，永久移出本模块**——事实已备齐，判断归 agent |

## 扩展纪律
平台 API 一律封装 utils 层可注入；系统调用失败降级 None 行；
先表后码；无 ⬜ 中间态。
