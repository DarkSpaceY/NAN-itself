# NAN Network Perception · 网络感知权威清单

> 三原则同前（push-only 流 / 只出事实 / 要么做要么不做）。
> 图例：🟢 psutil/标准库　🟡 平台命令或出站查询　🔴 特权/第三方/强隐私

## ✅ MVP（本次全做）

### L1 链路与地址
| 事实 | 来源 |
|---|---|
| 接口 UP-DOWN、本地 IPv4 | 🟢 psutil.net_if_addrs/stats |
| 默认网关 + 出接口 | 🟡 route/netsh 解析 |
| 隧道接口在途（tun/tap/ppp/wg/ipsec） | 🟢 接口名模式 + 网关出接口 |

### L2 公网身份
| 事实 | 来源 |
|---|---|
| 公网 IP（10min 低频）+ 变化事件 | 🟡 出站查询（端点可配，默认 ipify/3322 兜底）⚠️出站第三方 |

### L3 流量与连接（push-only 核心）
| 事实 | 来源 |
|---|---|
| 每接口 ↓↑ 速率（字节差分） | 🟢 net_io_counters |
| 断流/突增 | 🟢 相对基线 |
| 连接表：ESTABLISHED 总数/公网连接数/监听数 | 🟢 net_connections |
| 新外联事件（进程名 → IP:port，仅元数据） | 🟢 连接 diff |
| 监听端口新增 | 🟢 diff |

### L4 可达性与质量
| 事实 | 来源 |
|---|---|
| 在线/离线 + 恢复时长 | 🟡 主动探测（端点可配，默认 generate_204/百度） |
| 延迟 ms | 🟡 探测计时 |
| 系统代理设置 | 🟡 env/scutil |

### L7 自身出站
| 事实 | 来源 |
|---|---|
| agent 出站端点可达性 | 🟢 复用探测结果（LLM 端点不可达=自知） |

## 渲染契约
```
[Network]
- link: wi-fi "HOME" 192.168.1.23 -> gw 192.168.1.1 | online 34ms
- public: 203.0.113.7 | tunnel: none
- traffic: ↓2.3MB/s ↑210KB/s
- conn: 68 est (12 public) | listen 9
- events: new outbound Chrome -> 140.82.x.x:443 | offline 40s -> online
```
安静纪律：events 仅在存在时出现；offline 持续时 link 行降级为事实陈述。

## 🚫 明确不做（含理由）
| 项 | 理由 |
|---|---|
| WiFi RSSI/频段/信道、以太网协商、MTU | 无当前用例 |
| 公网 IP→地理/ASN | 出站第三方 + 隐私 |
| 逐进程流量账本 | 特权（nettop/nethogs/eBPF）+ 隐私 |
| DNS 查询内容 / 抓包 | 强隐私，永久 |
| 丢包率统计 | 探测频率撑不起 |
| captive portal / IPv6 / 防火墙态 / hosts / ARP 欺骗 / NTP 偏移 | 无用例或第二批候选（未承诺） |
| "视频会议中/网络不可信"等合成 | 违反启发式纪律——事实归本模块，结论归 agent |

## 扩展纪律
探测端点全部可配（NAN_NET_PROBE_URL / NAN_NET_IP_URL）；
出站查询失败静默降级；先表后码；无 ⬜ 中间态。
