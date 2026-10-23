// 窗口集合 + 唯一活动窗口：整个桌面的导航状态。
//
// 两个字段：order（rail 的固定顺序，注册序）与 active（占据内容区的
// 那一个窗口）。没有次区 —— 一个内容位，切换就是切换，点两下必然
// 回到原处。位置即状态，角色不存储。
//
// 纯函数：输入状态返回新状态，可直接单测。

export interface WindowRef {
  id: string;
  /** 内建 agent 面板，不可注销；服务窗口由注册产生。 */
  kind: "agent" | "service";
  title: string;
  /** 16px 图标的 SVG path 内容；缺省时用 id 首字母。 */
  icon?: string;
  /** 服务声明的主色，仅用于状态点之外的可选强调；留空即跟随主题。 */
  serviceId?: string;
  url?: string;
  /** rail 上的未读角标；激活时清零。 */
  unread?: number;
}

export interface QueueState {
  /** rail 的渲染顺序。注册后不再变动，chip 永不跳动。 */
  order: string[];
  /** 内容区当前展示的窗口；null = 空工作区。 */
  active: string | null;
}

export function emptyQueue(): QueueState {
  return { order: [], active: null };
}

function without(list: string[], id: string): string[] {
  return list.filter((x) => x !== id);
}

/** 内建 agent 面板永远在册；服务窗口按注册顺序追加到尾部。 */
export function register(
  state: QueueState,
  id: string,
  first = false,
): QueueState {
  if (state.order.includes(id)) return state;
  return {
    order: first ? [id, ...state.order] : [...state.order, id],
    // 注册不抢内容位：新窗口先在 rail 等
    active: state.active,
  };
}

/** 服务注销：窗口真正消失。活动窗口被注销则回到空工作区。 */
export function unregister(state: QueueState, id: string): QueueState {
  return {
    order: without(state.order, id),
    active: state.active === id ? null : state.active,
  };
}

/** 切换：把该窗口放到唯一的内容位。 */
export function focus(state: QueueState, id: string): QueueState {
  if (!state.order.includes(id) || state.active === id) return state;
  return { ...state, active: id };
}

/** 关闭：退回 rail，工作区留空（窗口没被销毁）。 */
export function close(state: QueueState, id: string): QueueState {
  if (state.active !== id) return state;
  return { ...state, active: null };
}

export function isActive(state: QueueState, id: string): boolean {
  return state.active === id;
}
