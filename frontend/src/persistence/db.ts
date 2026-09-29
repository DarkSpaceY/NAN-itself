// 历史轮持久化：IndexedDB，keyPath = round.key。
// 后端不负责历史；这里只在前端本地存（会话生命周期由用户管理）。
// 无 IndexedDB 环境（隐私模式/测试）退化为内存 Map。

import type { HistoryRound } from "../history/mapper";

const DB_NAME = "nan-history";
const DB_VERSION = 1;
const STORE = "rounds";

export interface RoundStore {
  saveRound(round: HistoryRound): Promise<void>;
  loadRounds(): Promise<HistoryRound[]>;
  clear(): Promise<void>;
}

// ------------------------------------------------------------------
// 内存实现（退化 & 测试）
// ------------------------------------------------------------------

export function memoryRoundStore(): RoundStore {
  const map = new Map<string, HistoryRound>();
  return {
    async saveRound(round) {
      map.set(round.key, structuredClone(round));
    },
    async loadRounds() {
      return [...map.values()].map((r) => structuredClone(r));
    },
    async clear() {
      map.clear();
    },
  };
}

// ------------------------------------------------------------------
// IndexedDB 实现
// ------------------------------------------------------------------

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE, { keyPath: "key" });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function reqAsPromise<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export function idbRoundStore(): RoundStore {
  let conn: IDBDatabase | null = null;
  let opening: Promise<IDBDatabase> | null = null;

  const db = async (): Promise<IDBDatabase> => {
    if (conn) return conn;

    opening ??= openDB();
    const dbi = await opening;

    // 别的上下文（另一个标签页、deleteDatabase、版本升级）要求
    // 让出连接时必须关闭，否则那个请求会一直 blocked。关掉后清空
    // 缓存，下次调用自动重开。
    dbi.onversionchange = () => {
      dbi.close();
      conn = null;
      opening = null;
    };

    conn = dbi;
    return conn;
  };

  return {
    async saveRound(round) {
      const dbi = await db();
      const tx = dbi.transaction(STORE, "readwrite");
      await reqAsPromise(tx.objectStore(STORE).put(round));
    },

    async loadRounds() {
      const dbi = await db();
      const tx = dbi.transaction(STORE, "readonly");
      const all = await reqAsPromise(tx.objectStore(STORE).getAll() as IDBRequest<HistoryRound[]>);
      return all.sort((a, b) => a.ts - b.ts);
    },

    async clear() {
      const dbi = await db();
      const tx = dbi.transaction(STORE, "readwrite");
      await reqAsPromise(tx.objectStore(STORE).clear());
    },
  };
}

/** 环境探测：无 indexedDB 时退化。 */
export function createRoundStore(): RoundStore {
  return typeof indexedDB !== "undefined" ? idbRoundStore() : memoryRoundStore();
}
