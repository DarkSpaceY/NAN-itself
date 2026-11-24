'use strict';

// Thin desktop shell: spawn the Python core, wait for its gateway, and
// load the same UI in a BrowserWindow. Closing the window quits the
// core; no tray, no background service.

const { app, BrowserWindow, dialog, shell } = require('electron');
const { spawn } = require('node:child_process');
const http = require('node:http');
const net = require('node:net');
const path = require('node:path');

const HOST = '127.0.0.1';
const PORT = 8765;
const GATEWAY_URL = `http://${HOST}:${PORT}/`;
const REPO_ROOT = path.resolve(__dirname, '..');
const ICON_PATH = path.join(__dirname, 'icon.png');

const POLL_INTERVAL_MS = 250;
const STARTUP_TIMEOUT_MS = 60_000;
const SHUTDOWN_TIMEOUT_MS = 10_000;

let core = null;
let win = null;
let quitting = false;
let running = false;

// ---------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function portInUse() {
  return new Promise((resolve) => {
    const socket = net.connect({ host: HOST, port: PORT });
    const done = (inUse) => {
      socket.destroy();
      resolve(inUse);
    };

    socket.once('connect', () => done(true));
    socket.once('error', () => done(false));
    socket.setTimeout(1000, () => done(false));
  });
}

function probeGateway() {
  return new Promise((resolve) => {
    const request = http.get(GATEWAY_URL, (response) => {
      response.resume();
      resolve(response.statusCode === 200);
    });

    request.setTimeout(1000, () => request.destroy());
    request.once('error', () => resolve(false));
  });
}

function isLocalUrl(url) {
  try {
    return new URL(url).origin === `http://${HOST}:${PORT}`;
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------
// core process
// ---------------------------------------------------------------------

async function startCore() {
  // The core refuses to boot on an occupied port; catching it here
  // gives a readable message instead of a child that exits silently.
  if (await portInUse()) {
    throw new Error(
      `端口 ${PORT} 已被占用，可能已有一个 NAN 核心在运行。请先关闭它再启动桌面端。`,
    );
  }

  core = spawn('uv', ['run', '--frozen', 'nan', '--no-browser'], {
    cwd: REPO_ROOT,
    env: process.env,
    stdio: ['ignore', 'pipe', 'pipe'],
    // The child leads its own process group, so a group signal reaches
    // the Python core under `uv run` as well as uv itself.
    detached: true,
  });

  core.stdout.on('data', (chunk) => process.stdout.write(chunk));
  core.stderr.on('data', (chunk) => process.stderr.write(chunk));

  core.once('error', (error) => {
    console.error('failed to spawn the NAN core:', error);
  });

  core.once('exit', (code, signal) => {
    if (quitting) {
      return;
    }

    // A startup-time exit is reported by the awaiting caller.
    if (!running) {
      return;
    }

    dialog.showErrorBox(
      'NAN 已停止',
      `核心进程退出（code ${code ?? '-'}，signal ${signal ?? '-'}）。`,
    );
    app.quit();
  });

  // A failed spawn (uv missing from PATH, for example) emits 'error'
  // instead of 'exit'; surface it now rather than after the timeout.
  const spawnFailed = new Promise((_, reject) => core.once('error', reject));

  await Promise.race([waitForGateway(), spawnFailed]);
}

async function waitForGateway() {
  const deadline = Date.now() + STARTUP_TIMEOUT_MS;

  while (Date.now() < deadline) {
    if (core.exitCode !== null || core.signalCode !== null) {
      throw new Error(
        `核心进程在网关就绪前退出（code ${core.exitCode ?? '-'}，signal ${core.signalCode ?? '-'}）。` +
          '请查看终端输出中的日志。',
      );
    }

    if (await probeGateway()) {
      return;
    }

    await sleep(POLL_INTERVAL_MS);
  }

  throw new Error(`等待网关 ${GATEWAY_URL} 超时（${STARTUP_TIMEOUT_MS / 1000}s）。`);
}

function signalCore(signal) {
  if (!core || core.pid === undefined) {
    return false;
  }

  try {
    // Negative pid: the whole group (uv and the Python core inside it).
    process.kill(-core.pid, signal);
    return true;
  } catch (error) {
    if (error.code !== 'ESRCH') {
      console.error(`failed to send ${signal} to the NAN core:`, error);
    }

    return false;
  }
}

function waitForCoreExit(timeoutMs) {
  if (!core || core.exitCode !== null || core.signalCode !== null) {
    return Promise.resolve();
  }

  return Promise.race([
    new Promise((resolve) => core.once('exit', resolve)),
    sleep(timeoutMs),
  ]);
}

function groupAlive() {
  if (!core || core.pid === undefined) {
    return false;
  }

  try {
    process.kill(-core.pid, 0);
    return true;
  } catch {
    return false;
  }
}

async function stopCore() {
  if (!groupAlive()) {
    return;
  }

  // SIGTERM lets the core suspend agents, unregister services, and
  // write the snapshot; SIGKILL is the fallback, not the first move.
  signalCore('SIGTERM');

  // Wait for the whole group, not just uv: uv exits immediately on
  // SIGTERM while the Python core keeps draining, and a restart must
  // not race it for the fixed ports.
  const deadline = Date.now() + SHUTDOWN_TIMEOUT_MS;

  while (Date.now() < deadline) {
    if (!groupAlive()) {
      return;
    }

    await sleep(POLL_INTERVAL_MS);
  }

  console.warn(`core still alive after ${SHUTDOWN_TIMEOUT_MS / 1000}s; sending SIGKILL`);
  signalCore('SIGKILL');
  await waitForCoreExit(5000);
}

// ---------------------------------------------------------------------
// window
// ---------------------------------------------------------------------

function createWindow() {
  win = new BrowserWindow({
    width: 1280,
    height: 860,
    minWidth: 900,
    minHeight: 600,
    title: 'NAN',
    backgroundColor: '#0f1115',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
    },
  });

  // The renderer's HTTP cache survives restarts; without a changing
  // document URL Chromium may keep serving an old index.html (and its
  // hashed bundle) even after a rebuild.
  win.loadURL(`${GATEWAY_URL}?boot=${Date.now()}`);

  // Links that leave the local UI open in the system browser.
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (!isLocalUrl(url)) {
      void shell.openExternal(url);
    }

    return { action: 'deny' };
  });

  win.webContents.on('will-navigate', (event, url) => {
    if (!isLocalUrl(url)) {
      event.preventDefault();
      void shell.openExternal(url);
    }
  });

  win.on('closed', () => {
    win = null;
  });
}

// ---------------------------------------------------------------------
// app lifecycle
// ---------------------------------------------------------------------

async function main() {
  try {
    // Unbundled runs (npm start) show Electron's default Dock icon;
    // set ours explicitly. It would come from the bundle once packaged.
    if (process.platform === 'darwin' && app.dock) {
      app.dock.setIcon(ICON_PATH);
    }

    await startCore();
    running = true;
    createWindow();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);

    dialog.showErrorBox('NAN 启动失败', message);

    quitting = true;
    await stopCore();
    app.exit(1);
  }
}

if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (win) {
      if (win.isMinimized()) {
        win.restore();
      }

      win.focus();
    }
  });

  app.on('window-all-closed', () => {
    app.quit();
  });

  app.on('before-quit', (event) => {
    if (quitting) {
      return;
    }

    quitting = true;
    event.preventDefault();

    void stopCore().finally(() => app.exit(0));
  });

  // `npm start` runs Electron in a terminal: Ctrl+C should take the
  // core down through the same graceful path, not orphan it.
  for (const signal of ['SIGINT', 'SIGTERM']) {
    process.on(signal, () => app.quit());
  }

  app.whenReady().then(main);
}
