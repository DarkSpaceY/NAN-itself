'use strict';

// Pure window: the core owns the lifecycle. `nan` starts the gateway,
// spawns this process with NAN_URL, and shuts down when the window
// closes. No core is spawned here; no tray, no background service.

const { app, BrowserWindow, dialog, shell } = require('electron');
const path = require('node:path');

const ICON_PATH = path.join(__dirname, 'icon.png');
const APP_URL = process.env.NAN_URL;
const APP_ORIGIN = APP_URL ? new URL(APP_URL).origin : null;

let win = null;

function isLocalUrl(url) {
  try {
    return new URL(url).origin === APP_ORIGIN;
  } catch {
    return false;
  }
}

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

  win.loadURL(APP_URL);

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

if (!APP_URL) {
  // Started by hand (npm start) or by an older core: refuse loudly
  // instead of showing a blank window.
  app.whenReady().then(() => {
    dialog.showErrorBox('NAN 启动失败', '缺少 NAN_URL：请通过 `uv run nan` 启动。');
    app.exit(1);
  });
} else {
  app.on('window-all-closed', () => {
    app.quit();
  });

  app.whenReady().then(() => {
    // Unbundled runs show Electron's default Dock icon; set ours
    // explicitly. It would come from the bundle once packaged.
    if (process.platform === 'darwin' && app.dock) {
      app.dock.setIcon(ICON_PATH);
    }

    createWindow();
  });
}
