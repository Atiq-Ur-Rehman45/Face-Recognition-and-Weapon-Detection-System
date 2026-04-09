/**
 * app.js — Shared utilities, WebSocket setup, toast notifications
 * Loaded on every page via base.html
 */

// ── WebSocket Connection ──────────────────────────────────────────────────────
const socket = io({ transports: ["websocket", "polling"] });

let isNavigating = false;
window.addEventListener("beforeunload", () => {
  isNavigating = true;
});

socket.on("connect", () => {
  setSidebarStatus(true);
});

socket.on("disconnect", () => {
  setSidebarStatus(false);
  if (!isNavigating) {
    showToast("Connection lost — reconnecting...", "warning");
  }
});

socket.on("status:init", (data) => {
  document.getElementById("sidebar-engine-text").textContent =
    data.engine || "";
  setSidebarStatus(true, data.model_loaded);
});

socket.on("status:fps_update", (data) => {
  // Let individual page scripts handle this if present
  if (window.onFpsUpdate) window.onFpsUpdate(data);
});

socket.on("alert:face_detected", (data) => {
  if (window.onFaceAlert) window.onFaceAlert(data);
  // Show toast on every page
  showToast(`⚠ Criminal: ${data.name} (${data.confidence}%)`, "error", 6000);
});

socket.on("alert:weapon_detected", (data) => {
  if (window.onWeaponAlert) window.onWeaponAlert(data);
  showToast(`🔫 Weapon: ${data.weapon_types} [${data.threat_level}]`, "warning", 6000);
});

socket.on("status:feed_started", () => {
  if (window.onFeedStarted) window.onFeedStarted();
});

socket.on("status:feed_stopped", () => {
  if (window.onFeedStopped) window.onFeedStopped();
});

socket.on("status:video_ended", () => {
  showToast("Video file playback complete", "info");
  if (window.onFeedStopped) window.onFeedStopped();
});

socket.on("snapshot:saved", (data) => {
  showToast("📸 Snapshot saved", "success");
});

// ── Sidebar Status Indicator ──────────────────────────────────────────────────
function setSidebarStatus(online, modelLoaded) {
  const dot  = document.getElementById("status-dot");
  const text = document.getElementById("sidebar-status-text");
  if (!dot || !text) return;

  if (!online) {
    dot.className = "system-status-dot offline";
    text.textContent = "Disconnected";
    return;
  }

  if (modelLoaded === false) {
    dot.className = "system-status-dot warning";
    text.textContent = "Model not trained";
  } else if (modelLoaded === true) {
    dot.className = "system-status-dot";
    text.textContent = "System online";
  } else {
    dot.className = "system-status-dot";
    text.textContent = "Connected";
  }
}

// ── Toast Notifications ───────────────────────────────────────────────────────
const TOAST_ICONS = { success: "✅", error: "🚨", warning: "⚠️", info: "ℹ️" };

function showToast(message, type = "info", duration = 4000) {
  const container = document.getElementById("toast-container");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <span class="toast-icon">${TOAST_ICONS[type] || "ℹ️"}</span>
    <span class="toast-text">${message}</span>
  `;
  container.appendChild(toast);

  setTimeout(() => {
    toast.classList.add("hide");
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// ── Confirm Modal ─────────────────────────────────────────────────────────────
let _confirmCallback = null;

function showConfirm(title, message, onOk, okLabel = "Confirm", dangerous = true) {
  document.getElementById("confirm-title").textContent   = title;
  document.getElementById("confirm-message").textContent = message;
  const okBtn = document.getElementById("confirm-ok-btn");
  okBtn.textContent  = okLabel;
  okBtn.className    = `btn ${dangerous ? "btn-danger" : "btn-primary"}`;
  _confirmCallback   = onOk;
  document.getElementById("confirm-modal").classList.remove("hidden");
}

function closeConfirm() {
  document.getElementById("confirm-modal").classList.add("hidden");
  _confirmCallback = null;
}

document.getElementById("confirm-ok-btn")?.addEventListener("click", () => {
  const cb = _confirmCallback;
  closeConfirm();
  if (cb) cb();
});

// Close modal on overlay click
document.getElementById("confirm-modal")?.addEventListener("click", (e) => {
  if (e.target === e.currentTarget) closeConfirm();
});

// ── REST Helpers ──────────────────────────────────────────────────────────────
async function apiGet(url) {
  const res = await fetch(url);
  return res.json();
}

async function apiPost(url, body = {}) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

async function apiPut(url, body = {}) {
  const res = await fetch(url, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

async function apiDelete(url) {
  const res = await fetch(url, { method: "DELETE" });
  return res.json();
}

// ── Feed Controls (shared by Dashboard and Monitor) ───────────────────────────
async function startFeed(cameraIndex = 0) {
  const res = await apiPost("/api/feed/start", { source: "camera", camera_index: cameraIndex });
  if (!res.success) showToast(res.error || "Failed to start feed", "error");
  else showToast("📹 Live feed started", "success");
  return res;
}

async function stopFeed() {
  const res = await apiPost("/api/feed/stop");
  if (!res.success) showToast(res.error || "Failed to stop feed", "error");
  return res;
}

async function takeSnapshot() {
  const res = await apiPost("/api/feed/snapshot");
  if (!res.success) showToast("No active feed for snapshot", "warning");
}

// ── Build Alert Card HTML ─────────────────────────────────────────────────────
function buildFaceAlertCard(data) {
  const thumb = data.snapshot_url
    ? `<img class="alert-thumb" src="${data.snapshot_url}" alt="snapshot"/>`
    : `<div class="alert-thumb-placeholder">👤</div>`;
  return `
    <div class="alert-card face">
      ${thumb}
      <div class="alert-body">
        <div class="alert-name">${data.name}</div>
        <div class="alert-meta">${data.crime_type} · ${data.confidence}% match</div>
      </div>
      <div class="alert-time">${data.timestamp}</div>
    </div>`;
}

function buildWeaponAlertCard(data) {
  const thumb = data.snapshot_url
    ? `<img class="alert-thumb" src="${data.snapshot_url}" alt="snapshot"/>`
    : `<div class="alert-thumb-placeholder">🔫</div>`;
  const badgeClass = { CRITICAL: "badge-red", HIGH: "badge-orange", MEDIUM: "badge-yellow" }[data.threat_level] || "badge-muted";
  return `
    <div class="alert-card weapon">
      ${thumb}
      <div class="alert-body">
        <div class="alert-name">${data.weapon_types}</div>
        <div class="alert-meta"><span class="badge ${badgeClass}">${data.threat_level}</span> · ${data.confidence}%</div>
      </div>
      <div class="alert-time">${data.timestamp}</div>
    </div>`;
}

// ── Pagination Builder ────────────────────────────────────────────────────────
function buildPagination(containerId, currentPage, totalPages, onPageClick) {
  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = "";
  if (totalPages <= 1) return;

  const addBtn = (label, page, active = false, disabled = false) => {
    const btn = document.createElement("button");
    btn.className = `page-btn${active ? " active" : ""}`;
    btn.textContent = label;
    btn.disabled = disabled;
    btn.onclick = () => onPageClick(page);
    container.appendChild(btn);
  };

  addBtn("‹", currentPage - 1, false, currentPage === 1);
  for (let p = 1; p <= totalPages; p++) {
    if (totalPages > 7 && Math.abs(p - currentPage) > 2 && p !== 1 && p !== totalPages) {
      if (p === 2 || p === totalPages - 1) { addBtn("…", p, false, true); }
      continue;
    }
    addBtn(p, p, p === currentPage);
  }
  addBtn("›", currentPage + 1, false, currentPage === totalPages);
}

// ── Status Badge HTML ─────────────────────────────────────────────────────────
function statusBadge(status) {
  const map = {
    Wanted:   "badge-red",
    Arrested: "badge-orange",
    Released: "badge-muted",
    Deceased: "badge-muted",
  };
  return `<span class="badge ${map[status] || "badge-muted"}">${status}</span>`;
}

// ── Clock ─────────────────────────────────────────────────────────────────────
function updateClock(elementId) {
  const el = document.getElementById(elementId);
  if (!el) return;
  const tick = () => {
    el.textContent = new Date().toLocaleTimeString("en-US", { hour12: false });
  };
  tick();
  setInterval(tick, 1000);
}
