/** dashboard.js — Dashboard page logic */

updateClock("dash-time");

let alertsEmpty = true;
const MAX_DASH_ALERTS = 10;

// ── WebSocket callbacks ───────────────────────────────────────────────────────
window.onFaceAlert = function (data) {
  prependAlert(buildFaceAlertCard(data));
  document.getElementById("stat-face-logs").textContent =
    parseInt(document.getElementById("stat-face-logs").textContent || 0) + 1;
};

window.onWeaponAlert = function (data) {
  prependAlert(buildWeaponAlertCard(data));
  document.getElementById("stat-weapon-logs").textContent =
    parseInt(document.getElementById("stat-weapon-logs").textContent || 0) + 1;
};

window.onFpsUpdate = function (data) {
  document.getElementById("feed-fps-badge").textContent = `FPS: ${data.fps}`;
  document.getElementById("hud-fps").textContent    = data.fps;
  document.getElementById("hud-faces").textContent   = data.faces;
  const wEl = document.getElementById("hud-weapons");
  wEl.textContent = data.weapons;
  wEl.style.color = data.weapons > 0 ? "var(--red)" : "var(--text-primary)";
  document.getElementById("hud-source").textContent  = (data.source || "--").replace("camera:", "Cam ").replace("file:", "");
  document.getElementById("hud-card").style.display = "block";
};

window.onFeedStarted = function () {
  document.getElementById("feed-status-badge").style.display = "inline-flex";
};

window.onFeedStopped = function () {
  document.getElementById("feed-status-badge").style.display = "none";
  document.getElementById("hud-card").style.display = "none";
};

// ── Alert list helpers ────────────────────────────────────────────────────────
function prependAlert(html) {
  const list = document.getElementById("alerts-list");
  if (alertsEmpty) {
    list.innerHTML = "";
    alertsEmpty    = false;
  }
  list.insertAdjacentHTML("afterbegin", html);
  // Cap list size
  const cards = list.querySelectorAll(".alert-card");
  if (cards.length > MAX_DASH_ALERTS) {
    cards[cards.length - 1].remove();
  }
}

// ── Load recent alerts from DB on page load ───────────────────────────────────
async function loadRecentAlerts() {
  try {
    const [faceRes, weaponRes] = await Promise.all([
      apiGet("/api/logs/faces?limit=5"),
      apiGet("/api/logs/weapons?limit=3"),
    ]);

    const faceItems = (faceRes.data || []).map((r) => ({
      type: "face",
      ts: r.timestamp,
      html: buildFaceAlertCard({
        name: r.detected_name,
        confidence: Math.round((r.confidence || 0) * 100),
        crime_type: "Recorded",
        timestamp: (r.timestamp || "").slice(11, 19),
        snapshot_url: r.snapshot_url,
      }),
    }));

    const weaponItems = (weaponRes.data || []).map((r) => ({
      type: "weapon",
      ts: r.timestamp,
      html: buildWeaponAlertCard({
        weapon_types: r.weapon_types || "Unknown",
        threat_level: r.threat_level || "HIGH",
        confidence: Math.round((r.max_confidence || 0) * 100),
        timestamp: (r.timestamp || "").slice(11, 19),
        snapshot_url: r.snapshot_url,
      }),
    }));

    const all = [...faceItems, ...weaponItems]
      .sort((a, b) => (b.ts > a.ts ? 1 : -1))
      .slice(0, MAX_DASH_ALERTS);

    if (all.length > 0) {
      const list = document.getElementById("alerts-list");
      list.innerHTML = all.map((x) => x.html).join("");
      alertsEmpty = false;
    }
  } catch (e) {
    // Silently ignore — DB may not have data yet
  }
}

// ── Init ──────────────────────────────────────────────────────────────────────
loadRecentAlerts();
