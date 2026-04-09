/** logs.js — Detection Logs page logic */

let faceLogPage    = 1;
let weaponLogPage  = 1;
const LOG_LIMIT    = 20;
let currentTab     = "faces";

// ── Tab switching ─────────────────────────────────────────────────────────────
function switchTab(tab) {
  currentTab = tab;
  ["faces", "weapons"].forEach((t) => {
    document.getElementById(`tab-${t}`).classList.toggle("active", t === tab);
    document.getElementById(`tab-${t}-btn`).classList.toggle("active", t === tab);
  });
  if (tab === "faces")   loadFaceLogs();
  if (tab === "weapons") loadWeaponLogs();
}

// ── Face logs ─────────────────────────────────────────────────────────────────
async function loadFaceLogs() {
  faceLogPage = faceLogPage || 1;
  const offset = (faceLogPage - 1) * LOG_LIMIT;
  const name   = (document.getElementById("face-search")?.value || "").trim();
  const res    = await apiGet(`/api/logs/faces?limit=${LOG_LIMIT}&offset=${offset}`);
  if (!res.success) return;

  const total = res.total || 0;
  document.getElementById("face-total-text").textContent    = `${total} total detection${total !== 1 ? "s" : ""}`;
  document.getElementById("face-count-badge").textContent   = total;

  const rows = (res.data || []).filter((r) =>
    !name || (r.detected_name || "").toLowerCase().includes(name.toLowerCase())
  );

  const tbody = document.getElementById("face-logs-body");
  if (rows.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5"><div class="empty-state" style="padding:var(--sp-2xl)">
      <div class="empty-state-icon">📋</div>
      <div class="empty-state-title">No face detection logs</div>
    </div></td></tr>`;
    return;
  }

  tbody.innerHTML = rows.map((r) => {
    const conf    = r.confidence != null ? `${Math.round(r.confidence * 100)}%` : "—";
    const confPct = r.confidence != null ? Math.round(r.confidence * 100) : 0;
    const snap    = r.snapshot_url
      ? `<img class="snap-thumb" src="${r.snapshot_url}" alt="snap" onclick="openSnapModal('${r.snapshot_url}')">`
      : `<span class="text-muted">—</span>`;
    return `
      <tr>
        <td class="mono text-sm">${(r.timestamp || "").replace("T", " ").slice(0, 19)}</td>
        <td><strong>${r.detected_name || "Unknown"}</strong></td>
        <td>
          ${conf}
          <span class="conf-bar"><span class="conf-fill" style="width:${confPct}%; background:${confPct > 70 ? "var(--red)" : "var(--yellow)"}"></span></span>
        </td>
        <td class="text-muted text-sm">${r.camera_id || "—"}</td>
        <td>${snap}</td>
      </tr>`;
  }).join("");

  buildPagination("face-pagination", faceLogPage, Math.ceil(total / LOG_LIMIT), (p) => {
    faceLogPage = p;
    loadFaceLogs();
  });
}

// ── Weapon logs ───────────────────────────────────────────────────────────────
async function loadWeaponLogs() {
  weaponLogPage = weaponLogPage || 1;
  const offset = (weaponLogPage - 1) * LOG_LIMIT;
  const res    = await apiGet(`/api/logs/weapons?limit=${LOG_LIMIT}&offset=${offset}`);
  if (!res.success) return;

  const total = res.total || 0;
  document.getElementById("weapon-total-text").textContent  = `${total} total detection${total !== 1 ? "s" : ""}`;
  document.getElementById("weapon-count-badge").textContent = total;

  const tbody = document.getElementById("weapon-logs-body");
  const rows  = res.data || [];

  if (rows.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6"><div class="empty-state" style="padding:var(--sp-2xl)">
      <div class="empty-state-icon">🔫</div>
      <div class="empty-state-title">No weapon detection logs</div>
    </div></td></tr>`;
    return;
  }

  tbody.innerHTML = rows.map((r) => {
    const conf         = r.max_confidence != null ? `${Math.round(r.max_confidence * 100)}%` : "—";
    const threatClass  = `threat-${r.threat_level || ""}`;
    const threatBadge  = { CRITICAL: "badge-red", HIGH: "badge-orange", MEDIUM: "badge-yellow" }[r.threat_level] || "badge-muted";
    const snap         = r.snapshot_url
      ? `<img class="snap-thumb" src="${r.snapshot_url}" alt="snap" onclick="openSnapModal('${r.snapshot_url}')">`
      : `<span class="text-muted">—</span>`;
    return `
      <tr>
        <td class="mono text-sm">${(r.timestamp || "").replace("T", " ").slice(0, 19)}</td>
        <td><strong>${r.weapon_types || "Unknown"}</strong></td>
        <td class="${threatClass}">${conf}</td>
        <td><span class="badge ${threatBadge}">${r.threat_level || "—"}</span></td>
        <td class="text-muted text-sm">${r.camera_id || "—"}</td>
        <td>${snap}</td>
      </tr>`;
  }).join("");

  buildPagination("weapon-pagination", weaponLogPage, Math.ceil(total / LOG_LIMIT), (p) => {
    weaponLogPage = p;
    loadWeaponLogs();
  });
}

// ── Snapshot lightbox ─────────────────────────────────────────────────────────
function openSnapModal(url) {
  document.getElementById("snap-modal-img").src = url;
  document.getElementById("snap-modal").classList.remove("hidden");
}
function closeSnapModal() {
  document.getElementById("snap-modal").classList.add("hidden");
}
document.getElementById("snap-modal")?.addEventListener("click", (e) => {
  if (e.target === e.currentTarget) closeSnapModal();
});

// ── CSV Export ────────────────────────────────────────────────────────────────
function exportLogs(type) {
  window.location.href = `/api/logs/export?type=${type}`;
  showToast(`Downloading ${type} logs as CSV`, "info");
}

// ── Real-time: new alerts → reload active tab ─────────────────────────────────
window.onFaceAlert = function () {
  if (currentTab === "faces") setTimeout(loadFaceLogs, 1500);
};
window.onWeaponAlert = function () {
  if (currentTab === "weapons") setTimeout(loadWeaponLogs, 1500);
};

// ── Init ──────────────────────────────────────────────────────────────────────
loadFaceLogs();
loadWeaponLogs(); // Pre-load count for weapon badge
