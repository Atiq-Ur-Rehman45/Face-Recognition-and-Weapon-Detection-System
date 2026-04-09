/** records.js — Criminal Records CRUD logic */

let allRecords   = [];
let currentPage  = 1;
const PAGE_SIZE  = 10;
let expandedId   = null;
let editingId    = null;

// ── Load records ──────────────────────────────────────────────────────────────
async function loadRecords() {
  const search = document.getElementById("rec-search").value.trim();
  const status = document.getElementById("rec-status-filter").value;
  const url    = `/api/criminals?search=${encodeURIComponent(search)}&status=${encodeURIComponent(status)}&page=${currentPage}&limit=${PAGE_SIZE}`;

  const res = await apiGet(url);
  if (!res.success) { showToast("Failed to load records", "error"); return; }

  allRecords = res.data || [];
  const total = res.total || 0;
  document.getElementById("rec-total-label").textContent = `${total} record${total !== 1 ? "s" : ""}`;

  renderTable(allRecords);

  const totalPages = Math.ceil(total / PAGE_SIZE);
  buildPagination("rec-pagination", currentPage, totalPages, (p) => {
    currentPage = p;
    loadRecords();
  });
}

function onSearch() {
  currentPage = 1;
  loadRecords();
}

// ── Render table ──────────────────────────────────────────────────────────────
function renderTable(records) {
  const tbody = document.getElementById("records-tbody");

  if (records.length === 0) {
    tbody.innerHTML = `<tr><td colspan="9">
      <div class="empty-state" style="padding:var(--sp-2xl)">
        <div class="empty-state-icon">🗂️</div>
        <div class="empty-state-title">No records found</div>
        <div class="empty-state-text">Try a different search or <a href="/enroll" style="color:var(--blue)">enroll a new person</a></div>
      </div></td></tr>`;
    return;
  }

  tbody.innerHTML = records.map((r) => {
    const thumb = r.thumbnail_url
      ? `<img class="avatar" src="${r.thumbnail_url}" alt="${r.name}"/>`
      : `<div class="avatar-placeholder">👤</div>`;
    const date = (r.created_at || "").slice(0, 10);
    const isExpanded = r.id === expandedId;
    return `
      <tr data-id="${r.id}" class="${isExpanded ? "selected" : ""}" onclick="toggleDetail(${r.id})">
        <td class="text-muted mono">${r.id}</td>
        <td>${thumb}</td>
        <td><strong>${r.name}</strong></td>
        <td class="text-muted mono text-sm">${r.cnic || "—"}</td>
        <td>${r.crime_type || "—"}</td>
        <td>${statusBadge(r.status)}</td>
        <td class="mono">${r.image_count}</td>
        <td class="text-muted text-sm">${date}</td>
        <td onclick="event.stopPropagation()">
          <div class="flex gap-sm">
            <button class="btn btn-ghost btn-sm btn-icon" title="Edit" onclick="openEditModal(${r.id})">✏️</button>
            <button class="btn btn-ghost btn-sm btn-icon" title="Delete" onclick="confirmDeleteRecord(${r.id}, '${r.name.replace(/'/g, "\\'")}')">🗑</button>
          </div>
        </td>
      </tr>`;
  }).join("");
}

// ── Expand detail panel ───────────────────────────────────────────────────────
async function toggleDetail(id) {
  const panel = document.getElementById("detail-panel");

  if (expandedId === id) {
    // Collapse
    panel.classList.remove("open");
    expandedId = null;
    document.querySelectorAll("#records-tbody tr").forEach(r => r.classList.remove("selected"));
    return;
  }

  expandedId = id;
  document.querySelectorAll("#records-tbody tr").forEach(r => {
    r.classList.toggle("selected", parseInt(r.dataset.id) === id);
  });

  const res = await apiGet(`/api/criminals/${id}`);
  if (!res.success) { showToast("Failed to load details", "error"); return; }

  const c = res.data;

  // Photos grid
  const photoGrid = document.getElementById("detail-photos");
  photoGrid.innerHTML = c.images.length > 0
    ? c.images.map(url => `<img src="${url}" alt="photo" onclick="openPhotoModal('${url}')"/>`).join("")
    : `<div class="text-muted text-sm">No images</div>`;

  // Fields
  document.getElementById("detail-fields").innerHTML = `
    <div><div class="detail-field-label">Full Name</div><div class="detail-field-value">${c.name}</div></div>
    <div><div class="detail-field-label">CNIC</div><div class="detail-field-value">${c.cnic || "—"}</div></div>
    <div><div class="detail-field-label">Status</div><div class="detail-field-value">${statusBadge(c.status)}</div></div>
    <div><div class="detail-field-label">Crime Type</div><div class="detail-field-value">${c.crime_type || "—"}</div></div>
    <div><div class="detail-field-label">Face Label</div><div class="detail-field-value mono">${c.face_label}</div></div>
    <div><div class="detail-field-label">Enrolled</div><div class="detail-field-value text-sm">${(c.created_at || "").slice(0, 16)}</div></div>
    ${c.notes ? `<div style="grid-column:1/-1"><div class="detail-field-label">Notes</div><div class="detail-field-value text-sm">${c.notes}</div></div>` : ""}
  `;

  document.getElementById("detail-enroll-more-btn").href = `/enroll`;
  document.getElementById("detail-delete-btn").onclick   = () => confirmDeleteRecord(id, c.name);

  // Append panel after the matching row
  const row = document.querySelector(`#records-tbody tr[data-id="${id}"]`);
  if (row) row.after(panel);
  panel.classList.add("open");
}

// ── Edit Modal ────────────────────────────────────────────────────────────────
async function openEditModal(id) {
  editingId = id || expandedId;
  if (!editingId) return;
  const res = await apiGet(`/api/criminals/${editingId}`);
  if (!res.success) { showToast("Failed to load record", "error"); return; }
  const c = res.data;
  document.getElementById("edit-name").value   = c.name   || "";
  document.getElementById("edit-cnic").value   = c.cnic   || "";
  document.getElementById("edit-crime").value  = c.crime_type || "";
  document.getElementById("edit-notes").value  = c.notes  || "";
  document.getElementById("edit-status").value = c.status || "Wanted";
  document.getElementById("edit-modal").classList.remove("hidden");
}

function closeEditModal() {
  document.getElementById("edit-modal").classList.add("hidden");
  editingId = null;
}

async function saveEdit() {
  if (!editingId) return;
  const body = {
    name:       document.getElementById("edit-name").value.trim(),
    cnic:       document.getElementById("edit-cnic").value.trim() || null,
    crime_type: document.getElementById("edit-crime").value.trim(),
    status:     document.getElementById("edit-status").value,
    notes:      document.getElementById("edit-notes").value.trim() || null,
  };
  const res = await apiPut(`/api/criminals/${editingId}`, body);
  if (res.success) {
    showToast("Record updated", "success");
    closeEditModal();
    loadRecords();
  } else {
    showToast(res.error || "Update failed", "error");
  }
}

// ── Delete ────────────────────────────────────────────────────────────────────
function confirmDeleteRecord(id, name) {
  showConfirm(
    "Delete Criminal Record",
    `Are you sure you want to delete "${name}"? This will also delete all enrollment images and trigger a model retrain.`,
    async () => {
      const res = await apiDelete(`/api/criminals/${id}`);
      if (res.success) {
        showToast(`Deleted "${name}" — model retraining in background`, "success");
        expandedId = null;
        document.getElementById("detail-panel").classList.remove("open");
        loadRecords();
      } else {
        showToast(res.error || "Delete failed", "error");
      }
    }
  );
}

// ── Photo lightbox ────────────────────────────────────────────────────────────
function openPhotoModal(url) {
  document.getElementById("photo-modal-img").src = url;
  document.getElementById("photo-modal").classList.remove("hidden");
}
function closePhotoModal() {
  document.getElementById("photo-modal").classList.add("hidden");
}
document.getElementById("photo-modal").addEventListener("click", (e) => {
  if (e.target === e.currentTarget) closePhotoModal();
});

// ── Close edit modal on overlay click ────────────────────────────────────────
document.getElementById("edit-modal").addEventListener("click", (e) => {
  if (e.target === e.currentTarget) closeEditModal();
});

// ── Init ──────────────────────────────────────────────────────────────────────
loadRecords();
