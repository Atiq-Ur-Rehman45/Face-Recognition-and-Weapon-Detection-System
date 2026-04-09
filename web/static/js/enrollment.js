/** enrollment.js — Enrollment page logic */

let enrollCriminalId = null;
let totalTarget      = 20;  // updated from server

// ── Start Enrollment ──────────────────────────────────────────────────────────
async function beginEnrollment() {
  const name = document.getElementById("f-name").value.trim();
  if (!name) {
    showToast("Please enter a full name", "warning");
    document.getElementById("f-name").focus();
    return;
  }

  // Create criminal record first
  const createRes = await apiPost("/api/criminals", {
    name:       name,
    cnic:       document.getElementById("f-cnic").value.trim() || null,
    status:     document.getElementById("f-status").value,
    crime_type: document.getElementById("f-crime").value,
    notes:      document.getElementById("f-notes").value.trim() || null,
  });

  if (!createRes.success) {
    showToast(createRes.error || "Failed to create record", "error");
    return;
  }

  enrollCriminalId = createRes.data.id;

  // Start enrollment camera
  const startRes = await apiPost("/api/enrollment/start", {
    criminal_id: enrollCriminalId,
    camera_index: 0,
  });

  if (!startRes.success) {
    showToast(startRes.error || "Cannot open camera", "error");
    // Clean up the created empty record
    if (enrollCriminalId) {
      await apiDelete(`/api/criminals/${enrollCriminalId}?purge_snapshots=true`);
      enrollCriminalId = null;
    }
    return;
  }

  // Switch UI to enrollment mode
  document.getElementById("btn-start-enroll").style.display  = "none";
  document.getElementById("btn-cancel-enroll").style.display = "inline-flex";
  document.getElementById("stages-card").style.display       = "block";
  document.getElementById("cam-banner").style.display        = "block";
  showToast("📷 Enrollment started — look at the camera", "info", 4000);
}

async function cancelEnrollment() {
  showConfirm(
    "Cancel Enrollment",
    "Are you sure? Partially captured images and the record will be deleted.",
    async () => {
      const cid = enrollCriminalId;
      resetEnrollForm();
      
      // Stop the background camera service AND atomic delete on server
      await apiPost("/api/enrollment/cancel", { criminal_id: cid });
      
      showToast("Enrollment cancelled and record deleted", "warning");
    }
  );
}

function resetEnrollForm() {
  enrollCriminalId = null;
  document.getElementById("f-name").value  = "";
  document.getElementById("f-cnic").value  = "";
  document.getElementById("f-notes").value = "";
  document.getElementById("btn-start-enroll").style.display  = "inline-flex";
  document.getElementById("btn-cancel-enroll").style.display = "none";
  document.getElementById("stages-card").style.display       = "none";
  document.getElementById("cam-banner").style.display        = "none";
  document.getElementById("enroll-complete-card").style.display = "none";
  document.getElementById("stages-list").innerHTML = "";
}

// ── WebSocket: enrollment events ──────────────────────────────────────────────
socket.on("enrollment:progress", (data) => {
  totalTarget = data.total_target;
  const progress = data.total_captured / Math.max(data.total_target, 1);

  // Progress ring
  const ring = document.getElementById("ring-circle");
  const circumference = 201;
  ring.style.strokeDashoffset = circumference * (1 - progress);
  document.getElementById("ring-text").textContent = data.total_captured;

  // Instruction
  document.getElementById("cam-instruction").textContent =
    data.quality_message || `Capturing: ${data.stage}`;

  // Total bar
  document.getElementById("total-progress").style.width = `${Math.round(progress * 100)}%`;
  document.getElementById("prog-count").textContent     = `${data.total_captured} / ${data.total_target} captured`;
  document.getElementById("prog-pct").textContent       = `${Math.round(progress * 100)}%`;

  // Stage items
  if (data.all_stages) {
    renderStages(data.all_stages, data.stage);
  }
});

socket.on("enrollment:stage_change", (data) => {
  document.getElementById("cam-instruction").textContent = `→ ${data.instruction}`;
  showToast(`Next pose: ${data.instruction}`, "info", 3000);
});

socket.on("enrollment:complete", (data) => {
  document.getElementById("btn-start-enroll").style.display  = "inline-flex";
  document.getElementById("btn-cancel-enroll").style.display = "none";
  document.getElementById("enroll-complete-card").style.display = "block";
  document.getElementById("enroll-complete-msg").textContent =
    `Successfully captured ${data.total_images} face images.`;
  document.getElementById("cam-banner").style.display = "none";
  showToast(`✅ Enrollment complete — ${data.total_images} images captured`, "success", 6000);
});

socket.on("enrollment:cancelled", () => {
  showToast("Enrollment cancelled", "warning");
  resetEnrollForm();
});

socket.on("enrollment:error", (data) => {
  showToast(data.message || "Enrollment error", "error");
  // The backend was supposed to fire _enrollment().cancel(), we can also request strict cleanup
  const cid = enrollCriminalId;
  resetEnrollForm();
  if (cid) {
    apiPost("/api/enrollment/cancel", { criminal_id: cid });
  }
});

// ── Render stage progress bars ────────────────────────────────────────────────
function renderStages(stages, currentStage) {
  const list = document.getElementById("stages-list");
  list.innerHTML = stages.map((s) => {
    const pct       = Math.min(100, Math.round((s.captured / s.target) * 100));
    const done      = s.captured >= s.target;
    const isActive  = s.angle === currentStage && !done;
    return `
      <div class="stage-item ${done ? "complete" : ""}">
        <div class="stage-label">${s.angle}</div>
        <div class="progress-wrap" style="flex:1; height:5px">
          <div class="progress-bar ${done ? "green" : ""}" style="width:${pct}%"></div>
        </div>
        <div class="stage-count">${s.captured}/${s.target}</div>
      </div>`;
  }).join("");
}
