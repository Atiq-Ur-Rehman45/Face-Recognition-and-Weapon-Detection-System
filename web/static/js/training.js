/** training.js — Training page logic */

socket.on("training:progress", (data) => {
  document.getElementById("train-status-msg").textContent = data.message || "Training...";
  document.getElementById("train-prog-bar").style.width   = `${data.percent || 0}%`;
  appendTrainingLog(`[${data.status?.toUpperCase()}] ${data.message}`);
});

socket.on("training:complete", (data) => {
  const progSection = document.getElementById("training-progress-section");
  const successSection = document.getElementById("training-success");
  const trainBtn = document.getElementById("btn-train");

  progSection.style.display = "none";
  trainBtn.disabled = false;

  if (data.success) {
    document.getElementById("train-done-msg").textContent =
      `Completed in ${data.elapsed_seconds}s · ${data.persons_trained} person(s) trained`;
    successSection.style.display = "block";
    showToast("🧠 Model trained successfully!", "success", 5000);
    setSidebarStatus(true, true);
  } else {
    showToast("Training failed: " + (data.error || "Unknown error"), "error");
    appendTrainingLog(`[ERROR] ${data.error || "Training failed"}`);
  }
});

async function startTraining() {
  const btn = document.getElementById("btn-train");
  btn.disabled = true;

  document.getElementById("training-progress-section").style.display = "block";
  document.getElementById("training-success").style.display          = "none";
  document.getElementById("training-log").textContent                = "";
  document.getElementById("train-prog-bar").style.width              = "0%";
  document.getElementById("train-status-msg").textContent            = "Starting...";

  const res = await apiPost("/api/training/start");
  if (!res.success) {
    showToast(res.error || "Training failed to start", "error");
    btn.disabled = false;
    document.getElementById("training-progress-section").style.display = "none";
    return;
  }

  appendTrainingLog("[INFO] Training job submitted...");
  showToast("🧠 Training started", "info");
}

function appendTrainingLog(message) {
  const log = document.getElementById("training-log");
  if (!log) return;
  const ts   = new Date().toLocaleTimeString("en-GB", { hour12: false });
  log.textContent += `[${ts}] ${message}\n`;
  log.scrollTop    = log.scrollHeight;
}
