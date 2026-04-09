/** settings.js — Settings page logic */

async function saveSettings() {
  const inputs = document.querySelectorAll(".config-input");
  const payload = {};

  inputs.forEach((input) => {
    const key = input.dataset.key;
    const type = input.dataset.type;
    
    let val;
    if (type === "bool") {
      val = input.checked;
    } else if (type === "float") {
      val = parseFloat(input.value);
    } else if (type === "int") {
      val = parseInt(input.value, 10);
    } else {
      val = input.value;
    }
    
    if (key) {
      payload[key] = val;
    }
  });

  const res = await apiPut("/api/settings", payload);
  if (res.success) {
    showToast("⚙️ Settings saved! Please restart the server for all changes to take effect.", "success", 8000);
  } else {
    showToast(res.error || "Failed to save settings", "error");
  }
}

