const form = document.querySelector("#predict-form");
const manualForm = document.querySelector("#manual-form");
const lotInput = document.querySelector("#lot-id");
const statusEl = document.querySelector("#connection-status");
const sampleButtons = document.querySelector("#sample-buttons");
const lotOptions = document.querySelector("#lot-options");
const exportButton = document.querySelector("#export-button");
const bulkDatasetInput = document.querySelector("#bulk-dataset");
const bulkFileName = document.querySelector("#bulk-file-name");
const clearBulkFileButton = document.querySelector("#clear-bulk-file");
const bulkExportButton = document.querySelector("#bulk-export-button");
const mergeUploadedDataset = document.querySelector("#merge-uploaded-dataset");
const datasetTab = document.querySelector("#dataset-tab");
const manualTab = document.querySelector("#manual-tab");
const datasetPanel = document.querySelector("#dataset-panel");
const manualPanel = document.querySelector("#manual-panel");
const photoInput = document.querySelector("#photo");
const photoFileName = document.querySelector("#photo-file-name");
const clearPhotoButton = document.querySelector("#clear-photo");

const fields = {
  condition: document.querySelector("#condition"),
  riskScore: document.querySelector("#risk-score"),
  action: document.querySelector("#action"),
  actionProbs: document.querySelector("#action-probs"),
  conditionProbs: document.querySelector("#condition-probs"),
  remainingWindow: document.querySelector("#remaining-window"),
  maxTemp: document.querySelector("#max-temp"),
  above10: document.querySelector("#above-10"),
  visualProxy: document.querySelector("#visual-proxy"),
  scenario: document.querySelector("#scenario"),
  claimNote: document.querySelector("#claim-note"),
  visualUploadSection: document.querySelector("#visual-upload-section"),
  visualUploadCard: document.querySelector("#visual-upload-card"),
  visualUploadFile: document.querySelector("#visual-upload-file"),
  visualUploadStatus: document.querySelector("#visual-upload-status"),
  visualUploadScore: document.querySelector("#visual-upload-score"),
  visualUploadNote: document.querySelector("#visual-upload-note"),
};

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("is-error", isError);
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function renderBars(container, items) {
  container.innerHTML = "";
  for (const item of items) {
    const row = document.createElement("div");
    row.className = "bar-row";

    const label = document.createElement("span");
    label.textContent = item.label;

    const track = document.createElement("div");
    track.className = "bar-track";

    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = formatPercent(item.probability);
    track.appendChild(fill);

    const value = document.createElement("strong");
    value.textContent = formatPercent(item.probability);

    row.append(label, track, value);
    container.appendChild(row);
  }
}

function renderPrediction(result) {
  const ref = result.dataset_reference;
  fields.condition.textContent = result.predicted_condition;
  fields.riskScore.textContent = `${result.predicted_risk_score_0_100.toFixed(1)} / 100`;
  fields.action.textContent = result.predicted_action;
  fields.remainingWindow.textContent = `${ref.remaining_quality_window_h.toFixed(1)} h`;
  fields.maxTemp.textContent = `${ref.max_temp_c.toFixed(1)} C`;
  fields.above10.textContent = `${ref.time_above_10c_h.toFixed(1)} h`;
  fields.visualProxy.textContent = `${ref.proxy_visual_score_0_16.toFixed(0)} / 16`;
  fields.scenario.textContent = ref.handling_scenario;
  fields.claimNote.textContent = result.claim_note;
  renderBars(fields.actionProbs, result.action_probabilities);
  renderBars(fields.conditionProbs, result.condition_probabilities);

  if (result.visual_upload) {
    fields.visualUploadSection.hidden = false;
    const upload = result.visual_upload;
    const hasScore = typeof upload.anomaly_score === "number";
    fields.visualUploadCard.className = `visual-card visual-${(upload.status || "unknown").toLowerCase()}`;
    fields.visualUploadFile.textContent = upload.filename || "Uploaded photo";
    fields.visualUploadStatus.textContent = upload.status || "PHOTO_RECEIVED";
    fields.visualUploadScore.textContent = hasScore ? upload.anomaly_score.toFixed(4) : "-";
    fields.visualUploadNote.textContent = hasScore
      ? `Check threshold ${upload.check_threshold.toFixed(4)}. Anomaly threshold ${upload.anomaly_threshold.toFixed(4)}.`
      : upload.note || "";
  } else {
    fields.visualUploadSection.hidden = true;
    fields.visualUploadCard.className = "visual-card";
    fields.visualUploadFile.textContent = "-";
    fields.visualUploadStatus.textContent = "-";
    fields.visualUploadScore.textContent = "-";
    fields.visualUploadNote.textContent = "";
  }
}

function updatePhotoLabel() {
  const file = photoInput.files && photoInput.files[0];
  photoFileName.textContent = file ? file.name : "No photo selected";
  clearPhotoButton.hidden = !file;
}

function updateBulkFileLabel() {
  const file = bulkDatasetInput.files && bulkDatasetInput.files[0];
  bulkFileName.textContent = file ? file.name : "No dataset selected";
  clearBulkFileButton.hidden = !file;
}

async function exportUploadedDataset() {
  const file = bulkDatasetInput.files && bulkDatasetInput.files[0];
  if (!file) {
    setStatus("Choose a CSV dataset first", true);
    return;
  }

  setStatus("Running uploaded dataset predictions...");
  const formData = new FormData();
  formData.append("dataset", file);
  formData.append("merge_to_dataset", mergeUploadedDataset.checked ? "true" : "false");

  const response = await fetch("/api/export/uploaded-predictions.csv", {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Uploaded dataset prediction failed" }));
    throw new Error(error.detail || "Uploaded dataset prediction failed");
  }

  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "uploaded_chillfish_predictions.csv";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  const savedRows = Number(response.headers.get("X-Saved-Rows") || 0);
  setStatus(savedRows > 0 ? `Uploaded dataset export complete; merged ${savedRows} rows` : "Uploaded dataset export complete");
  if (savedRows > 0) {
    loadLots().catch(() => {});
  }
}

async function predictLot(lotId) {
  setStatus("Running inference...");
  const response = await fetch(`/api/predict/${encodeURIComponent(lotId)}`);
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Inference failed" }));
    throw new Error(error.detail || "Inference failed");
  }
  const result = await response.json();
  renderPrediction(result);
  setStatus("Inference complete");
}

async function predictManual(formData) {
  setStatus("Running manual inference...");
  const response = await fetch("/api/predict/manual", {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Manual inference failed" }));
    throw new Error(error.detail || "Manual inference failed");
  }
  const result = await response.json();
  renderPrediction(result);
  setStatus(result.saved_to_dataset ? "Manual inference complete and saved" : "Manual inference complete");
  if (result.saved_to_dataset) {
    loadLots().catch(() => {});
  }
}

function setMode(mode) {
  const manual = mode === "manual";
  manualPanel.hidden = !manual;
  datasetPanel.hidden = manual;
  manualTab.classList.toggle("active", manual);
  datasetTab.classList.toggle("active", !manual);
}

async function loadLots() {
  const params = new URLSearchParams({ limit: "12" });
  const query = lotInput.value.trim();
  if (query) {
    params.set("q", query);
  }

  const response = await fetch(`/api/lots?${params.toString()}`);
  if (!response.ok) {
    throw new Error("Failed to load lots");
  }
  const data = await response.json();
  sampleButtons.innerHTML = "";
  lotOptions.innerHTML = "";
  for (const lot of data.lots) {
    const option = document.createElement("option");
    option.value = lot.lot_id;
    lotOptions.appendChild(option);

    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `${lot.lot_id} - ${lot.condition_status} - ${lot.handling_scenario}`;
    button.addEventListener("click", () => {
      lotInput.value = lot.lot_id;
      predictLot(lot.lot_id).catch((error) => setStatus(error.message, true));
    });
    sampleButtons.appendChild(button);
  }
}

function debounce(callback, delay = 180) {
  let timer;
  return (...args) => {
    window.clearTimeout(timer);
    timer = window.setTimeout(() => callback(...args), delay);
  };
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const lotId = lotInput.value.trim();
  if (!lotId) {
    setStatus("Lot ID is required", true);
    return;
  }
  predictLot(lotId).catch((error) => setStatus(error.message, true));
});

lotInput.addEventListener(
  "input",
  debounce(() => {
    loadLots().catch((error) => setStatus(error.message, true));
  }),
);

exportButton.addEventListener("click", () => {
  setStatus("Exporting all dataset predictions...");
  window.location.href = "/api/export/predictions.csv";
  window.setTimeout(() => setStatus("Export started"), 500);
});

bulkDatasetInput.addEventListener("change", updateBulkFileLabel);

clearBulkFileButton.addEventListener("click", () => {
  bulkDatasetInput.value = "";
  updateBulkFileLabel();
});

bulkExportButton.addEventListener("click", () => {
  exportUploadedDataset().catch((error) => setStatus(error.message, true));
});

manualForm.addEventListener("submit", (event) => {
  event.preventDefault();
  predictManual(new FormData(manualForm)).catch((error) => setStatus(error.message, true));
});

photoInput.addEventListener("change", updatePhotoLabel);

clearPhotoButton.addEventListener("click", () => {
  photoInput.value = "";
  updatePhotoLabel();
});

datasetTab.addEventListener("click", () => setMode("dataset"));
manualTab.addEventListener("click", () => setMode("manual"));

loadLots()
  .then(() => predictLot(lotInput.value))
  .catch((error) => setStatus(error.message, true));

updatePhotoLabel();
updateBulkFileLabel();
