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
const outputGuideTab = document.querySelector("#output-guide-tab");
const inputGuideTab = document.querySelector("#input-guide-tab");
const outputGuidePanel = document.querySelector("#output-guide-panel");
const inputGuidePanel = document.querySelector("#input-guide-panel");
const langEnButton = document.querySelector("#lang-en");
const langIdButton = document.querySelector("#lang-id");

let currentLang = "en";

const uiText = {
  en: {
    brandSubtitle: "AI decision support for lemuru cold-chain lots",
    lotId: "Lot ID",
    runAi: "Run AI",
    exportAll: "Export All Predictions",
    uploadCsv: "Upload CSV Dataset",
    mergeUploaded: "Merge uploaded rows into local dataset",
    predictUploaded: "Predict Uploaded CSV",
    matchingLots: "Matching Lots",
    newLotId: "New Lot ID",
    lotMass: "Lot Mass kg",
    handlingScenario: "Handling Scenario",
    targetMarket: "Target Market Node",
    meanTemp: "Mean Temp C",
    maxTempInput: "Max Temp C",
    hoursAbove4: "Hours > 4 C",
    hoursAbove10: "Hours > 10 C",
    hoursAbove15: "Hours > 15 C",
    remainingQualityWindow: "Remaining Quality Window h",
    visualProxyScore: "Visual Proxy Score 0-16",
    photoOptional: "Photo optional",
    saveManual: "Save this manual lot to local dataset",
    predictNewLot: "Predict New Lot",
    conditionLabel: "Condition",
    riskScoreLabel: "Risk Score",
    recommendationLabel: "Recommendation",
    actionProbability: "Action Probability",
    conditionProbability: "Condition Probability",
    keySignals: "Key Signals",
    remainingSignal: "Remaining quality window",
    maxTempSignal: "Max temperature",
    above10Signal: "Time above 10 C",
    visualProxySignal: "Visual proxy score",
    scenarioSignal: "Handling scenario",
    visualAnomaly: "Visual Anomaly",
    anomalyScore: "Anomaly score",
    outputGuide: "Output Guide",
    inputGuide: "Manual Input Guide",
    ready: "Ready",
    noPhoto: "No photo selected",
    noDataset: "No dataset selected",
    chooseCsv: "Choose a CSV dataset first",
    uploadedRunning: "Running uploaded dataset predictions...",
    uploadedFailed: "Uploaded dataset prediction failed",
    uploadedComplete: "Uploaded dataset export complete",
    uploadedMerged: (rows) => `Uploaded dataset export complete; merged ${rows} rows`,
    runningInference: "Running inference...",
    inferenceFailed: "Inference failed",
    inferenceComplete: "Inference complete",
    runningManual: "Running manual inference...",
    manualFailed: "Manual inference failed",
    manualComplete: "Manual inference complete",
    manualSaved: "Manual inference complete and saved",
    lotRequired: "Lot ID is required",
    exportingAll: "Exporting all dataset predictions...",
    exportStarted: "Export started",
    claimNote: "Decision-support output only. Not food-safety, SNI, histamine, or export certification.",
    visualThresholdNote: (check, anomaly) => `Check threshold ${check.toFixed(4)}. Anomaly threshold ${anomaly.toFixed(4)}.`,
  },
  id: {
    brandSubtitle: "Dukungan keputusan AI untuk lot ikan lemuru dalam rantai dingin",
    lotId: "Lot ID (Kode Lot)",
    runAi: "Jalankan AI",
    exportAll: "Export All Predictions (Ekspor Semua Prediksi)",
    uploadCsv: "Upload CSV Dataset (Unggah Dataset CSV)",
    mergeUploaded: "Gabungkan baris upload ke dataset lokal",
    predictUploaded: "Predict Uploaded CSV (Prediksi CSV Upload)",
    matchingLots: "Matching Lots (Lot yang Cocok)",
    newLotId: "New Lot ID (Kode Lot Baru)",
    lotMass: "Lot Mass kg (Berat Lot kg)",
    handlingScenario: "Handling Scenario (Skenario Penanganan)",
    targetMarket: "Target Market Node (Tujuan Pasar)",
    meanTemp: "Mean Temp C (Suhu Rata-rata C)",
    maxTempInput: "Max Temp C (Suhu Maksimum C)",
    hoursAbove4: "Hours > 4 C (Jam di atas 4 C)",
    hoursAbove10: "Hours > 10 C (Jam di atas 10 C)",
    hoursAbove15: "Hours > 15 C (Jam di atas 15 C)",
    remainingQualityWindow: "Remaining Quality Window h (Sisa Waktu Kualitas jam)",
    visualProxyScore: "Visual Proxy Score 0-16 (Skor Visual 0-16)",
    photoOptional: "Photo optional (Foto opsional)",
    saveManual: "Simpan lot manual ini ke dataset lokal",
    predictNewLot: "Predict New Lot (Prediksi Lot Baru)",
    conditionLabel: "Condition (Kondisi)",
    riskScoreLabel: "Risk Score (Skor Risiko)",
    recommendationLabel: "Recommendation (Rekomendasi)",
    actionProbability: "Action Probability (Probabilitas Aksi)",
    conditionProbability: "Condition Probability (Probabilitas Kondisi)",
    keySignals: "Key Signals (Sinyal Utama)",
    remainingSignal: "Remaining quality window (Sisa waktu kualitas)",
    maxTempSignal: "Max temperature (Suhu maksimum)",
    above10Signal: "Time above 10 C (Waktu di atas 10 C)",
    visualProxySignal: "Visual proxy score (Skor visual)",
    scenarioSignal: "Handling scenario (Skenario penanganan)",
    visualAnomaly: "Visual Anomaly (Anomali Visual)",
    anomalyScore: "Anomaly score (Skor anomali)",
    outputGuide: "Output Guide (Panduan Output)",
    inputGuide: "Manual Input Guide (Panduan Input Manual)",
    ready: "Siap",
    noPhoto: "Belum ada foto",
    noDataset: "Belum ada dataset",
    chooseCsv: "Pilih dataset CSV dulu",
    uploadedRunning: "Menjalankan prediksi dataset upload...",
    uploadedFailed: "Prediksi dataset upload gagal",
    uploadedComplete: "Ekspor dataset upload selesai",
    uploadedMerged: (rows) => `Ekspor dataset upload selesai; ${rows} baris digabung`,
    runningInference: "Menjalankan inferensi...",
    inferenceFailed: "Inferensi gagal",
    inferenceComplete: "Inferensi selesai",
    runningManual: "Menjalankan inferensi manual...",
    manualFailed: "Inferensi manual gagal",
    manualComplete: "Inferensi manual selesai",
    manualSaved: "Inferensi manual selesai dan tersimpan",
    lotRequired: "Lot ID wajib diisi",
    exportingAll: "Mengekspor semua prediksi dataset...",
    exportStarted: "Ekspor dimulai",
    claimNote: "Output ini hanya untuk dukungan keputusan. Bukan sertifikasi keamanan pangan, SNI, histamin, atau ekspor.",
    visualThresholdNote: (check, anomaly) => `Ambang CHECK ${check.toFixed(4)}. Ambang ANOMALOUS ${anomaly.toFixed(4)}.`,
  },
};

const guideText = {
  en: {
    output: [
      {
        title: "Condition",
        body: "AI classification output for lot condition. In this MVP, the synthetic condition label is derived mainly from Visual Proxy Score, while the model still receives the full tabular feature set.",
        bullets: [
          ["NORMAL", "visual proxy 0-5, generally normal."],
          ["CHECK", "visual proxy 6-10, needs attention."],
          ["POOR", "visual proxy 11-16, higher concern."],
        ],
      },
      {
        title: "Risk Score",
        body: "Predicted quality downgrade risk from the AI model. Higher means more urgent.",
        bullets: [
          ["0-30", "lower risk."],
          ["31-60", "watch / moderate risk."],
          ["61-100", "high risk, prioritize action."],
        ],
      },
      {
        title: "Recommendation",
        body: "Operational action suggested by the model.",
        bullets: [
          ["HOLD_CHILLED", "keep chilled, no urgent action."],
          ["RE_ICE_AND_ROUTE", "add ice / improve cold-chain, then route."],
          ["PROCESS_IMMEDIATELY", "process quickly."],
          ["MANUAL_INSPECTION", "human inspection recommended."],
          ["DOMESTIC_CHILLED", "route as chilled domestic lot."],
          ["PROCESS_EXPORT_CANDIDATE", "candidate for processing/export scenario."],
          ["LOCAL_FRESH_ONLY", "local fresh route only."],
        ],
      },
      {
        title: "Probability",
        body: "Model confidence for the top condition or action. It is not a food-safety probability.",
      },
      {
        title: "Key Signals",
        body: "Input factors that help explain the prediction.",
        bullets: [
          ["Remaining quality window", "estimated hours before quality window runs out."],
          ["Max temperature", "highest recorded temperature."],
          ["Time above 10 C", "duration of stronger thermal exposure."],
          ["Visual proxy score", "manual/synthetic visual score from 0 normal to 16 poor."],
        ],
      },
      {
        title: "Visual Anomaly",
        body: "Photo-based comparison against normal lemuru reference images.",
        bullets: [
          ["NORMAL", "similar to reference photos."],
          ["CHECK", "somewhat different; inspect if needed."],
          ["ANOMALOUS", "noticeably different from reference photos."],
        ],
      },
    ],
    input: [
      { title: "Lot ID", body: "Unique name/code for the new fish lot. Example: NEW-LEMURU-001." },
      { title: "Lot Mass kg", body: "Total mass of the lot in kilograms. This helps represent shipment scale." },
      { title: "Handling Scenario", body: "Current handling condition, such as controlled ice, delayed chilling, melted ice, or loading excursion." },
      { title: "Target Market Node", body: "Destination scenario code used by the MVP dataset. It represents where the lot is planned to go." },
      { title: "Mean Temp C", body: "Average product temperature from logger/sensor history. Higher average temperature can increase quality risk." },
      { title: "Max Temp C", body: "Highest recorded product temperature. This captures possible temperature excursions." },
      { title: "Hours > 4 C", body: "Total time above 4 C. This indicates mild cold-chain exposure for chilled fish." },
      { title: "Hours > 10 C", body: "Total time above 10 C. This is a stronger warning signal than time above 4 C." },
      { title: "Hours > 15 C", body: "Total time above 15 C. Larger values usually indicate severe cold-chain abuse." },
      { title: "Remaining Quality Window h", body: "Estimated hours left before quality window becomes too short. Smaller values mean higher urgency." },
      {
        title: "Visual Proxy Score 0-16",
        body: "Structured visual/sensory score. Use 5 as default if unknown.",
        bullets: [
          ["0-5", "normal reference."],
          ["6-10", "check / moderate concern."],
          ["11-16", "poor / high concern."],
        ],
      },
      { title: "Photo Optional", body: "Uploaded photo is used for visual anomaly screening only. It does not replace visual proxy score in the XGBoost model." },
      { title: "Save This Manual Lot", body: "If checked, the manual row is appended to the local CSV dataset used by Dataset Mode." },
    ],
  },
  id: {
    output: [
      {
        title: "Condition (Kondisi)",
        body: "Output klasifikasi AI untuk kondisi lot. Pada MVP ini, label kondisi sintetik terutama diturunkan dari Visual Proxy Score, sementara model tetap menerima seluruh fitur tabular.",
        bullets: [
          ["NORMAL", "visual proxy 0-5, kondisi umumnya masih normal."],
          ["CHECK", "visual proxy 6-10, perlu perhatian atau pengecekan."],
          ["POOR", "visual proxy 11-16, risiko kualitas lebih tinggi."],
        ],
      },
      {
        title: "Risk Score (Skor Risiko)",
        body: "Prediksi risiko penurunan kualitas dari model AI. Semakin tinggi nilainya, semakin mendesak lot perlu ditangani.",
        bullets: [
          ["0-30", "risiko rendah."],
          ["31-60", "risiko sedang, perlu dipantau."],
          ["61-100", "risiko tinggi, prioritaskan tindakan."],
        ],
      },
      {
        title: "Recommendation (Rekomendasi)",
        body: "Aksi operasional yang disarankan oleh model untuk lot ikan.",
        bullets: [
          ["HOLD_CHILLED", "tetap simpan dingin, belum perlu tindakan mendesak."],
          ["RE_ICE_AND_ROUTE", "tambahkan es atau perbaiki rantai dingin, lalu kirim."],
          ["PROCESS_IMMEDIATELY", "segera proses lot ikan."],
          ["MANUAL_INSPECTION", "perlu inspeksi manual oleh operator."],
          ["DOMESTIC_CHILLED", "arahkan sebagai lot dingin untuk pasar domestik."],
          ["PROCESS_EXPORT_CANDIDATE", "kandidat untuk skenario proses/ekspor."],
          ["LOCAL_FRESH_ONLY", "hanya cocok untuk jalur segar lokal."],
        ],
      },
      {
        title: "Probability (Probabilitas)",
        body: "Tingkat keyakinan model untuk kondisi atau aksi teratas. Nilai ini bukan probabilitas keamanan pangan.",
      },
      {
        title: "Key Signals (Sinyal Utama)",
        body: "Faktor input yang membantu menjelaskan hasil prediksi.",
        bullets: [
          ["Remaining quality window", "perkiraan jam tersisa sebelum jendela kualitas habis."],
          ["Max temperature", "suhu tertinggi yang tercatat."],
          ["Time above 10 C", "durasi paparan suhu yang lebih kuat."],
          ["Visual proxy score", "skor visual/sensori manual atau sintetik dari 0 normal sampai 16 buruk."],
        ],
      },
      {
        title: "Visual Anomaly (Anomali Visual)",
        body: "Perbandingan foto upload dengan referensi foto lemuru normal.",
        bullets: [
          ["NORMAL", "mirip dengan foto referensi."],
          ["CHECK", "cukup berbeda, sebaiknya dicek."],
          ["ANOMALOUS", "berbeda signifikan dari referensi."],
        ],
      },
    ],
    input: [
      { title: "Lot ID (Kode Lot)", body: "Kode unik untuk lot ikan baru. Contoh: NEW-LEMURU-001." },
      { title: "Lot Mass kg (Berat Lot kg)", body: "Total berat lot dalam kilogram. Dipakai untuk merepresentasikan skala pengiriman." },
      { title: "Handling Scenario (Skenario Penanganan)", body: "Kondisi penanganan saat ini, misalnya controlled ice, delayed chilling, melted ice, atau loading excursion." },
      { title: "Target Market Node (Tujuan Pasar)", body: "Kode tujuan pada dataset MVP. Ini mewakili rencana tujuan pengiriman lot." },
      { title: "Mean Temp C (Suhu Rata-rata C)", body: "Suhu rata-rata produk dari riwayat sensor/logger. Semakin tinggi nilainya, risiko kualitas bisa meningkat." },
      { title: "Max Temp C (Suhu Maksimum C)", body: "Suhu tertinggi yang tercatat. Nilai ini menangkap kemungkinan kejadian suhu naik mendadak." },
      { title: "Hours > 4 C (Jam di atas 4 C)", body: "Total durasi suhu produk berada di atas 4 C. Ini menunjukkan paparan ringan pada rantai dingin." },
      { title: "Hours > 10 C (Jam di atas 10 C)", body: "Total durasi suhu produk berada di atas 10 C. Ini sinyal peringatan yang lebih kuat dibanding di atas 4 C." },
      { title: "Hours > 15 C (Jam di atas 15 C)", body: "Total durasi suhu produk berada di atas 15 C. Nilai besar biasanya menunjukkan gangguan rantai dingin yang berat." },
      { title: "Remaining Quality Window h (Sisa Waktu Kualitas jam)", body: "Perkiraan jam tersisa sebelum kualitas terlalu menurun. Nilai makin kecil berarti makin mendesak." },
      {
        title: "Visual Proxy Score 0-16 (Skor Visual 0-16)",
        body: "Skor visual/sensori terstruktur. Jika belum tahu nilainya, MVP memakai 5 sebagai asumsi awal.",
        bullets: [
          ["0-5", "normal atau mendekati referensi."],
          ["6-10", "perlu dicek, kekhawatiran sedang."],
          ["11-16", "buruk atau kekhawatiran tinggi."],
        ],
      },
      { title: "Photo Optional (Foto Opsional)", body: "Foto yang diupload dipakai hanya untuk screening anomali visual. Foto tidak menggantikan visual proxy score pada model XGBoost." },
      { title: "Save This Manual Lot (Simpan Lot Manual)", body: "Jika dicentang, baris input manual akan ditambahkan ke CSV lokal yang dipakai oleh Dataset Mode." },
    ],
  },
};

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

function t(key, ...args) {
  const value = uiText[currentLang][key] || uiText.en[key] || key;
  return typeof value === "function" ? value(...args) : value;
}

function renderGuideCards(container, cards) {
  container.innerHTML = "";
  for (const card of cards) {
    const article = document.createElement("article");
    const title = document.createElement("h3");
    const body = document.createElement("p");

    title.textContent = card.title;
    body.textContent = card.body;
    article.append(title, body);

    if (card.bullets) {
      const list = document.createElement("ul");
      for (const [term, description] of card.bullets) {
        const item = document.createElement("li");
        const strong = document.createElement("strong");
        strong.textContent = term;
        item.append(strong, `: ${description}`);
        list.appendChild(item);
      }
      article.appendChild(list);
    }

    container.appendChild(article);
  }
}

function refreshKnownStatusText() {
  if (!statusEl) {
    return;
  }

  const statusKeys = [
    "ready",
    "chooseCsv",
    "uploadedRunning",
    "uploadedFailed",
    "uploadedComplete",
    "runningInference",
    "inferenceFailed",
    "inferenceComplete",
    "runningManual",
    "manualFailed",
    "manualComplete",
    "manualSaved",
    "lotRequired",
    "exportingAll",
    "exportStarted",
  ];

  for (const key of statusKeys) {
    const english = uiText.en[key];
    const indonesia = uiText.id[key];
    if (typeof english === "string" && typeof indonesia === "string") {
      if (statusEl.textContent === english || statusEl.textContent === indonesia) {
        statusEl.textContent = t(key);
        return;
      }
    }
  }
}

function applyLanguage(lang) {
  currentLang = lang;
  document.documentElement.lang = lang === "id" ? "id" : "en";

  for (const element of document.querySelectorAll("[data-i18n]")) {
    element.textContent = t(element.dataset.i18n);
  }

  langEnButton.classList.toggle("active", lang === "en");
  langIdButton.classList.toggle("active", lang === "id");
  renderGuideCards(outputGuidePanel, guideText[lang].output);
  renderGuideCards(inputGuidePanel, guideText[lang].input);
  updatePhotoLabel();
  updateBulkFileLabel();
  fields.claimNote.textContent = t("claimNote");
  refreshKnownStatusText();
}

function setStatus(message, isError = false) {
  if (!statusEl) {
    return;
  }

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
  fields.claimNote.textContent = currentLang === "en" ? result.claim_note : t("claimNote");
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
      ? t("visualThresholdNote", upload.check_threshold, upload.anomaly_threshold)
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
  photoFileName.textContent = file ? file.name : t("noPhoto");
  clearPhotoButton.hidden = !file;
}

function updateBulkFileLabel() {
  const file = bulkDatasetInput.files && bulkDatasetInput.files[0];
  bulkFileName.textContent = file ? file.name : t("noDataset");
  clearBulkFileButton.hidden = !file;
}

async function exportUploadedDataset() {
  const file = bulkDatasetInput.files && bulkDatasetInput.files[0];
  if (!file) {
    setStatus(t("chooseCsv"), true);
    return;
  }

  setStatus(t("uploadedRunning"));
  const formData = new FormData();
  formData.append("dataset", file);
  formData.append("merge_to_dataset", mergeUploadedDataset.checked ? "true" : "false");

  const response = await fetch("/api/export/uploaded-predictions.csv", {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: t("uploadedFailed") }));
    throw new Error(error.detail || t("uploadedFailed"));
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
  setStatus(savedRows > 0 ? t("uploadedMerged", savedRows) : t("uploadedComplete"));
  if (savedRows > 0) {
    loadLots().catch(() => {});
  }
}

async function predictLot(lotId) {
  setStatus(t("runningInference"));
  const response = await fetch(`/api/predict/${encodeURIComponent(lotId)}`);
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: t("inferenceFailed") }));
    throw new Error(error.detail || t("inferenceFailed"));
  }
  const result = await response.json();
  renderPrediction(result);
  setStatus(t("inferenceComplete"));
}

async function predictManual(formData) {
  setStatus(t("runningManual"));
  const response = await fetch("/api/predict/manual", {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: t("manualFailed") }));
    throw new Error(error.detail || t("manualFailed"));
  }
  const result = await response.json();
  renderPrediction(result);
  setStatus(result.saved_to_dataset ? t("manualSaved") : t("manualComplete"));
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

function setGuideMode(mode) {
  const input = mode === "input";
  inputGuidePanel.hidden = !input;
  outputGuidePanel.hidden = input;
  inputGuidePanel.style.display = input ? "" : "none";
  outputGuidePanel.style.display = input ? "none" : "";
  inputGuideTab.classList.toggle("active", input);
  outputGuideTab.classList.toggle("active", !input);
  inputGuideTab.setAttribute("aria-selected", String(input));
  outputGuideTab.setAttribute("aria-selected", String(!input));
}

async function loadLots() {
  const params = new URLSearchParams({ limit: "12" });
  const query = lotInput.value.trim();
  if (query) {
    params.set("q", query);
  }

  const response = await fetch(`/api/lots?${params.toString()}`);
  if (!response.ok) {
    throw new Error(currentLang === "id" ? "Gagal memuat lot" : "Failed to load lots");
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
    setStatus(t("lotRequired"), true);
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
  setStatus(t("exportingAll"));
  window.location.href = "/api/export/predictions.csv";
  window.setTimeout(() => setStatus(t("exportStarted")), 500);
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
outputGuideTab.addEventListener("click", () => setGuideMode("output"));
inputGuideTab.addEventListener("click", () => setGuideMode("input"));
langEnButton.addEventListener("click", () => applyLanguage("en"));
langIdButton.addEventListener("click", () => applyLanguage("id"));

applyLanguage("en");

loadLots()
  .then(() => predictLot(lotInput.value))
  .catch((error) => setStatus(error.message, true));
