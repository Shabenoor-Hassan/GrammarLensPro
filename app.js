let authToken = localStorage.getItem("token") || null;
let currentUsername = localStorage.getItem("username") || null;
let userLevel = localStorage.getItem("english_level") || "intermediate";
let currentErrors = [];
let lastAnalyzedText = ""; // the exact sentence used for the most recent analysis

// Elements
const authModal = document.getElementById("authModal");
const openAuthModalBtn = document.getElementById("openAuthModalBtn");
const closeAuthModal = document.getElementById("closeAuthModal");
const tabLogin = document.getElementById("tabLogin");
const tabRegister = document.getElementById("tabRegister");
const loginForm = document.getElementById("loginForm");
const registerForm = document.getElementById("registerForm");
const userProfile = document.getElementById("userProfile");
const userNameDisplay = document.getElementById("userNameDisplay");
const logoutBtn = document.getElementById("logoutBtn");

const levelSelector = document.getElementById("levelSelector");
const inputText = document.getElementById("inputText");
const checkBtn = document.getElementById("checkBtn");
const clearBtn = document.getElementById("clearBtn");
const wordCount = document.getElementById("wordCount");

// Voice Elements
const voiceBtn = document.getElementById("voiceBtn");
const voiceBtnText = document.getElementById("voiceBtnText");
const voiceStatus = document.getElementById("voiceStatus");
const voiceStatusText = document.getElementById("voiceStatusText");
const voiceTimer = document.getElementById("voiceTimer");
const speakSourceBtn = document.getElementById("speakSourceBtn");
const speakOutputBtn = document.getElementById("speakOutputBtn");

// Metrics
const metricsBar = document.getElementById("metricsBar");
const metricCefr = document.getElementById("metricCefr");
const metricReadability = document.getElementById("metricReadability");
const metricVocab = document.getElementById("metricVocab");
const metricErrors = document.getElementById("metricErrors");

// Tabs & Views
const tabCorrections = document.getElementById("tabCorrections");
const tabVocab = document.getElementById("tabVocab");
const correctionsView = document.getElementById("correctionsView");
const vocabView = document.getElementById("vocabView");
const correctionsBadge = document.getElementById("correctionsBadge");
const vocabBadge = document.getElementById("vocabBadge");
const suggestionsList = document.getElementById("suggestionsList");
const vocabList = document.getElementById("vocabList");
const correctedOutput = document.getElementById("correctedOutput");
const copyBtn = document.getElementById("copyBtn");
const applyAllBtn = document.getElementById("applyAllBtn");

// Initialize Personalization Level Segment
function initLevelSelector() {
  document.querySelectorAll(".seg-btn").forEach((btn) => {
    if (btn.dataset.level === userLevel) btn.classList.add("active");
    else btn.classList.remove("active");

    btn.onclick = () => {
      document.querySelectorAll(".seg-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      userLevel = btn.dataset.level;
      localStorage.setItem("english_level", userLevel);
      if (inputText.value.trim()) checkBtn.click(); // Re-analyze under new level
    };
  });
}

// Live Word Count
// A small inline warning for empty-input Analyze clicks, created once and
// reused rather than a blocking alert() — appears right under the buttons.
const emptyInputWarning = document.createElement("div");
emptyInputWarning.textContent = "Please enter some text to analyze.";
emptyInputWarning.style.cssText = "color: var(--error-rose); font-size: 0.85rem; padding: 0 1.5rem 1rem; display: none;";
document.querySelector(".editor-panel").appendChild(emptyInputWarning);

inputText.oninput = () => {
  const words = inputText.value.trim().split(/\s+/).filter(Boolean);
  wordCount.textContent = `${words.length} words`;
  if (inputText.value.trim()) emptyInputWarning.style.display = "none";
};

// Switch Tabs in Inspector
tabCorrections.onclick = () => {
  tabCorrections.classList.add("active");
  tabVocab.classList.remove("active");
  correctionsView.classList.remove("hidden");
  vocabView.classList.add("hidden");
};

tabVocab.onclick = () => {
  tabVocab.classList.add("active");
  tabCorrections.classList.remove("active");
  vocabView.classList.remove("hidden");
  correctionsView.classList.add("hidden");
};

// Analyze
checkBtn.onclick = async () => {
  // Immediately stop any active voice recording and prevent delayed overwrite
  if (isRecording) {
    stopVoiceInput(true);
  }
  stopSpeaking();

  const text = inputText.value.trim();
  if (!text) {
    emptyInputWarning.style.display = "block";
    return;
  }
  emptyInputWarning.style.display = "none";
  lastAnalyzedText = text; // remember what this analysis was based on, for on-demand AI explanations later

  checkBtn.disabled = true;
  checkBtn.innerHTML = `<span>⟳</span> Elevating...`;

  try {
    const headers = { "Content-Type": "application/json" };
    if (authToken) headers["Authorization"] = `Bearer ${authToken}`;

    const res = await fetch("/api/check-grammar", {
      method: "POST",
      headers,
      body: JSON.stringify({ text, user_level: userLevel })
    });

    const data = await res.json();
    currentErrors = data.errors;

    // Update Metrics
    metricCefr.textContent = data.metrics.cefr_level || "B1";
    metricReadability.innerHTML = `${data.metrics.reading_ease || 80} <small>/ 100</small>`;
    metricVocab.textContent = data.metrics.vocab_richness || "70%";
    metricErrors.textContent = `${data.total_errors} Issues`;

    correctionsBadge.textContent = data.total_errors;
    vocabBadge.textContent = (data.vocab_upgrades || []).length;
    correctedOutput.textContent = data.corrected;

    renderCorrections(data.errors);
    renderVocabUpgrades(data.vocab_upgrades || []);

    // Reveal the metrics bar only now that we have a real result
    metricsBar.style.display = "grid";
  } catch (err) {
    alert("Analysis Error: " + err.message);
  } finally {
    checkBtn.disabled = false;
    checkBtn.innerHTML = `<span class="btn-icon">✦</span> Analyze & Elevate`;
  }
};

// Render Educational Correction Cards
function renderCorrections(errors) {
  suggestionsList.innerHTML = "";
  if (errors.length === 0) {
    suggestionsList.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">✓</div>
        <p>Exceptional! No grammatical flaws identified.</p>
      </div>`;
    applyAllBtn.classList.add("hidden");
    return;
  }

  applyAllBtn.classList.remove("hidden");

  errors.forEach((err) => {
    const card = document.createElement("div");
    card.className = "edu-card";
    card.innerHTML = `
      <div class="edu-header">
        <span class="edu-type">${err.type}</span>
        <button class="btn-ghost-sm accept-fix" data-id="${err.id}">Apply</button>
      </div>
      <div class="edu-diff">
        <del>${escapeHtml(err.original || "(missing)")}</del>
        <span>&rarr;</span>
        <ins>${escapeHtml(err.replacement || "(delete)")}</ins>
      </div>
      <div class="edu-explanation">
        <strong>Grammar Rule:</strong> ${escapeHtml(err.grammar_rule)}<br/>
        <span style="color: #cbd5e1;"><strong>Tip (${userLevel}):</strong> <span class="edu-tip-text">${escapeHtml(err.pedagogical_tip)}</span></span>
      </div>
      <div class="edu-actions">
        <button class="btn-ghost-sm explain-natural" data-id="${err.id}">✦ Explain more naturally</button>
        <button class="btn-card-listen" data-id="${err.id}" title="Pronounce correction">🔊 Hear Fix</button>
      </div>
    `;
    suggestionsList.appendChild(card);
  });

  document.querySelectorAll(".accept-fix").forEach((btn) => {
    btn.onclick = (e) => {
      const errId = parseInt(e.target.dataset.id);
      applyFix(errId);
    };
  });

  document.querySelectorAll(".btn-card-listen").forEach((btn) => {
    btn.onclick = () => {
      const errId = parseInt(btn.dataset.id);
      const err = currentErrors.find((x) => x.id === errId);
      if (!err) return;

      if (btn.classList.contains("speaking")) {
        stopSpeaking();
        return;
      }

      stopSpeaking();
      activeSpeakerBtn = btn;
      const phrase = `Original: ${err.original || "missing"}. Corrected: ${err.replacement || "delete"}.`;
      speakText(
        phrase,
        () => {
          btn.classList.add("speaking");
          btn.textContent = "⏹ Stop";
        },
        () => {
          btn.classList.remove("speaking");
          btn.textContent = "🔊 Hear Fix";
          activeSpeakerBtn = null;
        }
      );
    };
  });

  // On-demand AI explanation: only fires a generation call when the user
  // actually asks for it, one error at a time — this is what keeps the main
  // "Analyze" click fast regardless of how many mistakes are found.
  document.querySelectorAll(".explain-natural").forEach((btn) => {
    btn.onclick = async (e) => {
      const errId = parseInt(e.target.dataset.id);
      const err = currentErrors.find((x) => x.id === errId);
      if (!err) return;

      const card = e.target.closest(".edu-card");
      const tipSpan = card ? card.querySelector(".edu-tip-text") : null;

      btn.disabled = true;
      btn.textContent = "Thinking...";

      try {
        const headers = { "Content-Type": "application/json" };
        if (authToken) headers["Authorization"] = `Bearer ${authToken}`;

        const res = await fetch("/api/explain-natural", {
          method: "POST",
          headers,
          body: JSON.stringify({
            original: err.original,
            replacement: err.replacement,
            error_type: err.type,
            sentence: lastAnalyzedText,
            user_level: userLevel
          })
        });
        const data = await res.json();

        if (tipSpan) tipSpan.textContent = data.pedagogical_tip;

        if (data.tip_source === "ai") {
          btn.textContent = "✦ AI-explained";
          btn.disabled = true;
        } else {
          // Model wasn't available/usable this time — let them try again later,
          // but say so clearly instead of silently leaving the tip unchanged,
          // which otherwise looks like the click did nothing at all.
          btn.textContent = "Explain more naturally";
          btn.disabled = false;
          const note = document.createElement("div");
          note.textContent = "No better explanation available for this one — showing the standard tip.";
          note.style.cssText = "color: var(--text-muted); font-size: 0.75rem; margin-top: 0.4rem; font-style: italic;";
          btn.insertAdjacentElement("afterend", note);
          setTimeout(() => note.remove(), 4000);
        }
      } catch (fetchErr) {
        alert("Could not get a natural explanation: " + fetchErr.message);
        btn.textContent = "Explain more naturally";
        btn.disabled = false;
      }
    };
  });
}

// Render Vocabulary Upgrades
function renderVocabUpgrades(upgrades) {
  vocabList.innerHTML = "";
  if (upgrades.length === 0) {
    vocabList.innerHTML = `<div class="empty-state"><p>No vocabulary upgrades suggested for this excerpt.</p></div>`;
    return;
  }

  upgrades.forEach((item) => {
    const card = document.createElement("div");
    card.className = "edu-card";
    card.style.borderLeftColor = "var(--accent-cyan)";
    card.innerHTML = `
      <div class="edu-header">
        <span class="edu-type" style="color: var(--accent-cyan);">Vocabulary Elevation</span>
      </div>
      <div class="edu-diff">
        <del style="color: #94a3b8;">${escapeHtml(item.original)}</del>
        <span>&rarr;</span>
        <ins style="color: var(--accent-cyan);">${escapeHtml(item.alternatives.join(", "))}</ins>
      </div>
      <div class="edu-explanation">${escapeHtml(item.tip)}</div>
    `;
    vocabList.appendChild(card);
  });
}

// Apply single fix
function applyFix(errId) {
  const err = currentErrors.find((e) => e.id === errId);
  if (!err) return;
  const val = inputText.value;
  inputText.value = val.substring(0, err.start) + err.replacement + val.substring(err.end);
  checkBtn.click();
}

// Accept All
applyAllBtn.onclick = () => {
  if (correctedOutput.textContent) {
    inputText.value = correctedOutput.textContent;
    checkBtn.click();
  }
};

// Copy Polished Text
copyBtn.onclick = () => {
  navigator.clipboard.writeText(correctedOutput.textContent);
  alert("Polished output copied to clipboard!");
};

// Clear
clearBtn.onclick = () => {
  stopSpeaking();
  if (isRecording) stopVoiceInput(true);
  inputText.value = "";
  wordCount.textContent = "0 words";
  correctedOutput.textContent = "Your polished text will appear here.";
  suggestionsList.innerHTML = `<div class="empty-state"><p>Cleared.</p></div>`;
  correctionsBadge.textContent = "0";
  vocabBadge.textContent = "0";
  metricsBar.style.display = "none";
};

// Auth modal handlers
// ==========================================
// MODAL OPEN, CLOSE & TAB SWITCHING
// ==========================================

// Open Modal when top-right "Sign In" is clicked
openAuthModalBtn.onclick = () => {
  authModal.style.display = "flex";
  showTab("login");
};

// Close Modal when "X" is clicked
closeAuthModal.onclick = () => {
  authModal.style.display = "none";
};

// Close Modal if user clicks on the dark backdrop outside the card
authModal.onclick = (e) => {
  if (e.target === authModal) {
    authModal.style.display = "none";
  }
};

// Tab Switch Function (Strict Separation)
function showTab(tabName) {
  if (tabName === "login") {
    tabLogin.classList.add("active");
    tabRegister.classList.remove("active");
    loginForm.classList.add("active-form");
    registerForm.classList.remove("active-form");
  } else {
    tabRegister.classList.add("active");
    tabLogin.classList.remove("active");
    registerForm.classList.add("active-form");
    loginForm.classList.remove("active-form");
  }
}

// Click handlers for the two tabs
tabLogin.onclick = () => showTab("login");
tabRegister.onclick = () => showTab("register");

// ==========================================
// LOGIN & REGISTRATION FORM SUBMISSIONS
// ==========================================

// Handle Sign In Submission with Loading State
loginForm.onsubmit = async (e) => {
  e.preventDefault();
  const username = document.getElementById("loginUsername").value.trim();
  const password = document.getElementById("loginPassword").value.trim();
  const submitBtn = loginForm.querySelector("button[type='submit']");

  if (!username || !password) {
    alert("Please enter both your username and password.");
    return;
  }

  submitBtn.disabled = true;
  submitBtn.innerHTML = `<span class="btn-spinner"></span> Signing in...`;

  try {
    const res = await fetch("/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password })
    });
    const data = await res.json();

    if (!res.ok) {
      alert(data.detail || "Invalid credentials. If you haven't created an account yet, click 'Join GrammarLens'!");
      return;
    }

    setAuth(data.access_token, data.username, data.english_level);
    authModal.style.display = "none";
  } catch (err) {
    alert("Could not connect to server: " + err.message);
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = `Sign In`;
  }
};

// Handle Registration Submission with Loading State
registerForm.onsubmit = async (e) => {
  e.preventDefault();
  const username = document.getElementById("regUsername").value.trim();
  const email = document.getElementById("regEmail").value.trim();
  const password = document.getElementById("regPassword").value.trim();
  const submitBtn = registerForm.querySelector("button[type='submit']");

  if (!username || !email || !password) {
    alert("Please fill in all registration fields.");
    return;
  }

  submitBtn.disabled = true;
  submitBtn.innerHTML = `<span class="btn-spinner"></span> Creating account...`;

  try {
    const res = await fetch("/api/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, email, password, english_level: userLevel })
    });
    const data = await res.json();

    if (!res.ok) {
      alert(data.detail || "Registration failed. Try a different username or email.");
      return;
    }

    setAuth(data.access_token, data.username, data.english_level);
    authModal.style.display = "none";
    alert(`Account created successfully! Welcome, ${data.username}!`);
  } catch (err) {
    alert("Could not connect to server: " + err.message);
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = `Create Account`;
  }
};

logoutBtn.onclick = () => {
  localStorage.removeItem("token");
  localStorage.removeItem("username");
  authToken = null;
  currentUsername = null;
  updateAuthState();
};

function setAuth(token, username, level) {
  authToken = token;
  currentUsername = username;
  userLevel = level || "intermediate";
  localStorage.setItem("token", token);
  localStorage.setItem("username", username);
  localStorage.setItem("english_level", userLevel);
  updateAuthState();
}

function updateAuthState() {
  const token = localStorage.getItem("token");
  const username = localStorage.getItem("username");
  const userAvatar = document.getElementById("userAvatar");

  if (token && username) {
    userProfile.style.display = "flex";
    openAuthModalBtn.style.display = "none";
    userNameDisplay.textContent = username;
    if (userAvatar) userAvatar.textContent = username.charAt(0).toUpperCase();
  } else {
    userProfile.style.display = "none";
    openAuthModalBtn.style.display = "inline-flex";
  }
  initLevelSelector();
}

function escapeHtml(t) {
  const d = document.createElement("div");
  d.textContent = t;
  return d.innerHTML;
}

// Start
initLevelSelector();
updateAuthState();
console.log("[GrammarLens PRO] System initialized. Voice engine ready.");

// ==========================================
// VOICE FEATURES (SPEECH-TO-TEXT & TEXT-TO-SPEECH)
// ==========================================

let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];
let audioStream = null;
let recordingTimer = null;
let recordingSeconds = 0;
let speechRecognition = null;
let liveSpeechTranscript = "";
let baseInputText = "";
let currentUtterance = null;
let activeSpeakerBtn = null;

// Format seconds into mm:ss
function formatTime(sec) {
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}:${s < 10 ? "0" : ""}${s}`;
}

function cleanupAudioStream() {
  if (audioStream) {
    audioStream.getTracks().forEach((track) => {
      try { track.stop(); } catch (e) {}
    });
    audioStream = null;
  }
}

// Start voice dictation
async function startVoiceInput() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert("Voice input is not supported in this browser context (HTTPS or localhost is required).");
    return;
  }

  try {
    audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    alert("Microphone permission was denied or microphone is not found: " + err.message);
    return;
  }

  audioChunks = [];
  liveSpeechTranscript = "";
  baseInputText = inputText.value.trim();

  // Setup MediaRecorder for backend Faster-Whisper
  const mimeTypes = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/mp4"
  ];
  const selectedMime = mimeTypes.find(m => window.MediaRecorder && MediaRecorder.isTypeSupported(m)) || "";

  try {
    mediaRecorder = selectedMime ? new MediaRecorder(audioStream, { mimeType: selectedMime }) : new MediaRecorder(audioStream);
  } catch (e) {
    mediaRecorder = new MediaRecorder(audioStream);
  }

  mediaRecorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) {
      audioChunks.push(e.data);
    }
  };

  mediaRecorder.onstop = sendRecordedAudio;

  // Real-time live dictation via Web Speech API (instant feedback on screen)
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (SpeechRec) {
    try {
      speechRecognition = new SpeechRec();
      speechRecognition.continuous = true;
      speechRecognition.interimResults = true;
      speechRecognition.lang = "en-US";

      speechRecognition.onresult = (event) => {
        let interim = "";
        let finalTxt = "";
        for (let i = 0; i < event.results.length; ++i) {
          const trans = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalTxt += trans + " ";
          } else {
            interim += trans;
          }
        }
        liveSpeechTranscript = (finalTxt + interim).trim();
        if (liveSpeechTranscript) {
          inputText.value = baseInputText ? baseInputText + " " + liveSpeechTranscript : liveSpeechTranscript;
          inputText.dispatchEvent(new Event("input", { bubbles: true }));
          if (voiceStatusText) {
            voiceStatusText.textContent = "Hearing: " + (liveSpeechTranscript.length > 25 ? "..." + liveSpeechTranscript.slice(-25) : liveSpeechTranscript);
          }
        }
      };

      speechRecognition.onerror = (e) => {
        console.warn("Speech recognition notice:", e.error);
      };

      speechRecognition.start();
    } catch (e) {
      console.warn("SpeechRecognition start notice:", e);
    }
  }

  mediaRecorder.start(200); // 200ms slice
  isRecording = true;

  // Update UI
  if (voiceBtn) voiceBtn.classList.add("recording");
  if (voiceBtnText) voiceBtnText.textContent = "Stop Dictating";
  if (voiceStatus) {
    voiceStatus.classList.remove("hidden", "transcribing", "success");
    if (voiceStatusText) voiceStatusText.textContent = "Listening... speak now";
  }
  recordingSeconds = 0;
  if (voiceTimer) voiceTimer.textContent = "0:00";
  recordingTimer = setInterval(() => {
    recordingSeconds++;
    if (voiceTimer) voiceTimer.textContent = formatTime(recordingSeconds);
  }, 1000);
}

// Stop voice dictation
function stopVoiceInput(cancelUpload = false) {
  if (!isRecording) return;
  isRecording = false;

  clearInterval(recordingTimer);
  recordingTimer = null;

  if (speechRecognition) {
    try { speechRecognition.stop(); } catch (e) {}
  }

  if (voiceBtn) voiceBtn.classList.remove("recording");
  if (voiceBtnText) voiceBtnText.textContent = "Voice Input";

  // If live speech was already captured verbatim, or if cancelled (Analyze / Clear clicked),
  // NEVER call sendRecordedAudio or overwrite the input textarea!
  if (cancelUpload || (liveSpeechTranscript && liveSpeechTranscript.trim())) {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.onstop = null;
      try { mediaRecorder.stop(); } catch (e) {}
    }
    cleanupAudioStream();
    audioChunks = [];
    if (cancelUpload) {
      if (voiceStatus) voiceStatus.classList.add("hidden");
    } else {
      if (voiceStatus) {
        voiceStatus.classList.remove("transcribing");
        voiceStatus.classList.add("success");
        voiceStatusText.textContent = "Voice captured!";
        setTimeout(() => voiceStatus.classList.add("hidden"), 1500);
      }
    }
    return;
  }

  if (voiceStatus) {
    voiceStatus.classList.add("transcribing");
    if (voiceStatusText) voiceStatusText.textContent = "Transcribing audio...";
  }

  // Stop media recorder; onstop handler will trigger sendRecordedAudio() and then clean up tracks
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  } else {
    cleanupAudioStream();
  }
}

// Send audio to Faster-Whisper backend
async function sendRecordedAudio() {
  // Now that recorder has stopped and flushed data, clean up microphone tracks
  cleanupAudioStream();

  // If live speech recognition already captured the verbatim spoken words on-screen,
  // KEEP THEM so raw ungrammatical speech is preserved for grammar checking!
  if (liveSpeechTranscript && liveSpeechTranscript.trim()) {
    if (voiceStatus) {
      voiceStatus.classList.remove("transcribing");
      voiceStatus.classList.add("success");
      voiceStatusText.textContent = "Voice captured!";
      setTimeout(() => voiceStatus.classList.add("hidden"), 2000);
    }
    return;
  }

  if (!audioChunks.length) {
    if (voiceStatus) voiceStatus.classList.add("hidden");
    return;
  }

  const mime = (mediaRecorder && mediaRecorder.mimeType) || "audio/webm";
  const blob = new Blob(audioChunks, { type: mime });
  const ext = mime.includes("ogg") ? "ogg" : mime.includes("mp4") ? "mp4" : "webm";

  const formData = new FormData();
  formData.append("file", blob, `voice_recording.${ext}`);

  try {
    const headers = {};
    if (authToken) headers["Authorization"] = `Bearer ${authToken}`;

    const res = await fetch("/api/speech-to-text", {
      method: "POST",
      headers,
      body: formData
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Transcription error");
    }

    const transcribed = (data.transcript || "").trim();
    if (transcribed) {
      // Use Whisper output only if live speech recognition was not active
      inputText.value = baseInputText ? baseInputText + " " + transcribed : transcribed;
      inputText.dispatchEvent(new Event("input", { bubbles: true }));

      if (voiceStatus) {
        voiceStatus.classList.remove("transcribing");
        voiceStatus.classList.add("success");
        if (voiceStatusText) voiceStatusText.textContent = "Transcribed with Whisper!";
        setTimeout(() => {
          voiceStatus.classList.add("hidden");
        }, 2000);
      }
    } else {
      if (voiceStatus) voiceStatus.classList.add("hidden");
    }
  } catch (err) {
    console.warn("Server Whisper transcription error:", err);
    if (voiceStatus) {
      voiceStatus.classList.remove("transcribing");
      if (voiceStatusText) voiceStatusText.textContent = err.message || "Failed to transcribe";
      setTimeout(() => voiceStatus.classList.add("hidden"), 3000);
    }
  }
}

// Toggle voice input on button click
if (voiceBtn) {
  voiceBtn.onclick = () => {
    if (isRecording) {
      stopVoiceInput();
    } else {
      startVoiceInput();
    }
  };
}

// ==========================================
// TEXT-TO-SPEECH (VOICE PRONUNCIATION OUTPUT)
// ==========================================

let availableVoices = [];

function loadVoices() {
  if (!("speechSynthesis" in window)) return;
  availableVoices = window.speechSynthesis.getVoices();
}

if ("speechSynthesis" in window) {
  loadVoices();
  window.speechSynthesis.onvoiceschanged = loadVoices;
}

function getBestEnglishVoice() {
  if (!availableVoices.length) loadVoices();
  const preferred = availableVoices.find(v => 
    v.lang && v.lang.startsWith("en") && (
      v.name.includes("Natural") ||
      v.name.includes("Google") || 
      v.name.includes("Samantha") ||
      v.name.includes("Jenny") ||
      v.name.includes("Daniel") ||
      v.name.includes("US")
    )
  );
  if (preferred) return preferred;
  return availableVoices.find(v => v.lang && v.lang.startsWith("en")) || availableVoices[0] || null;
}

function speakText(text, onStart, onEnd) {
  if (!("speechSynthesis" in window)) {
    alert("Text-to-speech is not supported in this browser.");
    if (onEnd) onEnd();
    return false;
  }

  // Cancel prior speech and resume if Chrome is in paused state
  window.speechSynthesis.cancel();
  if (window.speechSynthesis.paused) {
    window.speechSynthesis.resume();
  }

  if (!text || !text.trim()) {
    if (onEnd) onEnd();
    return false;
  }

  const cleanText = text.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim();
  const utterance = new SpeechSynthesisUtterance(cleanText);
  utterance.lang = "en-US";
  utterance.rate = 0.95; // Clear pedagogical speed
  utterance.pitch = 1.0;

  const voice = getBestEnglishVoice();
  if (voice) utterance.voice = voice;

  utterance.onstart = () => {
    if (onStart) onStart();
  };

  utterance.onend = () => {
    currentUtterance = null;
    if (onEnd) onEnd();
  };

  utterance.onerror = (e) => {
    console.warn("SpeechSynthesis notice:", e);
    currentUtterance = null;
    if (onEnd) onEnd();
  };

  currentUtterance = utterance;
  window.speechSynthesis.speak(utterance);

  // Chromium watchdog to prevent speech synthesis queue stalling
  setTimeout(() => {
    if (window.speechSynthesis.paused) {
      window.speechSynthesis.resume();
    }
  }, 50);

  return true;
}

function stopSpeaking() {
  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
  if (activeSpeakerBtn) {
    activeSpeakerBtn.classList.remove("speaking");
    if (activeSpeakerBtn === speakSourceBtn || activeSpeakerBtn === speakOutputBtn) {
      activeSpeakerBtn.innerHTML = '<span class="voice-icon">🔊</span> Listen';
    }
    activeSpeakerBtn = null;
  }
}

// Speak Source Text Button
if (speakSourceBtn) {
  speakSourceBtn.onclick = () => {
    if (speakSourceBtn.classList.contains("speaking")) {
      stopSpeaking();
      return;
    }

    const text = inputText.value.trim();
    if (!text) {
      alert("Please enter or dictate some text into the Source Text box first.");
      return;
    }

    stopSpeaking();
    activeSpeakerBtn = speakSourceBtn;
    speakText(
      text,
      () => {
        speakSourceBtn.classList.add("speaking");
        speakSourceBtn.innerHTML = '<span class="voice-icon">⏹</span> Stop';
      },
      () => {
        speakSourceBtn.classList.remove("speaking");
        speakSourceBtn.innerHTML = '<span class="voice-icon">🔊</span> Listen';
        activeSpeakerBtn = null;
      }
    );
  };
}

// Speak Polished Output Button
if (speakOutputBtn) {
  speakOutputBtn.onclick = () => {
    if (speakOutputBtn.classList.contains("speaking")) {
      stopSpeaking();
      return;
    }

    const text = correctedOutput.textContent.trim();
    if (!text || text === "Your polished text will appear here.") {
      alert("No polished text yet! Click 'Analyze & Elevate' first to generate your polished output.");
      return;
    }

    stopSpeaking();
    activeSpeakerBtn = speakOutputBtn;
    speakText(
      text,
      () => {
        speakOutputBtn.classList.add("speaking");
        speakOutputBtn.innerHTML = '<span class="voice-icon">⏹</span> Stop';
      },
      () => {
        speakOutputBtn.classList.remove("speaking");
        speakOutputBtn.innerHTML = '<span class="voice-icon">🔊</span> Listen';
        activeSpeakerBtn = null;
      }
    );
  };
}