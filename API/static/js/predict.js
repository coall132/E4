// ===== Helpers =====
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const getCookie = (name) =>
  document.cookie.split("; ").reduce((acc, c) => {
    const [k, v] = c.split("="); return k === name ? decodeURIComponent(v) : acc;
  }, "");

function authFetch(url, options = {}) {
  const headers = new Headers(options.headers || {});
  const token = localStorage.getItem("ACCESS_TOKEN") || getCookie("auth_token") || "";
  if (token) headers.set("Authorization", "Bearer " + token);
  return fetch(url, { ...options, headers });
}

function datetimeToOpenKey(s) {
  if (!s) return "";
  const dt = new Date(s); if (isNaN(dt)) return "";
  const jours = ["dimanche","lundi","mardi","mercredi","jeudi","vendredi","samedi"];
  const jour = jours[dt.getDay()], h = dt.getHours();
  const moment = h < 12 ? "matin" : h < 18 ? "midi" : "soir";
  return `open_${jour}_${moment}`;
}

function formatHours(hours) {
  if (!hours) return "";
  if (typeof hours === "string") return hours;
  return Object.entries(hours).map(([j, p]) => {
    const t = Array.isArray(p) ? p.join(", ") : String(p);
    return `<div><strong>${j}</strong> : ${t}</div>`;
  }).join("");
}

// ===== Rendu résultats =====
function renderResults(data) {
  const cont = $("#predict-result");
  const items = data.items_rich || data.items || [];
  if (!items.length) { cont.innerHTML = `<div class="alert alert-warning">Aucun résultat.</div>`; return; }

  cont.innerHTML = items.map(it => {
    const nom = it.details?.nom || `Restaurant ${it.etab_id}`;
    const adresse = it.details?.adresse || "";
    const note = typeof it.details?.rating === "number" ? `<span class="small text-muted">Note ${it.details.rating.toFixed(1)}/5</span>` : "";
    return `
      <div class="card mb-2 bg-dark text-white border-secondary result-item" data-etab-id="${it.etab_id}" style="cursor:pointer;">
        <div class="card-body d-flex justify-content-between">
          <div class="pe-3">
            <h6 class="card-title mb-1"># – ${nom}</h6>
            <p class="card-text small text-muted mb-0">${adresse}</p>
          </div>
          <div class="text-end">${note}</div>
        </div>
      </div>`;
  }).join("");

  // stocker l'id de prédiction pour le feedback
  $("#prediction-id").value = data.prediction_id || data.id || "";
  $("#feedback-button-wrapper").classList.remove("d-none");
}

// ===== Détails établissement =====
async function openDetail(etabId) {
  const modal = new bootstrap.Modal($("#detailModal"));
  const body = $("#detailModalBody");
  $("#detailModalTitle").textContent = "Chargement…";
  body.innerHTML = '<p class="text-muted">Veuillez patienter…</p>';
  modal.show();

  try {
    const r = await authFetch(`/restaurant/${encodeURIComponent(etabId)}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = await r.json();

    $("#detailModalTitle").textContent = d.nom || "Restaurant";
    let html = "";
    if (d.description) html += `<p>${d.description}</p>`;
    html += `<p><strong>Adresse :</strong> ${d.adresse || "—"}</p>`;
    html += `<p><strong>Téléphone :</strong> ${d.telephone || "—"}</p>`;
    html += `<p><strong>Site :</strong> ${d.site_web ? `<a href="${d.site_web}" target="_blank" rel="noopener">${d.site_web}</a>` : "—"}</p>`;
    html += `<p><strong>Note :</strong> ${typeof d.rating === "number" ? `${d.rating.toFixed(1)}/5` : "—"}</p>`;
    html += `<p><strong>Niveau de prix :</strong> ${d.price_level ? "€".repeat(d.price_level) : "—"}</p>`;
    html += `<div class="mb-2"><strong>Horaires :</strong><br>${formatHours(d.horaires)}</div>`;

    if (d.latitude && d.longitude) {
      html += `
        <div class="ratio ratio-16x9 my-3">
          <iframe src="https://www.google.com/maps?q=${d.latitude},${d.longitude}&z=15&output=embed"
                  width="100%" height="100%" style="border:0"></iframe>
        </div>`;
    }

    html += `
      <div class="mt-3">
        <button class="btn btn-outline-info btn-sm" id="btnAvis">Voir les avis</button>
        <div id="avisContainer" class="mt-2"></div>
      </div>`;

    body.innerHTML = html;

    // Charger les avis sur clic
    $("#btnAvis")?.addEventListener("click", async () => {
      const box = $("#avisContainer");
      box.innerHTML = '<p class="text-muted">Chargement des avis…</p>';
      try {
        const rr = await authFetch(`/restaurant/${encodeURIComponent(etabId)}/reviews`);
        const reviews = rr.ok ? await rr.json() : [];
        box.innerHTML = reviews.length
          ? reviews.map(rv => {
              const date = rv.date ? rv.date.slice(0,10) : "";
              const note = typeof rv.rating === "number" ? `${rv.rating}/5` : "";
              const comment = rv.comment || "";
              return `<div class="border-top border-secondary pt-2 mt-2">
                        <small class="text-muted">${date} — Note ${note}</small><br>${comment}
                      </div>`;
            }).join("")
          : '<p class="text-muted">Aucun avis.</p>';
      } catch { box.innerHTML = '<p class="text-danger">Erreur chargement avis.</p>'; }
    });
  } catch (err) {
    body.innerHTML = '<p class="text-danger">Erreur lors du chargement des détails.</p>';
    console.error(err);
  }
}

// ===== Soumission prédiction =====
$("#predict-form")?.addEventListener("submit", async (e) => {
  e.preventDefault();
  const cp   = $("#cp").value.trim() || "37000";
  const desc = $("#desc").value.trim();
  const pRaw = $("#price").value || "2";
  const k    = parseInt($("#k").value, 10) || 10;
  const openKey = datetimeToOpenKey($("#datetime").value);
  const opts = [...($("#opts")?.selectedOptions ?? [])].map(o => o.value);


  const payload = {
    description: desc || "",
    open: openKey || "",
    options: opts,           
  };
  if (pRaw !== "") {
    const p = parseInt(pRaw, 10);
    if (!Number.isNaN(p)) payload.price_level = p;
  }
  if (cp) payload.city = cp;

  const cont = $("#predict-result");
  cont.innerHTML = `<div class="alert alert-secondary">Calcul en cours…</div>`;

  try {
    const r = await authFetch(`/predict?k=${encodeURIComponent(k)}&use_ml=true`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!r.ok) {
      let detail = ""; try { detail = (await r.json()).detail ?? ""; } catch {}
      throw new Error(`HTTP ${r.status}${detail ? " - " + JSON.stringify(detail) : ""}`);
    }
    renderResults(await r.json());
  } catch (err) {
    cont.innerHTML = `<div class="alert alert-danger">Erreur : ${err.message}</div>`;
    console.error(err);
  }
});

// ===== Clic sur un résultat (délégation) =====
document.addEventListener("click", (e) => {
  const card = e.target.closest(".result-item"); if (!card) return;
  const id = card.dataset.etabId; if (!id) return;
  openDetail(id);
});

// ===== Bouton ouverture modale Feedback =====
$("#open-feedback-btn")?.addEventListener("click", () => {
  new bootstrap.Modal($("#feedbackModal")).show();
});

// ===== Envoi feedback =====
$("#feedback-form")?.addEventListener("submit", async (e) => {
  e.preventDefault();
  const predId = $("#prediction-id").value;
  const rRaw   = $("#rating").value;
  const comment = $("#comment").value.trim();
  if (!predId) return alert("Identifiant de prédiction manquant.");

  const data = { prediction_id: predId };
  const rNum = parseInt(rRaw, 10);
  if (!Number.isNaN(rNum)) data.rating = rNum;
  if (comment) data.comment = comment;

  try {
    const r = await authFetch("/feedback", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    });
    if (!r.ok) {
      let detail = ""; try { detail = (await r.json()).detail ?? ""; } catch {}
      throw new Error(`HTTP ${r.status}${detail ? " - " + JSON.stringify(detail) : ""}`);
    }
    bootstrap.Modal.getInstance($("#feedbackModal"))?.hide();
    const ok = document.createElement("div");
    ok.className = "alert alert-success"; ok.textContent = "Merci pour votre retour !";
    $("#predict-result").prepend(ok); setTimeout(() => ok.remove(), 3000);
    $("#rating").value = ""; $("#comment").value = "";
  } catch (err) {
    alert("Erreur lors de l'envoi du feedback : " + err.message);
    console.error(err);
  }
});
