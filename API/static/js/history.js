(function () {
  // -------- Helpers auth & DOM
  function getCookie(name) {
  return document.cookie.split("; ").reduce((acc, c) => {
    const [k, v] = c.split("="); return k === name ? decodeURIComponent(v) : acc;
  }, "");
}
  function authFetch(url, options = {}) {
    const headers = new Headers(options.headers || {});
    // ⬇️ on regarde localStorage puis cookie ACCESS_TOKEN (puis auth_token en secours)
    const token =
      localStorage.getItem("ACCESS_TOKEN") ||
      getCookie("ACCESS_TOKEN") ||
      getCookie("auth_token") ||
      "";
    if (token) headers.set("Authorization", "Bearer " + token);
    return fetch(url, { ...options, headers });
}
  function $all(sel, root=document){ return Array.from(root.querySelectorAll(sel)); }

  // -------- Date formatter (robuste aux ISO avec timezone)
  function fmtDateTime(iso) {
    if (!iso) return "";
    const hasTz = /[+-]\d\d:\d\d$|Z$/.test(iso);
    const d = new Date(hasTz ? iso : (iso + "Z"));
    if (isNaN(d)) return iso;
    return d.toLocaleString('fr-FR', {
      year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit', second: '2-digit',
      hour12: false
    }).replace(',', '');
  }

  // -------- Rendu résumé de formulaire
  function renderFormSummary(form) {
    if (!form) return "";
    const chips = [];
    if (form.city) chips.push(`<span class="badge bg-secondary me-1">Ville: ${form.city}</span>`);
    if (form.open) chips.push(`<span class="badge bg-secondary me-1">Créneau: ${form.open}</span>`);
    if (typeof form.price_level === "number") chips.push(`<span class="badge bg-secondary me-1">Prix: ${"€".repeat(form.price_level)}</span>`);
    if (Array.isArray(form.options) && form.options.length) chips.push(`<span class="badge bg-secondary me-1">${form.options.length} option(s)</span>`);
    if (form.description) chips.push(`<span class="badge bg-info text-dark me-1">Desc.</span>`);
    return chips.join(" ");
  }

  function renderFormBlock(form) {
    if (!form) return "";
    return `
      <div class="border rounded p-2 mb-3">
        <div class="small text-muted mb-2">Formulaire soumis :</div>
        <div><strong>Ville : </strong>${form.city || "—"}</div>
        <div><strong>Créneau : </strong>${form.open || "—"}</div>
        <div><strong>Niveau de prix : </strong>${typeof form.price_level === "number" ? "€".repeat(form.price_level) : "—"}</div>
        <div><strong>Options : </strong>${
          Array.isArray(form.options) && form.options.length ? form.options.join(", ") : "—"
        }</div>
        <div><strong>Description : </strong>${form.description || "—"}</div>
        <div><strong>Soumis le : </strong>${form.created_at ? fmtDateTime(form.created_at) : "—"}</div>
      </div>`;
  }

  // -------- Charger la liste
  async function loadHistory() {
    const list = document.getElementById("history-list");
    list.innerHTML = '<p class="text-muted">Chargement…</p>';
    try {
      const resp = await authFetch("/history/predictions");
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();

      if (!data.length) {
        list.innerHTML = '<p class="text-muted">Aucune prédiction enregistrée.</p>';
        return;
      }

      list.innerHTML = data.map(p => {
        const title = fmtDateTime(p.created_at || p.form?.created_at); 
        const fbChip = (p.feedback && p.feedback.rating !== null)
          ? `<span class="badge bg-success">Note ${p.feedback.rating}/5</span>`
          : `<span class="badge bg-secondary">Pas de feedback</span>`;
        return `
          <div class="card mb-2 bg-dark text-white border-secondary pred-item" data-id="${p.id}" style="cursor:pointer;">
            <div class="card-body d-flex justify-content-between align-items-start">
              <div class="pe-3">
                <h6 class="card-title mb-1">${title || 'Prédiction'}</h6>
                <small class="text-muted d-block mb-2">#${String(p.id).slice(0,8)}</small>
                <div>${renderFormSummary(p.form)}</div>
              </div>
              <div class="text-end">
                ${fbChip}<br>
                <small>K=${p.k}</small>
              </div>
            </div>
          </div>`;
      }).join("");

    } catch (e) {
      list.innerHTML = '<p class="text-danger">Erreur lors du chargement de l’historique.</p>';
      console.error(e);
    }
  }

  // -------- Détail d'une prédiction
  async function openPredDetail(predId) {
    const predEl = document.getElementById("predDetailModal");
    const modal = new bootstrap.Modal(predEl);
    const body = document.getElementById("predDetailBody");
    body.innerHTML = '<p class="text-muted">Chargement…</p>';
    modal.show();

    try {
      const resp = await authFetch(`/history/prediction/${encodeURIComponent(predId)}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const d = await resp.json();

      const title = fmtDateTime(d.created_at || d.form?.created_at);
      let html = `<h5 class="mb-2">${title || ('Prédiction #' + d.id.slice(0,8))}</h5>`;
      html += `<p><strong>k :</strong> ${d.k} | <strong>Modèle :</strong> ${d.model_version} | <strong>Latence :</strong> ${d.latency_ms ?? "–"} ms</p>`;


      html += renderFormBlock(d.form);

      // Feedback
      if (d.feedback && d.feedback.rating !== null) {
        html += `<p><strong>Feedback :</strong> Note ${d.feedback.rating}/5<br>${d.feedback.comment || ''}</p>`;
      } else {
        html += `<p><em>Aucun feedback pour cette prédiction.</em></p>
                 <button class="btn btn-sm btn-outline-primary mb-3" id="addFeedbackBtn">Laisser un feedback</button>`;
      }

      // Items
      const items = (Array.isArray(d.items_rich) && d.items_rich.length) ? d.items_rich : d.items;

      html += `<h6>Items recommandés :</h6><div class="list-group" id="itemsList">`;
      items.forEach(it => {
        const name   = it?.name?? `#${it.etab_id}`;
        const rating = (it?.details && typeof it.rating === 'number') ? ` —  ${it.details.rating.toFixed(1)}` : '';
        const score  = Number(it.score).toFixed(3);
        html += `
          <div class="list-group-item list-group-item-action bg-dark text-white border-secondary result-item"
               data-etab-id="${it.etab_id}" style="cursor:pointer;">
            #${it.rank} – <strong>${name}</strong>${rating} (Score ${score})
          </div>`;
      });
      html += `</div>`;

      body.innerHTML = html;

      // bouton feedback → fermer la modale prédiction avant d’ouvrir la modale feedback
      const addBtn = document.getElementById("addFeedbackBtn");
      if (addBtn) {
        addBtn.addEventListener("click", () => {
          const feedbackEl = document.getElementById("feedbackModal");
          document.getElementById("prediction-id").value = d.id;

          const predModal = bootstrap.Modal.getOrCreateInstance(predEl);
          predModal?.hide();
          setTimeout(() => {
            new bootstrap.Modal(feedbackEl).show();
          }, 200);
        });
      }

      // clic item → fermer la modale prédiction avant d’ouvrir le détail resto
      $all(".result-item", body).forEach(el => {
        el.addEventListener("click", () => {
          const id = el.dataset.etabId;
          if (!id) return;

          const predModal = bootstrap.Modal.getOrCreateInstance(predEl);
          predModal?.hide();

          setTimeout(() => {
            if (typeof window.chargerDetail === "function") {
              window.chargerDetail(id);
            } else {
              chargerDetailLocal(id);
            }
          }, 200);
        });
      });

    } catch (e) {
      body.innerHTML = '<p class="text-danger">Erreur lors du chargement des détails.</p>';
      console.error(e);
    }
  }

  // -------- Fallback local pour charger un resto (si predict.js absent)
  async function chargerDetailLocal(etabId) {
    const modalEl = document.getElementById("detailModal");
    const titleEl = document.getElementById("detailModalTitle");
    const bodyEl  = document.getElementById("detailModalBody");
    titleEl.textContent = "Restaurant";
    bodyEl.innerHTML = '<p class="text-muted">Chargement…</p>';
    const m = new bootstrap.Modal(modalEl);
    m.show();

    try {
      const r = await authFetch(`/restaurant/${encodeURIComponent(etabId)}`);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const d = await r.json();

      titleEl.textContent = d.nom || "Restaurant";
      let html = "";
      if (d.description) html += `<p>${d.description}</p>`;
      html += `<p><strong>Adresse :</strong> ${d.adresse || "—"}</p>`;
      html += `<p><strong>Téléphone :</strong> ${d.telephone || "—"}</p>`;
      html += `<p><strong>Site :</strong> ${d.site_web ? `<a href="${d.site_web}" target="_blank">${d.site_web}</a>` : "—"}</p>`;
      html += `<p><strong>Note :</strong> ${typeof d.rating === "number" ? `${d.rating.toFixed(1)}/5` : "—"}</p>`;
      html += `<p><strong>Niveau de prix :</strong> ${d.price_level ? "€".repeat(d.price_level) : "—"}</p>`;

      if (d.horaires) {
        if (typeof d.horaires === "string") {
          html += `<div class="mb-2"><strong>Horaires :</strong><br>${d.horaires}</div>`;
        } else {
          html += `<div class="mb-2"><strong>Horaires :</strong><br>${
            Object.entries(d.horaires).map(([j,pl]) => {
              const disp = Array.isArray(pl) ? pl.join(", ") : String(pl);
              return `<div><strong>${j}</strong> : ${disp}</div>`;
            }).join("")
          }</div>`;
        }
      }

      if (d.latitude && d.longitude) {
        html += `
        <div class="ratio ratio-16x9 my-3">
          <iframe src="https://www.google.com/maps?q=${d.latitude},${d.longitude}&z=15&output=embed"
                  width="100%" height="100%" style="border:0"></iframe>
        </div>`;
      }

      html += `<div class="mt-3">
        <button class="btn btn-outline-info btn-sm" id="btnAvisLocal">Voir les avis</button>
        <div id="avisContainerLocal" class="mt-2"></div>
      </div>`;

      bodyEl.innerHTML = html;

      document.getElementById("btnAvisLocal").addEventListener("click", async () => {
        const container = document.getElementById("avisContainerLocal");
        container.innerHTML = '<p class="text-muted">Chargement des avis…</p>';
        try {
          const rr = await authFetch(`/restaurant/${encodeURIComponent(etabId)}/reviews`);
          const reviews = rr.ok ? await rr.json() : [];
          container.innerHTML = reviews.length ? reviews.map(rv => {
            const date = rv.date ? rv.date.slice(0,10) : "";
            const note = typeof rv.rating === "number" ? `${rv.rating}/5` : "";
            const comment = rv.comment || rv.original_text || "";
            return `<div class="border-top border-secondary pt-2 mt-2">
              <small class="text-muted">${date} — Note ${note}</small><br>
              <span>${comment}</span>
            </div>`;
          }).join("") : '<p class="text-muted">Aucun avis.</p>';
        } catch (e) {
          container.innerHTML = '<p class="text-danger">Erreur chargement avis.</p>';
          console.error(e);
        }
      });

    } catch (e) {
      bodyEl.innerHTML = '<p class="text-danger">Erreur lors du chargement des détails.</p>';
      console.error(e);
    }
  }

  // -------- Feedback depuis l’historique (form global)
  (function attachFeedbackHandler(){
    const form = document.getElementById("feedback-form");
    if (!form) return;

    form.addEventListener("submit", async (e) => {
      e.preventDefault();

      const predId = document.getElementById("prediction-id").value;
      const rating = parseInt(document.getElementById("rating").value || "", 10);
      const comment = document.getElementById("comment").value.trim();
      if (!predId) { alert("Identifiant de prédiction manquant."); return; }

      const payload = { prediction_id: predId };
      if (!Number.isNaN(rating)) payload.rating = rating;
      if (comment) payload.comment = comment;

      const submitBtn = form.querySelector('button[type="submit"]');
      submitBtn?.setAttribute("disabled", "disabled");

      try {
        const r = await authFetch("/feedback", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });

        let data = {};
        try { data = await r.json(); } catch {}

        if (r.status === 409 || (data && typeof data.detail === "string" && data.detail.includes("déjà existant"))) {
          bootstrap.Modal.getInstance(document.getElementById("feedbackModal"))?.hide();
          window.location.reload();
          return;
        }

        if (!r.ok) {
          const det = data?.detail ? ` - ${JSON.stringify(data.detail)}` : "";
          throw new Error(`HTTP ${r.status}${det}`);
        }

        bootstrap.Modal.getInstance(document.getElementById("feedbackModal"))?.hide();
        document.getElementById("rating").value = "";
        document.getElementById("comment").value = "";
        window.location.reload();

      } catch (err) {
        alert("Erreur lors de l'envoi du feedback : " + err.message);
        console.error(err);
      } finally {
        submitBtn?.removeAttribute("disabled");
      }
    });
  })();

  // -------- Clic sur une carte historique → ouvrir détail prédiction
  document.addEventListener("click", (e) => {
    const card = e.target.closest(".pred-item");
    if (!card) return;
    openPredDetail(card.dataset.id);
  });

  // -------- Cleanup global (modales)
  document.addEventListener("hidden.bs.modal", () => {
    if (!document.querySelector(".modal.show")) {
      document.querySelectorAll(".modal-backdrop").forEach(b => b.remove());
      document.body.classList.remove("modal-open");
      document.body.style.removeProperty("padding-right");
      document.body.style.removeProperty("overflow");
    }
  });

  // -------- Init
  document.addEventListener("DOMContentLoaded", loadHistory);
})();

