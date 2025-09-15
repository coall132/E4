
(function(){
  // ---------- Helpers auth & DOM ----------
  function getCookie(name){
    return document.cookie.split("; ").reduce((acc, c) => {
      const [k, v] = c.split("="); return k === name ? decodeURIComponent(v) : acc;
    }, "");
  }
  function getAnyToken(){
    return (
      localStorage.getItem("ACCESS_TOKEN") ||
      getCookie("ACCESS_TOKEN") ||
      getCookie("auth_token") ||
      ""
    );
  }
  function authFetch(input, options = {}) {
    const headers = new Headers(options.headers || {});
    const t = getAnyToken();
    if (t) headers.set("Authorization", "Bearer " + t);
    return fetch(input, { ...options, headers, credentials: "same-origin" });
  }

  function ensureDetailModal(){
    let modalEl = document.getElementById("detailModal");
    if (!modalEl) {
      modalEl = document.createElement("div");
      modalEl.id = "detailModal";
      modalEl.className = "modal fade";
      modalEl.tabIndex = -1;
      modalEl.innerHTML = `
        <div class="modal-dialog modal-lg modal-dialog-scrollable">
          <div class="modal-content bg-dark text-white border-secondary">
            <div class="modal-header">
              <h5 class="modal-title" id="detailModalTitle">Restaurant</h5>
              <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="Fermer"></button>
            </div>
            <div class="modal-body" id="detailModalBody">
              <p class="text-muted">Chargement…</p>
            </div>
          </div>
        </div>`;
      document.body.appendChild(modalEl);
    }
    const titleEl = document.getElementById("detailModalTitle");
    const bodyEl  = document.getElementById("detailModalBody");

    modalEl.addEventListener("hidden.bs.modal", () => {
      if (!document.querySelector(".modal.show")) {
        document.querySelectorAll(".modal-backdrop").forEach(b => b.remove());
        document.body.classList.remove("modal-open");
        document.body.style.removeProperty("padding-right");
        document.body.style.removeProperty("overflow");
      }
    });

    return { modalEl, titleEl, bodyEl };
  }

  // ---------- API liste ----------
  async function fetchList(params = {}){
    const url = new URL("/ui/api/restaurants", window.location.origin);
    Object.entries(params).forEach(([k, v]) => {
      if (v !== null && v !== undefined && v !== "") url.searchParams.append(k, v);
    });
    const r = await authFetch(url.toString());
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  }

  function euros(n){
    return typeof n === "number" && n >= 1 && n <= 4 ? "€".repeat(n) : "—";
  }

  async function loadTable(params){
    const tbody = document.querySelector("#data-table tbody");
    const info  = document.getElementById("table-info");
    if (!tbody) return;

    tbody.innerHTML = '<tr><td colspan="4" class="text-center text-muted">Chargement…</td></tr>';
    if (info) info.textContent = "";

    try {
      const data = await fetchList(params);
      if (!data.items || !data.items.length) {
        tbody.innerHTML = '<tr><td colspan="4" class="text-center text-muted">Aucun résultat.</td></tr>';
        return;
      }

      tbody.innerHTML = data.items.map(item => `
        <tr class="restaurant-row" data-id="${item.id}" style="cursor:pointer;">
          <td>${item.id}</td>
          <td>${item.nom || "—"}</td>
          <td>${typeof item.rating === "number" ? item.rating.toFixed(1) : "—"}</td>
          <td>${euros(item.price_level)}</td>
        </tr>
      `).join("");

      tbody.querySelectorAll(".restaurant-row").forEach(tr => {
        tr.addEventListener("click", () => openRestaurantDetail(tr.dataset.id));
      });

      if (info) info.textContent = `Affichage de ${data.items.length} sur ${data.total} résultat(s)`;
    } catch (err) {
      console.error("[loadTable] erreur :", err);
      tbody.innerHTML = '<tr><td colspan="4" class="text-center text-danger">Erreur lors du chargement.</td></tr>';
    }
  }

  // ---------- Détail restaurant ----------
  async function openRestaurantDetail(etabId){
    const { modalEl, titleEl, bodyEl } = ensureDetailModal();
    titleEl.textContent = "Restaurant";
    bodyEl.innerHTML = '<p class="text-muted">Chargement…</p>';
    const modal = new bootstrap.Modal(modalEl);
    modal.show();

    try {
      const url = new URL(`/restaurant/${encodeURIComponent(etabId)}`, window.location.origin).toString();
      console.debug("[detail] fetch", url);
      const r = await authFetch(url);

      const raw = await r.text();
      if (!r.ok) {
        bodyEl.innerHTML = `<p class="text-danger">Erreur (HTTP ${r.status}).</p><pre class="small text-muted">${raw.slice(0,500)}</pre>`;
        return;
      }

      let d;
      try {
        d = JSON.parse(raw);
      } catch {
        bodyEl.innerHTML = `<p class="text-danger">Réponse non-JSON (probable redirection ou erreur).</p><pre class="small text-muted">${raw.slice(0,500)}</pre>`;
        return;
      }

      if (!d || (typeof d !== "object") || (!d.nom && !d.adresse && !d.rating && !d.price_level)) {
        bodyEl.innerHTML = `<p class="text-warning">Aucune donnée détaillée reçue pour l'établissement #${etabId}.</p>`;
        return;
      }

      titleEl.textContent = d.nom || `Restaurant #${etabId}`;
      let html = "";
      if (d.description) html += `<p>${d.description}</p>`;
      html += `<p><strong>Adresse :</strong> ${d.adresse || "—"}</p>`;
      html += `<p><strong>Téléphone :</strong> ${d.telephone || "—"}</p>`;
      html += `<p><strong>Site :</strong> ${d.site_web ? `<a href="${d.site_web}" target="_blank" rel="noopener">${d.site_web}</a>` : "—"}</p>`;
      html += `<p><strong>Note :</strong> ${typeof d.rating === "number" ? `${d.rating.toFixed(1)}/5` : "—"}</p>`;
      html += `<p><strong>Niveau de prix :</strong> ${d.price_level ? "€".repeat(d.price_level) : "—"}</p>`;

      if (d.horaires) {
        if (typeof d.horaires === "string") {
          html += `<div class="mb-2"><strong>Horaires :</strong><br>${d.horaires}</div>`;
        } else if (Array.isArray(d.horaires)) {
          const uniq = [...new Set(d.horaires)];
          html += `<div class="mb-2"><strong>Horaires :</strong><br>${uniq.join("<br>")}</div>`;
        } else {
          html += `<div class="mb-2"><strong>Horaires :</strong><br>${
            Object.entries(d.horaires).map(([j, pl]) => {
              const arr = Array.isArray(pl) ? pl : [String(pl)];
              const uniq = [...new Set(arr)];
              return `<div><strong>${j}</strong> : ${uniq.join(", ")}</div>`;
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

      html += `
        <div class="mt-3">
          <button class="btn btn-outline-info btn-sm" id="btnAvisLocal">Voir les avis</button>
          <div id="avisContainerLocal" class="mt-2"></div>
        </div>`;

      bodyEl.innerHTML = html;

      // Avis
      document.getElementById("btnAvisLocal")?.addEventListener("click", async () => {
        const avisBox = document.getElementById("avisContainerLocal");
        avisBox.innerHTML = '<p class="text-muted">Chargement des avis…</p>';
        try {
          const rr = await authFetch(new URL(`/restaurant/${encodeURIComponent(etabId)}/reviews`, window.location.origin).toString());
          const raw2 = await rr.text();
          if (!rr.ok) {
            avisBox.innerHTML = `<p class="text-danger">Erreur (HTTP ${rr.status})</p><pre class="small text-muted">${raw2.slice(0,400)}</pre>`;
            return;
          }
          let reviews;
          try { reviews = JSON.parse(raw2); } catch { reviews = []; }
          avisBox.innerHTML = (Array.isArray(reviews) && reviews.length)
            ? reviews.map(rv => {
                const date = rv.date ? rv.date.slice(0,10) : "";
                const note = typeof rv.rating === "number" ? `${rv.rating}/5` : "";
                const comment = rv.comment || rv.original_text || "";
                return `<div class="border-top border-secondary pt-2 mt-2">
                  <small class="text-muted">${date} — Note ${note}</small><br>
                  <span>${comment}</span>
                </div>`;
              }).join("")
            : '<p class="text-muted">Aucun avis.</p>';
        } catch (e) {
          console.error("[avis] erreur :", e);
          avisBox.innerHTML = '<p class="text-danger">Erreur chargement avis.</p>';
        }
      });
    } catch (e) {
      console.error("[detail] erreur :", e);
      bodyEl.innerHTML = '<p class="text-danger">Erreur lors du chargement des détails.</p>';
    }
  }

  // ---------- Lecture formulaire + recherche ----------
  function readParamsFromForm(){
    return {
      q:      document.getElementById("search")?.value.trim(),
      city:   document.getElementById("city")?.value.trim(),
      price_level: document.getElementById("price")?.value || "",
      open_day:    document.getElementById("open_day")?.value || "",
      options:     [...(document.getElementById("options")?.selectedOptions || [])].map(o => o.value).join(","),
      sort_by: document.getElementById("sort_by")?.value || "",
      sort_dir: document.getElementById("sort_dir")?.value || "",
      limit:   document.getElementById("limit")?.value || "50",
      skip:    0
    };
  }

  function onSubmit(e){
    e.preventDefault();
    loadTable(readParamsFromForm());
  }

  document.getElementById("filter-form")?.addEventListener("submit", onSubmit);
  document.addEventListener("DOMContentLoaded", () => {
    onSubmit(new Event("submit"));
  });

  document.addEventListener("hidden.bs.modal", () => {
    if (!document.querySelector(".modal.show")) {
      document.querySelectorAll(".modal-backdrop").forEach(b => b.remove());
      document.body.classList.remove("modal-open");
      document.body.style.removeProperty("padding-right");
      document.body.style.removeProperty("overflow");
    }
  });
})();

