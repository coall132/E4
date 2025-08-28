(function(){
  // Helper auth (lit token dans localStorage ou cookie)
  function getCookie(name){
    return document.cookie.split("; ").reduce((acc, c) => {
      const [k, v] = c.split("=");
      return k === name ? decodeURIComponent(v) : acc;
    }, "");
  }
  async function apiFetch(params = {}){
    const headers = new Headers();
    const token = localStorage.getItem("ACCESS_TOKEN") || getCookie("auth_token") || "";
    if (token) headers.set("Authorization", "Bearer " + token);
    const url = new URL("/ui/api/restaurants", window.location.origin);
    Object.entries(params).forEach(([k, v]) => {
      if (v !== null && v !== undefined && v !== "") url.searchParams.append(k, v);
    });
    const r = await fetch(url, { headers });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  }

  // Convert price_level number to string of €
  function euros(n){
    return typeof n === "number" && n >=1 && n <=4 ? "€".repeat(n) : "—";
  }

  async function loadTable(params){
    const tbody = document.querySelector("#data-table tbody");
    const info  = document.getElementById("table-info");
    tbody.innerHTML = '<tr><td colspan="4" class="text-center text-muted">Chargement…</td></tr>';
    info.textContent = "";
    try {
      const data = await apiFetch(params);
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
      info.textContent = `Affichage de ${data.items.length} sur ${data.total} résultat(s)`;
    } catch (err) {
      console.error(err);
      tbody.innerHTML = '<tr><td colspan="4" class="text-center text-danger">Erreur lors du chargement.</td></tr>';
      info.textContent = "";
    }
  }

  // Lecture des champs du formulaire et lancement de la recherche
  function submitHandler(e){
    e.preventDefault();
    const params = {
      q:      document.getElementById("search").value.trim(),
      city:   document.getElementById("city").value.trim(),
      price_level: document.getElementById("price").value || "",
      open_day:    document.getElementById("open_day").value || "",
      options:     [...document.getElementById("options").selectedOptions].map(o => o.value).join(","),
      sort_by: document.getElementById("sort_by").value || "",
      sort_dir: document.getElementById("sort_dir").value || "",
      limit:   document.getElementById("limit").value || "50",
      skip:    0
    };
    loadTable(params);
  }

  // Clic sur une ligne du tableau
  function rowClickHandler(e){
    const row = e.target.closest(".restaurant-row");
    if (!row) return;
    const id = row.dataset.id;
    if (!id) return;
    // Si predict.js est chargé, on utilise chargerDetail (modale déjà définie)
    if (typeof window.chargerDetail === "function") {
      window.chargerDetail(id);
    } else {
      // fallback minimal : utiliser une requête locale
      chargerDetailLocal(id);
    }
  }

  // Fallback local si predict.js n'est pas chargé
  async function chargerDetailLocal(etabId){
    const modalEl = document.getElementById("detailModal");
    const titleEl = document.getElementById("detailModalTitle");
    const bodyEl  = document.getElementById("detailModalBody");
    titleEl.textContent = "Restaurant";
    bodyEl.innerHTML = '<p class="text-muted">Chargement…</p>';
    new bootstrap.Modal(modalEl).show();
    try {
      const d = await apiFetch({ id: etabId }); // on suppose que l’API /restaurant/<id> est accessible
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
          html += `<div><strong>Horaires :</strong><br>${d.horaires}</div>`;
        } else {
          html += `<div><strong>Horaires :</strong><br>${
            Object.entries(d.horaires).map(([j,p]) => {
              const txt = Array.isArray(p) ? p.join(", ") : String(p);
              return `<div><strong>${j}</strong> : ${txt}</div>`;
            }).join("")
          }</div>`;
        }
      }
      bodyEl.innerHTML = html;
    } catch (e) {
      bodyEl.innerHTML = '<p class="text-danger">Erreur lors du chargement.</p>';
      console.error(e);
    }
  }

  // Init : brancher événements
  document.getElementById("filter-form").addEventListener("submit", submitHandler);
  document.addEventListener("click", rowClickHandler);
  document.addEventListener("DOMContentLoaded", () => {
    // première charge (avec paramètres par défaut)
    submitHandler(new Event("submit"));
  });
})();
