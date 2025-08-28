(function(){
  // helpers pour récupérer token
  function getCookie(name){
    return document.cookie.split("; ").reduce((acc, c) => {
      const [k, v] = c.split("="); return k === name ? decodeURIComponent(v) : acc;
    }, "");
  }
  async function fetchData(params={}) {
    const headers = new Headers();
    const token = localStorage.getItem("ACCESS_TOKEN") || getCookie("auth_token") || "";
    if (token) headers.set("Authorization","Bearer "+token);
    const url = new URL("/ui/api/restaurants", window.location.origin);
    Object.entries(params).forEach(([k,v]) => { if (v) url.searchParams.append(k, v); });
    const r = await fetch(url, {headers});
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  }

  // Convert price level number to € symbols
  function euros(n){ return typeof n === "number" && n >=1 && n <=4 ? "€".repeat(n) : "—"; }

  async function loadTable() {
    const form = document.getElementById("filter-form");
    const params = {};
    const q  = document.getElementById("search").value.trim();
    const city = document.getElementById("city").value.trim();
    const price = document.getElementById("price").value;
    const day = document.getElementById("open_day").value;
    const opts = [...document.getElementById("options").selectedOptions].map(o => o.value);
    if (q) params.q = q;
    if (city) params.city = city;
    if (price) params.price_level = price;
    if (day) params.open_day = day;
    if (opts.length) params.options = opts.join(",");
    params.limit = 50;
    params.skip = 0;

    const tbody = document.querySelector("#data-table tbody");
    const info = document.getElementById("table-info");
    tbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted">Chargement…</td></tr>';
    info.textContent = "";

    try {
      const data = await fetchData(params);
      if (!Array.isArray(data.items) || !data.items.length) {
        tbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted">Aucun résultat.</td></tr>';
        return;
      }
      tbody.innerHTML = data.items.map(item => `
        <tr class="restaurant-row" data-id="${item.id}" style="cursor:pointer;">
          <td>${item.id}</td>
          <td>${item.nom || "—"}</td>
          <td>${item.adresse || "—"}</td>
          <td>${typeof item.rating === "number" ? item.rating.toFixed(1) : "—"}</td>
          <td>${euros(item.price_level)}</td>
        </tr>
      `).join("");
      info.textContent = `Affichage de ${data.items.length} sur ${data.total} résultat(s)`;
    } catch (err) {
      console.error(err);
      tbody.innerHTML = '<tr><td colspan="5" class="text-center text-danger">Erreur lors du chargement.</td></tr>';
      info.textContent = "";
    }
  }

  // ouverture fiche restaurant
  document.addEventListener("click", e => {
    const row = e.target.closest(".restaurant-row");
    if (!row) return;
    const id = row.dataset.id;
    if (!id) return;
    if (typeof window.chargerDetail === "function") {
      window.chargerDetail(id);
    } else {
      // fallback minimal si predict.js n'est pas chargé
      // (on peut réutiliser une version condensée de chargerDetailLocal)
      console.warn("chargerDetail introuvable");
    }
  });

  // soumission du formulaire
  document.getElementById("filter-form").addEventListener("submit", (e) => {
    e.preventDefault();
    loadTable();
  });

  // premier chargement
  document.addEventListener("DOMContentLoaded", loadTable);
})();
