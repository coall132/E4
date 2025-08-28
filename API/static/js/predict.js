document.addEventListener('DOMContentLoaded', () => {
  // --- Auth helper: lit le token depuis localStorage OU le cookie "auth_token"
  function getCookie(name) {
    return document.cookie.split('; ').reduce((acc, c) => {
      const [k, v] = c.split('=');
      return k === name ? decodeURIComponent(v) : acc;
    }, '');
  }
  function authFetch(url, options = {}) {
    const headers = new Headers(options.headers || {});
    const token = localStorage.getItem('ACCESS_TOKEN') || getCookie('auth_token') || '';
    if (token) headers.set('Authorization', 'Bearer ' + token);
    return fetch(url, { ...options, headers });
  }

  // --- Convertit un datetime local "YYYY-MM-DDTHH:mm" en clé open_<jour>_<moment>
  function datetimeToOpenKey(dtString) {
    if (!dtString) return '';
    const dt = new Date(dtString);
    if (isNaN(dt)) return '';
    // getDay() : 0=dimanche, 1=lundi, ... 6=samedi
    const jours = ['dimanche','lundi','mardi','mercredi','jeudi','vendredi','samedi'];
    const jour = jours[dt.getDay()];
    const h = dt.getHours();
    const moment = (h < 12) ? 'matin' : (h < 18 ? 'midi' : 'soir');
    return `open_${jour}_${moment}`;
  }

  // --- Affiche les résultats renvoyés par /predict
  function afficherResultats(data) {
    const items = data.items_rich || data.items || [];
    const cont = document.getElementById('predict-result');
    if (!cont) return;

    if (!items.length) {
      cont.innerHTML = `<div class="alert alert-warning">Aucun résultat.</div>`;
      return;
    }
    cont.innerHTML = items.map(it => `
      <div class="card mb-2 bg-dark text-white border-secondary">
        <div class="card-body d-flex justify-content-between">
          <div>
            <h6 class="card-title mb-1">#${it.rank} – ${it.details?.nom || 'Etablissement ' + it.etab_id}</h6>
            <p class="card-text small text-muted mb-0">${it.details?.adresse || ''}</p>
          </div>
          <div class="text-end">
            <span class="badge bg-primary">Score ${(it.score ?? 0).toFixed(3)}</span><br>
            ${typeof it.details?.rating === 'number' ? `<span class="small text-muted">Note ${(it.details.rating).toFixed(1)}</span>` : ''}
          </div>
        </div>
      </div>
    `).join('');
  }

  // --- Soumission du formulaire principal (prédiction)
  const form = document.getElementById('predict-form');
  if (form) {
    form.addEventListener('submit', async (e) => {
      e.preventDefault();

      // 1) Lire les champs
      const cp       = document.getElementById('cp')?.value.trim() ?? '';
      const desc     = document.getElementById('desc')?.value.trim() ?? '';
      const priceRaw = document.getElementById('price')?.value ?? '';     // peut être ''
      const kRaw     = document.getElementById('k')?.value ?? '';
      const k        = parseInt(kRaw, 10) || 10;                          // défaut 10 si vide
      const dtString = document.getElementById('datetime')?.value ?? '';  // jour + heure
      const openKey  = datetimeToOpenKey(dtString);

      const optionsSelect = document.getElementById('opts');
      const opts = optionsSelect
        ? Array.from(optionsSelect.selectedOptions).map(opt => opt.value)
        : [];

      // 2) Construire le payload (toujours inclure description & open)
      const payload = {
        description: desc || "",   // requis côté backend
        open: openKey || ""        // requis côté backend
      };
      if (priceRaw !== '') {
        const price = parseInt(priceRaw, 10);
        if (!Number.isNaN(price)) payload.price_level = price;
      }
      if (cp)          payload.city    = cp;
      if (opts.length) payload.options = opts;

      // --- LOG de debug (modif #2)
      console.log('payload /predict ->', payload);

      // 3) UI: état de chargement
      const cont = document.getElementById('predict-result');
      if (cont) cont.innerHTML = `<div class="alert alert-secondary">Calcul en cours…</div>`;

      // 4) Appel API
      try {
        const response = await authFetch(`/predict?k=${encodeURIComponent(k)}&use_ml=true`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        if (!response.ok) {
          let detail = '';
          try { detail = (await response.json()).detail ?? ''; } catch {}
          throw new Error(`HTTP ${response.status}${detail ? ' - ' + JSON.stringify(detail) : ''}`);
        }

        const data = await response.json();
        afficherResultats(data);

        const predId = data.prediction_id || data.id || '';
        const predInput = document.getElementById('prediction-id');
        if (predInput) predInput.value = predId;

        const btnWrap = document.getElementById('feedback-button-wrapper');
        if (btnWrap) btnWrap.classList.remove('d-none');

      } catch (err) {
        if (cont) cont.innerHTML = `<div class="alert alert-danger">Erreur : ${err.message}</div>`;
        console.error(err);
      }
    });
  }

  // --- Bouton pour ouvrir la modale de feedback
  const btn = document.getElementById('open-feedback-btn');
  if (btn) {
    btn.addEventListener('click', () => {
      const modalEl = document.getElementById('feedbackModal');
      if (!modalEl) return;
      const modal = new bootstrap.Modal(modalEl);
      modal.show();
    });
  }

  // --- Soumission du formulaire de feedback
  const fbForm = document.getElementById('feedback-form');
  if (fbForm) {
    fbForm.addEventListener('submit', async (e) => {
      e.preventDefault();

      const predId    = document.getElementById('prediction-id')?.value ?? '';
      const ratingRaw = document.getElementById('rating')?.value ?? ''; // peut être ''
      const comment   = document.getElementById('comment')?.value.trim() ?? '';

      if (!predId) {
        alert("Identifiant de prédiction manquant.");
        return;
      }

      const payload = { prediction_id: predId };
      if (ratingRaw !== '') {
        const r = parseInt(ratingRaw, 10);
        if (!Number.isNaN(r)) payload.rating = r;
      }
      if (comment) payload.comment = comment;

      try {
        const resp = await authFetch('/feedback', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        if (!resp.ok) {
          let detail = '';
          try { detail = (await resp.json()).detail ?? ''; } catch {}
          throw new Error(`HTTP ${resp.status}${detail ? ' - ' + JSON.stringify(detail) : ''}`);
        }

        // Fermer la modale + alerte
        const modalEl = document.getElementById('feedbackModal');
        if (modalEl) bootstrap.Modal.getInstance(modalEl)?.hide();

        const cont = document.getElementById('predict-result');
        if (cont) {
          const okAlert = document.createElement('div');
          okAlert.className = 'alert alert-success';
          okAlert.textContent = 'Merci pour votre retour !';
          cont.prepend(okAlert);
          setTimeout(() => okAlert.remove(), 3500);
        }

        // Optionnel : reset des champs feedback
        const rEl = document.getElementById('rating');
        const cEl = document.getElementById('comment');
        if (rEl) rEl.value = '';
        if (cEl) cEl.value = '';

      } catch (err) {
        alert('Erreur lors de l’envoi du feedback : ' + err.message);
        console.error(err);
      }
    });
  }
});

document.addEventListener('click', async function(event) {
  const target = event.target.closest('.result-item');
  if (!target) return; // On n'a pas cliqué sur une carte

  const etabId = target.dataset.etabId;
  if (!etabId) return;

  try {
    // 1) Récupérer les détails depuis votre API
    const resp = await authFetch(`/restaurant/${encodeURIComponent(etabId)}`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const details = await resp.json();

    // 2) Construire le contenu HTML pour la modale
    const body = document.getElementById('detailModalBody');
    const title = document.getElementById('detailModalTitle');
    title.textContent = details.nom || 'Restaurant';
    let html = '';

    // Description
    if (details.description) {
      html += `<p>${details.description}</p>`;
    }
    // Adresse
    html += `<p><strong>Adresse :</strong> ${details.adresse || 'Non renseignée'}</p>`;

    // Téléphone
    if (details.telephone) {
      html += `<p><strong>Téléphone :</strong> ${details.telephone}</p>`;
    }
    // Site web
    if (details.site_web) {
      html += `<p><strong>Site :</strong> <a href="${details.site_web}" target="_blank" class="link-info">${details.site_web}</a></p>`;
    }
    // Rating
    if (typeof details.rating === 'number') {
      html += `<p><strong>Note :</strong> ${details.rating.toFixed(1)}/5</p>`;
    }
    // Horaires (format libre, à adapter)
    if (details.hours) {
      html += `<p><strong>Horaires :</strong><br><span class="small">${details.hours}</span></p>`;
    }
    // Carte
    if (details.latitude && details.longitude) {
      const lat = details.latitude;
      const lon = details.longitude;
      // Ici on intègre une iframe Google Maps (nécessite une clé si vous utilisez l'API Google). 
      // Remplacez par un embed de votre choix (OpenStreetMap via Leaflet, par exemple).
      html += `
        <div class="ratio ratio-16x9 mt-3">
          <iframe src="https://www.google.com/maps?q=${lat},${lon}&hl=fr&z=15&output=embed" 
                  width="100%" height="100%" style="border:0" allowfullscreen loading="lazy"></iframe>
        </div>`;
    }

    // Bouton pour charger les avis
    html += `<div class="mt-3">
               <button class="btn btn-outline-info" id="loadReviewsBtn">Voir les avis</button>
               <div id="reviewsContainer" class="mt-2"></div>
             </div>`;

    body.innerHTML = html;

    // 3) Afficher la modale
    const modal = new bootstrap.Modal(document.getElementById('detailModal'));
    modal.show();

    // 4) Événement pour charger les avis au clic
    document.getElementById('loadReviewsBtn').addEventListener('click', async () => {
      const container = document.getElementById('reviewsContainer');
      container.innerHTML = '<p class="text-muted">Chargement des avis…</p>';
      const rResp = await authFetch(`/restaurant/${encodeURIComponent(etabId)}/reviews`);
      if (!rResp.ok) {
        container.innerHTML = '<p class="text-danger">Impossible de charger les avis.</p>';
        return;
      }
      const reviews = await rResp.json();
      if (!reviews.length) {
        container.innerHTML = '<p class="text-muted">Aucun avis.</p>';
      } else {
        container.innerHTML = reviews.map(rv => `
          <div class="border-bottom py-1">
            <small class="text-muted">${rv.date.substr(0, 10)} — Note ${rv.rating}/5</small><br>
            <span>${rv.comment || ''}</span>
          </div>
        `).join('');
      }
    });

  } catch (err) {
    console.error(err);
    alert('Erreur lors du chargement des détails.');
  }
});