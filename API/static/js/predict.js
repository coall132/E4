// utils pour récupérer le token et faire fetch autorisé
function authFetch(url, options={}) {
  const headers = new Headers(options.headers || {});
  const token = localStorage.getItem('ACCESS_TOKEN') || '';
  if (token) headers.set('Authorization', 'Bearer ' + token);
  return fetch(url, {...options, headers});
}

// Convertit un datetime (local) en clef open_jour_moment
function datetimeToOpenKey(dtString) {
  // dtString "2025-09-02T19:30"
  const dt = new Date(dtString);
  if (isNaN(dt)) return '';
  const jour = ['lundi','mardi','mercredi','jeudi','vendredi','samedi','dimanche'][dt.getDay()];
  const heure = dt.getHours();
  const moment = (heure < 12) ? 'matin' : (heure < 18 ? 'midi' : 'soir');
  return `open_${jour}_${moment}`;
}

// Soumission du formulaire principal
document.getElementById('predict-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  // récupérer les valeurs
  const cp = document.getElementById('cp').value.trim();
  const desc = document.getElementById('desc').value.trim();
  const price = document.getElementById('price').value;
  const k = parseInt(document.getElementById('k').value, 10) || 10;
  const dtString = document.getElementById('datetime').value;
  const openKey = datetimeToOpenKey(dtString);

  const optionsSelect = document.getElementById('opts');
  const opts = Array.from(optionsSelect.selectedOptions).map(opt => opt.value);

  const payload = {
    price_level: parseInt(price),
    city: cp,
    description: desc,
    options: opts,
    open: openKey
  };

  try {
    const response = await authFetch(`/predict?k=${k}&use_ml=true`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    if (!response.ok) throw new Error('HTTP ' + response.status);
    const data = await response.json();
    // Affichage résultats
    afficherResultats(data);
    // Préparer feedback
    document.getElementById('prediction-id').value = data.prediction_id || data.id;
    new bootstrap.Modal(document.getElementById('feedbackModal')).show();
  } catch(err) {
    console.error(err);
    document.getElementById('predict-result').innerHTML =
      `<div class="alert alert-danger">Erreur : ${err.message}</div>`;
  }
});

// Affiche les items retournés
function afficherResultats(data) {
  const items = data.items_rich || data.items || [];
  const cont = document.getElementById('predict-result');
  cont.innerHTML = items.map(it => `
    <div class="card mb-2 bg-dark text-white">
      <div class="card-body d-flex justify-content-between">
        <div>
          <h6 class="card-title mb-1">#${it.rank} – ${it.details?.nom || 'id '+it.etab_id}</h6>
          <p class="card-text small text-muted">${it.details?.adresse || ''}</p>
        </div>
        <div class="text-end">
          <span class="badge bg-primary">Score ${(it.score??0).toFixed(3)}</span><br>
          ${it.details?.rating ? `<span class="small text-muted">Note ${(it.details.rating).toFixed(1)}</span>` : ''}
        </div>
      </div>
    </div>
  `).join('');
}

// Envoi du feedback
document.getElementById('feedback-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const predId = document.getElementById('prediction-id').value;
  const rating = parseInt(document.getElementById('rating').value, 10);
  const comment = document.getElementById('comment').value;
  try {
    const resp = await authFetch('/feedback', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        prediction_id: predId,
        rating,
        comment: comment || undefined
      })
    });
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    document.getElementById('feedbackModal').querySelector('.modal-title').textContent = 'Merci !';
    document.getElementById('feedbackModal').querySelector('form').innerHTML =
      '<div class="modal-body"><p>Feedback envoyé avec succès.</p></div>';
  } catch(err) {
    alert('Erreur lors de l’envoi du feedback: ' + err.message);
  }
});
