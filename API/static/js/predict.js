// static/js/predict.js
(function() {
  const form = document.getElementById('predict-form');
  const box  = document.getElementById('predict-result');
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    box.innerHTML = '<div class="text-muted">… traitement …</div>';
    const fd = new FormData(form);
    const payload = {
      price_level: parseInt(fd.get('price_level') || '2', 10),
      city: (fd.get('city') || '').trim(),
      open: (fd.get('open') || '').trim(),
      options: (fd.get('options') || '').split(',').map(s => s.trim()).filter(Boolean),
      description: (fd.get('description') || '').trim()
    };
    const useMl = document.getElementById('use_ml').checked;
    try {
      const url = `/predict?k=10&use_ml=${useMl ? 'true' : 'false'}`;
      const r = await authFetch(url, {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify(payload)
      });
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const data = await r.json();
      const items = data.items_rich || data.items || [];
      box.innerHTML = `
        <div class="mb-2 text-muted">latence: ${data.latency_ms ?? 'NA'} ms — modèle: ${data.model_version ?? 'dev'}</div>
        <div class="list-group">
          ${items.map(it => `
            <div class="list-group-item list-group-item-dark">
              <div class="d-flex justify-content-between">
                <div>
                  <div class="fw-semibold">#${it.rank} — ${it.details?.nom || ('id ' + it.etab_id)}</div>
                  <div class="small text-muted">${it.details?.adresse || ''}</div>
                </div>
                <div class="text-end">
                  <span class="badge bg-primary">score ${(it.score ?? 0).toFixed(4)}</span><br>
                  ${it.details?.rating ? `<span class="small text-muted">note ${(it.details.rating).toFixed(1)}</span>` : ''}
                </div>
              </div>
            </div>
          `).join('')}
        </div>
        <div class="mt-3 small text-muted">ID prédiction : ${data.prediction_id || data.id || 'NA'}</div>
      `;
    } catch (err) {
      box.innerHTML = `<div class="text-danger">Erreur : ${err}</div>`;
    }
  });
})();
