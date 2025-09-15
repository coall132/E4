import { test, expect, request as pwRequest } from '@playwright/test';

test('historique: user -> login -> prédiction -> ouverture détail', async ({ page, context }) => {
  const base = process.env.BASE_URL || 'http://127.0.0.1:8000';

  // Contexte API avec baseURL explicite
  const api = await pwRequest.newContext({ baseURL: base });

  const username = `u${Date.now()}`;
  const email = `${username}@e2e.test`;
  const password = 'pass1234!';

  // 1) Register
  const reg = await api.post('/users', {
    data: { username, email, password }, // ton endpoint accepte JSON brut
  });
  expect(reg.ok()).toBeTruthy();

  // 2) Login (OAuth2 form) — TURNSTILE_DEV_BYPASS=1 côté CI
  const login = await api.post('/auth/web/token', {
    form: { username, password, 'cf-turnstile-response': '' },
  });
  expect(login.ok()).toBeTruthy();
  const { access_token } = await login.json();
  expect(access_token).toBeTruthy();

  // 3) Cookie d’UI pour la session
  await context.addCookies([{ name: 'ACCESS_TOKEN', value: access_token, url: base }]);

  // 4) Créer une prédiction via l’API
  const predRes = await api.post('/predict?k=5&use_ml=false', {
    headers: { Authorization: `Bearer ${access_token}` },
    data: {
      description: 'italien terrasse',
      price_level: 2,
      city: 'Tours',
      open: 'soir_weekend',
      options: ['reservable', 'outdoorSeating'],
    },
  });
  expect(predRes.ok()).toBeTruthy();
  const pred = await predRes.json();
  expect(pred?.id).toBeTruthy();

  // 5) Ouvrir /history et cliquer sur la carte correspondante
  await page.goto(`${base}/history`);
  const shortId = String(pred.id).slice(0, 8);
  const card = page.locator('.pred-item', { hasText: shortId });
  await expect(card).toBeVisible({ timeout: 15000 });
  await card.click();

  // 6) Vérifier la modale de détail
  await expect(page.locator('#predDetailModal, #predDetailBody')).toBeVisible();
  await expect(page.getByText(new RegExp(`Prédiction #${shortId}`))).toBeVisible();
});