import { test, expect, request as pwRequest } from '@playwright/test';

test('historique: user -> login -> prédiction -> ouverture détail', async ({ page, request, context }) => {
  // Récupère le baseURL défini dans playwright.config.ts ou l’ENV
  const base = (test.info().project.use.baseURL as string)
            || process.env.BASE_URL
            || 'http://127.0.0.1:8000';

  const username = `u${Date.now()}`;
  const email = `${username}@e2e.test`;
  const password = 'pass1234!';

  // 1) Register (JSON)
  const reg = await request.post('/users', {
    json: { username, email, password }
  });
  expect(reg.ok()).toBeTruthy();

  // 2) Login (form OAuth2 + bypass Turnstile via TURNSTILE_DEV_BYPASS=1)
  const login = await request.post('/auth/web/token', {
    form: { username, password, 'cf-turnstile-response': '' }
  });
  expect(login.ok()).toBeTruthy();
  const { access_token } = await login.json();
  expect(access_token).toBeTruthy();

  // 3) Cookie côté UI (utilise une URL absolue pour lier au bon domaine)
  await context.addCookies([{ name: 'ACCESS_TOKEN', value: access_token, url: base }]);

  // 4) Créer une prédiction via l’API authentifiée
  const apiCtx = await pwRequest.newContext({
    baseURL: base,
    extraHTTPHeaders: { Authorization: `Bearer ${access_token}` }
  });
  const predRes = await apiCtx.post('/predict?k=5&use_ml=false', {
    json: {
      description: 'italien terrasse',
      price_level: 2,
      city: 'Tours',
      open: 'soir_weekend',
      options: ['reservable', 'outdoorSeating']
    }
  });
  expect(predRes.ok()).toBeTruthy();
  const pred = await predRes.json();
  expect(pred).toHaveProperty('id');

  // 5) Ouvrir /history et cliquer sur la carte de la prédiction
  await page.goto(`${base}/history`);
  const shortId = String(pred.id).slice(0, 8);
  const card = page.locator('.pred-item', { hasText: shortId });
  await expect(card).toBeVisible({ timeout: 10_000 });

  await card.click();

  // 6) Vérifier la modale de détail
  await expect(page.locator('#predDetailModal, #predDetailBody')).toBeVisible();
  await expect(page.getByText(new RegExp(`Prédiction #${shortId}`))).toBeVisible();
});
