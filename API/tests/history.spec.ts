// API/tests/history.spec.ts
import { test, expect, request as pwRequest } from '@playwright/test';

test('historique: user -> login -> ouverture détail prédiction seedée', async ({ page, context }) => {
  const base = process.env.BASE_URL || 'http://127.0.0.1:8000';

  // Contexte API "anonyme"
  const api = await pwRequest.newContext({ baseURL: base });

  // 1) Login utilisateur seedé
  const loginRes = await api.post('/auth/web/token', {
    form: {
      username: 'e2e',
      password: 'pass1234!',
      'cf-turnstile-response': '' // BYPASS=1 côté CI
    },
  });
  const loginText = await loginRes.text();
  expect(loginRes.ok(), `login failed: ${loginText}`).toBeTruthy();
  const { access_token } = JSON.parse(loginText);
  expect(access_token).toBeTruthy();

  // 2) Contexte API authentifié (pour verifier l’historique via l’API)
  const authed = await pwRequest.newContext({
    baseURL: base,
    extraHTTPHeaders: { Authorization: `Bearer ${access_token}` }
  });
  const histRes = await authed.get('/history/predictions');
  const histText = await histRes.text();
  expect(histRes.ok(), `history API failed: ${histText}`).toBeTruthy();
  const list = JSON.parse(histText);
  expect(Array.isArray(list) && list.length > 0, 'no seeded predictions').toBeTruthy();

  // 3) Cookie UI + navigation
  await context.addCookies([{ name: 'ACCESS_TOKEN', value: access_token, url: base }]);
  await page.goto(`${base}/history`);

  // 4) Cliquer la première carte de prédiction
  const firstCard = page.locator('.pred-item').first();
  await expect(firstCard).toBeVisible({ timeout: 15000 });
  await firstCard.click();

  // 5) Vérifier la modale détail + au moins un item
  await expect(page.locator('#predDetailModal, #predDetailBody')).toBeVisible({ timeout: 10000 });
  await expect(page.locator('#itemsList .result-item').first()).toBeVisible();
});
