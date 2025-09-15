import { test, expect, request as pwRequest } from '@playwright/test';

test('historique: user -> login -> prédiction -> ouverture détail', async ({ page, request, context, baseURL }) => {
  // baseURL garanti par la config
  expect(baseURL).toBeTruthy();

  const username = `u${Date.now()}`;
  const email = `${username}@e2e.test`;
  const password = 'pass1234!';

  // 1) Register (JSON)
  const reg = await request.post('/users', {
    json: { username, email, password }
  });
  expect(reg.ok()).toBeTruthy();

  // 2) Login (form OAuth2 + bypass Turnstile)
  const login = await request.post('/auth/web/token', {
    form: {
      username,
      password,
      'cf-turnstile-response': '' // TURNSTILE_DEV_BYPASS=1 => OK
    }
  });
  expect(login.ok()).toBeTruthy();
  const { access_token } = await login.json();

  // 3) Injecter le cookie côté UI
  await context.addCookies([
    { name: 'ACCESS_TOKEN', value: access_token, url: baseURL! }
  ]);

  // 4) Créer une prédiction côté API (avec Authorization)
  const apiCtx = await pwRequest.newContext({
    baseURL,
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

  // 5) Ouvrir /history et cliquer sur la prédiction
  await page.goto('/history');
  // La carte contient "Prédiction #<8 chars>"
  const shortId = String(pred.id).slice(0, 8);
  const card = page.locator('.pred-item', { hasText: shortId });
  await expect(card).toBeVisible();

  await card.click();

  // 6) Vérifier la modale de détail
  await expect(page.locator('#predDetailModal .modal-title, #predDetailBody')).toBeVisible();
  await expect(page.getByText(new RegExp(`Prédiction #${shortId}`))).toBeVisible();
});
