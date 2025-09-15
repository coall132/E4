// API/tests/history.spec.ts
import { test, expect } from '@playwright/test';

test('historique: user -> login -> ouverture détail prédiction seedée', async ({ page, request, context }, testInfo) => {
  const baseURL = process.env.BASE_URL || 'http://127.0.0.1:8000';

  // 1) Login API (utilisateur seedé e2e / pass1234!)
  const loginRes = await request.post(`${baseURL}/auth/web/token`, {
    form: { username: 'e2e', password: 'pass1234!', 'cf-turnstile-response': 'bypass' }
  });
  const loginText = await loginRes.text();
  expect(loginRes.ok(), `login failed: ${loginText}`).toBeTruthy();
  const { access_token } = JSON.parse(loginText);
  expect(access_token).toBeTruthy();

  // 2) Cookie pour l’accès aux pages HTML
  const { hostname } = new URL(baseURL);
  await context.addCookies([{
    name: 'ACCESS_TOKEN',
    value: access_token,
    domain: hostname,    // 127.0.0.1
    path: '/',
    httpOnly: false,
    secure: false,
    sameSite: 'Lax'
  }]);

  // 3) localStorage pour que le JS de la page mette bien le header Authorization
  await context.addInitScript((token, originHost) => {
    // Ne le fait que sur notre origine pour éviter les warnings
    if (window.location.hostname === originHost) {
      try { localStorage.setItem('ACCESS_TOKEN', token); } catch {}
    }
  }, access_token, hostname);

  // 4) Aller sur /history
  await page.goto(`${baseURL}/history`, { waitUntil: 'domcontentloaded' });

  // 5) Attendre l’appel /history/predictions et vérifier qu’il renvoie des données
  const resp = await page.waitForResponse(
    r => r.url().endsWith('/history/predictions') && r.ok(),
    { timeout: 15000 }
  );
  const predictions = await resp.json();
  expect(Array.isArray(predictions), 'predictions must be an array').toBeTruthy();
  expect(predictions.length, 'no predictions returned for user').toBeGreaterThan(0);

  // 6) Attendre l’affichage d’au moins une carte .pred-item et cliquer
  const firstCard = page.locator('.pred-item').first();
  await expect(firstCard).toBeVisible({ timeout: 15000 });
  await firstCard.click();

  // 7) Vérifier la modale de détail
  const modal = page.locator('#predDetailModal .modal-content');
  await expect(modal).toBeVisible({ timeout: 10000 });
  // au moins un item rendu
  await expect(page.locator('#itemsList .result-item').first()).toBeVisible({ timeout: 10000 });
});
