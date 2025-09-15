import { test, expect } from '@playwright/test';

test('historique: user -> login -> prédiction -> ouverture détail', async ({ page, request, baseURL, context }) => {

  const uid = Date.now();
  const username = `user_${uid}`;
  const email = `user_${uid}@test.local`;
  const password = 'P@ssw0rd!123';


  const reg = await request.post(`${baseURL}/users`, {
    data: {
      username,
      email,
      password,
      'cf-turnstile-response': 'test', 
    },
  });
  expect(reg.ok()).toBeTruthy();

  const tok = await request.post(`${baseURL}/auth/web/token`, {
    form: {
      username,
      password,
      'cf-turnstile-response': 'test', 
    },
  });
  expect(tok.ok()).toBeTruthy();
  const tokJson = await tok.json();
  const token: string = tokJson.access_token;
  expect(token).toBeTruthy();


  const host = new URL(baseURL!).hostname;
  await context.addCookies([{
    name: 'ACCESS_TOKEN',
    value: token,
    domain: host,
    path: '/',
    httpOnly: false,
    secure: false,
    sameSite: 'Lax'
  }]);

  await page.goto('/'); 
  await page.evaluate((t) => localStorage.setItem('ACCESS_TOKEN', t), token);

  await page.goto('/predict');
  await page.fill('#desc', 'italien terrasse');         
  const kSelect = page.locator('#k');
  if (await kSelect.count()) {
    await kSelect.selectOption({ value: '5' }).catch(() => {});
  }

  await page.click('#predict-form [type="submit"]');

  await page.waitForSelector('#predict-result .result-item', { timeout: 30_000 });

  await page.goto('/history');
  await page.waitForSelector('.pred-item', { timeout: 30_000 });

  await page.click('.pred-item');

  await page.waitForSelector('#predDetailModal .modal-body', { timeout: 15_000 });
  await expect(page.locator('#predDetailBody')).toContainText('Items recommandés');

  const firstItem = page.locator('#itemsList .result-item').first();
  if (await firstItem.count()) {
    await firstItem.click();
    await page.waitForSelector('#detailModal .modal-body', { timeout: 15_000 });
    await expect(page.locator('#detailModalTitle')).not.toHaveText('');
  }
});
