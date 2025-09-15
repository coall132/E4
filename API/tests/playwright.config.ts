import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: 'API/tests',
  reporter: [['html', { open: 'never' }], ['line']],
  use: {
    baseURL: process.env.BASE_URL || 'http://127.0.0.1:8000',
  },
  timeout: 120_000,
});
