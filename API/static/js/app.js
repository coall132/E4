// static/js/app.js
(function() {
  function getToken() { return localStorage.getItem('ACCESS_TOKEN') || ''; }
  function getApiKey() { return localStorage.getItem('API_KEY') || ''; }
  window.authFetch = async function(url, init = {}) {
    const headers = new Headers(init.headers || {});
    const token = getToken();
    if (token) headers.set('Authorization', 'Bearer ' + token);
    return fetch(url, { ...init, headers });
  };
  window.__auth = { getToken, getApiKey };
})();
