FROM redis:7-alpine

# Copie la conf
COPY docker/redis.conf /usr/local/etc/redis/redis.conf

# (facultatif) healthcheck intégré
HEALTHCHECK --interval=10s --timeout=3s --retries=3 CMD redis-cli ping | grep -q PONG || exit 1

# Lance Redis avec la conf
CMD ["redis-server", "/usr/local/etc/redis/redis.conf"]