FROM crazymax/fail2ban:latest
RUN apk add --no-cache redis
# (optionnel) COPY de tes filtres / jails :
COPY jail.d/ /etc/fail2ban/jail.d/
COPY filter.d/ /etc/fail2ban/filter.d/
COPY action.d/redis-ban.conf /etc/fail2ban/action.d/redis-ban.conf