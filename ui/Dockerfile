FROM nginx:alpine

ARG BUILDKIT_INLINE_CACHE=1

COPY index.html /usr/share/nginx/html/index.html
# Template is rendered by nginx's built-in envsubst at startup using
# SCALPER_UPSTREAM (defaults to scalper:8000 for backward compatibility).
COPY default.conf.template /etc/nginx/templates/default.conf.template
ENV SCALPER_UPSTREAM=scalper:8000
