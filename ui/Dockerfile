FROM nginx:alpine

# Enable BuildKit inline cache
ARG BUILDKIT_INLINE_CACHE=1

COPY index.html /usr/share/nginx/html/index.html
COPY nginx.conf /etc/nginx/conf.d/default.conf
