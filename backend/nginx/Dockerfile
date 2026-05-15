# Use the official Nginx image
FROM nginx:stable-alpine

# Install envsubst (part of gettext)
RUN apk add --no-cache gettext

# Copy the template file into a dedicated folder
COPY ./backend.conf.template /etc/nginx/templates/backend.conf.template

# Update the default CMD to first substitute env vars and then run nginx
CMD envsubst '$WEBSOCKET_SERVER_PORT $MEDIA_SERVER_PORT' \
    < /etc/nginx/templates/backend.conf.template \
    > /etc/nginx/conf.d/backend.conf && \
    exec nginx -g 'daemon off;'
