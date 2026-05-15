# Use the official Nginx image
FROM nginx:stable-alpine

# Install envsubst (part of gettext)
RUN apk add --no-cache gettext

# Copy the template file into a dedicated folder
COPY ./frontend.conf.template /etc/nginx/templates/frontend.conf.template

# Update the default CMD to first substitute env vars and then run nginx
CMD envsubst '$PORT' \
    < /etc/nginx/templates/frontend.conf.template \
    > /etc/nginx/conf.d/frontend.conf && \
    exec nginx -g 'daemon off;'
