# Dockerfile.websocket-server

# 1. Base Image: Use an official Node.js image
FROM node:22 AS builder

# 2. Set working directory
WORKDIR /app

#3. Install node modules.
# Do this before specifically so that modules are cached and not reinstalled when code is modified.
COPY package.json package-lock.json ./
RUN npm install

# 4. Copy Files
COPY . .

# 5. Expose the Node.js application's own service port (5000)
# EXPOSE 5000

CMD [ "npm", "start" ]