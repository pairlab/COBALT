# Stage 1: Build the media server binary
FROM golang:1.23-alpine AS builder
# Install git (if needed for go mod downloads)
RUN apk add --no-cache git

WORKDIR /app

# Copy go.mod and go.sum first for caching dependency downloads
COPY go.mod go.sum ./
RUN go mod download

# Now copy the rest of your source code, including main.go, config files, etc.
COPY . .

# # Build a fully static binary (CGO disabled)
# RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o media-server .

# Run the binary
CMD ["go", "run", "main.go"]