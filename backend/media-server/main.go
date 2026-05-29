package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/joho/godotenv"
	"github.com/pion/webrtc/v4"
	"github.com/pion/webrtc/v4/pkg/media"
	"github.com/redis/go-redis/v9"

	"media-server/config"
)

// OfferRequest defines the JSON structure sent by the client to /connect,
// containing the session ID.
type OfferRequest struct {
	SessionID string `json:"sessionId"`
}

// OfferResponse defines the JSON structure returned from /connect.
// It contains the generated SDP offer, the offer type (always "offer"),
// and a unique connection ID to be used in subsequent signaling.
type OfferResponse struct {
	ID    string   `json:"id"`
	SDP   string   `json:"sdp"`
	Type  string   `json:"type"`
	Views []string `json:"views,omitempty"`
}

// AnswerRequest defines the JSON structure sent by the client to /answer,
// containing the connection ID (from the offer) and the SDP answer.
type AnswerRequest struct {
	SessionID string `json:"sessionId"`
	SDP       string `json:"sdp"`
	Type      string `json:"type"`
}

// PeerConnection holds a WebRTC PeerConnection and a channel to signal when to stop streaming.
type PeerConnection struct {
	Connection *webrtc.PeerConnection
	StopChan   chan struct{}
	Tracks     map[string]*webrtc.TrackLocalStaticSample
}

// TurnServer struct to store TURN server details
type TurnServer struct {
	URL        string
	Username   string
	Credential string
}

type FrameData struct {
	Data string  `json:"frame"`
	I    int     `json:"index"`
	TS   float64 `json:"ts"` // New: Unix epoch timestamp (seconds)
}

// Global map of used variables
var (
	peerConnections   = make(map[string]PeerConnection)
	peerConnectionsMu sync.Mutex
	ctx               = context.Background()
	rdb               *redis.Client
	cfg               *config.Config
	turnServers       []TurnServer
	api               *webrtc.API
)

// handleOffer is called when a client clicks connect.
// It creates a new PeerConnection with a unique video track, generates an SDP offer,
// stores the connection in a global map, and returns the offer to the client.
func handleOffer(w http.ResponseWriter, r *http.Request) {

	var req OfferRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		log.Printf("Failed to decode offer request: %v", err)
		return
	}

	peerConnectionsMu.Lock()
	_, connExists := peerConnections[req.SessionID]
	peerConnectionsMu.Unlock()

	if connExists {
		http.Error(w, "WebRTC Connection already exists! Only 1 streaming client supported per user.", http.StatusBadRequest)
		log.Printf("WebRTC Connection already exists for session %s", req.SessionID)
		return
	}

	views := getSessionViews(req.SessionID)
	if len(views) == 0 {
		http.Error(w, fmt.Sprintf("Simulation has not started yet for session %s.", req.SessionID), http.StatusBadRequest)
		log.Printf("Simulation has not started yet for session %s", req.SessionID)
		return
	}

	log.Printf("Received offer request for session %s", req.SessionID)

	// Prepare the config.
	ICEServers := []webrtc.ICEServer{
		{
			URLs: []string{"stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"},
		},
	}

	// Add TURN servers to the configuration.
	for _, server := range turnServers {
		ICEServers = append(ICEServers, webrtc.ICEServer{
			URLs:       []string{server.URL},
			Username:   server.Username,
			Credential: server.Credential,
		})
	}

	fmt.Printf("ICEServers: %v\n", ICEServers)

	config := webrtc.Configuration{
		ICEServers: ICEServers,
	}

	// Create a new PeerConnection using the above configuration.
	peerConnection, err := api.NewPeerConnection(config)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to create PeerConnection: %v", err), http.StatusInternalServerError)
		return
	}

	// Create a new video track that expects data in H264 format.
	var mimeType string
	switch cfg.Codec {
	case "h264":
		mimeType = webrtc.MimeTypeH264
	case "vp8":
		mimeType = webrtc.MimeTypeVP8
	case "vp9":
		mimeType = webrtc.MimeTypeVP9
	}

	videoTracks := make(map[string]*webrtc.TrackLocalStaticSample)
	for _, view := range views {
		trackID := fmt.Sprintf("video-%s", view)
		streamID := fmt.Sprintf("stream-%s", view)
		videoTrack, trackErr := webrtc.NewTrackLocalStaticSample(
			webrtc.RTPCodecCapability{MimeType: mimeType},
			trackID,
			streamID,
		)
		if trackErr != nil {
			http.Error(w, fmt.Sprintf("Failed to create video track for view %s: %v", view, trackErr), http.StatusInternalServerError)
			return
		}

		if _, trackErr = peerConnection.AddTrack(videoTrack); trackErr != nil {
			http.Error(w, fmt.Sprintf("Failed to add video track for view %s: %v", view, trackErr), http.StatusInternalServerError)
			return
		}

		videoTracks[view] = videoTrack
	}

	// Create an SDP offer.
	offer, err := peerConnection.CreateOffer(nil)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to create offer: %v", err), http.StatusInternalServerError)
		return
	}

	gatherComplete := webrtc.GatheringCompletePromise(peerConnection)

	// Set the local description to the generated offer.
	if err = peerConnection.SetLocalDescription(offer); err != nil {
		http.Error(w, fmt.Sprintf("Failed to set local description: %v", err), http.StatusInternalServerError)
		return
	}

	<-gatherComplete

	// Retrieve the connection ID from the request.
	connID := req.SessionID

	// Store the PeerConnection in our global map.
	peerConnectionsMu.Lock()
	peerConnections[connID] = PeerConnection{Connection: peerConnection, StopChan: make(chan struct{}), Tracks: videoTracks}
	peerConnectionsMu.Unlock()

	// Prepare and send the offer response.
	resp := OfferResponse{
		SDP:   peerConnection.LocalDescription().SDP,
		Type:  peerConnection.LocalDescription().Type.String(),
		Views: views,
	}
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		log.Printf("Failed to encode offer response: %v", err)
	}

	// Register a callback to detect when the connection state changes.
	peerConnection.OnConnectionStateChange(func(state webrtc.PeerConnectionState) {
		log.Printf("PeerConnection for connection %s state changed: %s", connID, state.String())
		if state == webrtc.PeerConnectionStateFailed ||
			state == webrtc.PeerConnectionStateClosed ||
			state == webrtc.PeerConnectionStateDisconnected {
			// Remove this connection from the global map.
			peerConnectionsMu.Lock()
			if pc, exists := peerConnections[connID]; exists {
				close(pc.StopChan)
				delete(peerConnections, connID)
			}
			peerConnectionsMu.Unlock()
			// Close the connection if it hasn't been closed already.
			if err := peerConnection.Close(); err != nil {
				log.Printf("Error closing peer connection %s: %v", connID, err)
			}
			log.Printf("Cleaned up connection: %s", connID)
		}

		if state == webrtc.PeerConnectionStateConnected {
			log.Println("Starting video stream for connection", connID)
			for viewName, track := range videoTracks {
				go streamVideo(connID, viewName, track)
			}
		}
	})

	peerConnection.OnICEConnectionStateChange(func(state webrtc.ICEConnectionState) {
		log.Println("ICE Connection State has changed:", state)
	})

	peerConnection.OnICECandidate(func(candidate *webrtc.ICECandidate) {
		if candidate != nil {
			log.Println("Received ICE candidate:", candidate)
			// Handle sending the candidate to the client (via signaling)
		}
	})
}

// handleAnswer receives the client's SDP answer and sets it as the remote description
// for the corresponding PeerConnection.
func handleAnswer(w http.ResponseWriter, r *http.Request) {
	var answer AnswerRequest
	if err := json.NewDecoder(r.Body).Decode(&answer); err != nil {
		http.Error(w, fmt.Sprintf("Failed to decode answer: %v", err), http.StatusBadRequest)
		return
	}

	// Retrieve the corresponding PeerConnection using the provided connection ID.
	peerConnectionsMu.Lock()
	peerConnection, exists := peerConnections[answer.SessionID]
	peerConnectionsMu.Unlock()

	if !exists {
		http.Error(w, "Peer connection not found.", http.StatusNotFound)
		return
	}

	// Create a session description from the received answer and set it as the remote description.
	remoteDesc := webrtc.SessionDescription{
		Type: webrtc.SDPTypeAnswer,
		SDP:  answer.SDP,
	}
	if err := peerConnection.Connection.SetRemoteDescription(remoteDesc); err != nil {
		http.Error(w, fmt.Sprintf("Failed to set remote description: %v", err), http.StatusInternalServerError)
		return
	}

	// Return success.
	w.WriteHeader(http.StatusOK)
}

// Start a goroutine to simulate streaming video frames.
func streamVideo(connID string, viewName string, videoTrack *webrtc.TrackLocalStaticSample) {
	defer func() {
		if err := recover(); err != nil {
			log.Printf("Recovered from Error in streamVideo for connection %s: %v", connID, err)
		}
	}()

	// Default frame duration based on configuration (fallback duration)
	defaultFrameDuration := time.Duration(1000/cfg.FrameRate) * time.Millisecond
	var lastTimestamp time.Time

	// Retrieve the stop channel for this connection.
	peerConnectionsMu.Lock()
	stopChan := peerConnections[connID].StopChan
	peerConnectionsMu.Unlock()

	for {
		select {
		case <-stopChan:
			log.Printf("Stopping video stream for connection %s view %s", connID, viewName)
			return
		default:
			redisKeys := getViewRedisKeys(connID, viewName)

			timeoutCtx, cancel := context.WithTimeout(ctx, time.Minute)
			result, err := rdb.BLPop(timeoutCtx, time.Minute, redisKeys...).Result()
			cancel()

			if err == redis.Nil || err == context.DeadlineExceeded {
				log.Printf("Redis timeout reached for connection %s. Ending session.", connID)
				terminateSession(connID, "No frames received within 1 minute.")
				return
			} else if err != nil {
				log.Printf("Error retrieving frame from Redis for connection %s: %v", connID, err)
				return
			}

			if len(result) < 2 {
				continue
			}

			var frameData FrameData
			if err := json.Unmarshal([]byte(result[1]), &frameData); err != nil {
				log.Printf("Error unmarshalling frame data for connection %s: %v", connID, err)
				continue
			}

			packetBytes, err := base64.StdEncoding.DecodeString(frameData.Data)
			if err != nil {
				log.Printf("Error decoding frame data for connection %s: %v", connID, err)
				continue
			}

			// Compute the sample duration using the provided timestamp.
			currentTimestamp := time.Unix(0, int64(frameData.TS*1e9))
			var sampleDuration time.Duration
			if lastTimestamp.IsZero() {
				sampleDuration = defaultFrameDuration
			} else {
				sampleDuration = currentTimestamp.Sub(lastTimestamp)
				if sampleDuration <= 0 {
					sampleDuration = defaultFrameDuration
				}
			}
			lastTimestamp = currentTimestamp

			err = videoTrack.WriteSample(media.Sample{
				Data:     packetBytes,
				Duration: sampleDuration,
			})
			if err != nil {
				log.Printf("Error writing video sample for connection %s: %v", connID, err)
				return
			}
		}
	}
}

func getViewRedisKeys(connID string, viewName string) []string {
	primaryKey := fmt.Sprintf("sessions-%s:images:%s", connID, viewName)
	return []string{primaryKey}
}

func getSessionViews(sessionID string) []string {
	viewsSetKey := fmt.Sprintf("sessions-%s:image_views", sessionID)
	views, err := rdb.SMembers(ctx, viewsSetKey).Result()
	if err != nil {
		log.Printf("Failed to read image views for session %s: %v", sessionID, err)
	}

	if len(views) == 0 {
		return []string{}
	}

	for i := range views {
		views[i] = strings.TrimSpace(views[i])
	}

	sort.Strings(views)

	return views
}

// terminateSession handles closing the WebRTC connection and notifying the user.
func terminateSession(connID string, message string) {
	peerConnectionsMu.Lock()
	if pc, exists := peerConnections[connID]; exists {
		close(pc.StopChan)
		delete(peerConnections, connID)
		pc.Connection.Close()
	}
	peerConnectionsMu.Unlock()

	log.Printf("Session %s terminated: %s", connID, message)
}

func loadEnv() {
	// Load environment variables from .env file
	if err := godotenv.Load(); err != nil {
		log.Print("Error loading .env file: ", err)
	}
}

func loadConfig() {
	// Load the configuration file
	var err error
	cfg, err = config.LoadConfig("config/config.json")
	if err != nil {
		log.Fatalf("Error loading config file: %v", err)
	}
	log.Printf("Loaded config: %+v", cfg)
}

func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Set CORS headers
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func recoverMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if err := recover(); err != nil {
				log.Printf("Recovered from panic: %v", err)
				http.Error(w, "Internal Server Error", http.StatusInternalServerError)
			}
		}()
		next.ServeHTTP(w, r)
	})
}

func chainMiddleware(h http.Handler, middlewares ...func(http.Handler) http.Handler) http.Handler {
	for _, m := range middlewares {
		h = m(h)
	}
	return h
}

func registerHTTPHandlers() {
	http.Handle("/offer", chainMiddleware(http.HandlerFunc(handleOffer), corsMiddleware, recoverMiddleware))
	http.Handle("/answer", chainMiddleware(http.HandlerFunc(handleAnswer), corsMiddleware, recoverMiddleware))
}

func normalizeICEServerURL(raw string) (string, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return "", fmt.Errorf("empty URL")
	}
	lower := strings.ToLower(raw)
	if strings.HasPrefix(lower, "stun:") || strings.HasPrefix(lower, "turn:") || strings.HasPrefix(lower, "turns:") {
		return raw, nil
	}
	// Allow host/IP with optional port (e.g. 192.168.1.10 or 192.168.1.10:3478).
	if !strings.Contains(raw, ":") {
		raw = raw + ":3478"
	}
	return "turn:" + raw, nil
}

func loadTURNServers() {
	// Parse TURN servers from the environment variables
	for i := 1; ; i++ {
		prefix := "TURN_SERVER_" + strconv.Itoa(i) + "_"
		rawURL := os.Getenv(prefix + "URL")
		if rawURL == "" {
			break // No more TURN servers
		}
		url, err := normalizeICEServerURL(rawURL)
		if err != nil {
			log.Printf("Skipping TURN server %d: %v", i, err)
			continue
		}
		username := os.Getenv(prefix + "USERNAME")
		credential := os.Getenv(prefix + "CREDENTIAL")
		if credential == "" {
			credential = os.Getenv(prefix + "PASSWORD")
		}

		turnServers = append(turnServers, TurnServer{
			URL:        url,
			Username:   username,
			Credential: credential,
		})
	}

	fmt.Printf("Turn servers: %v\n", turnServers)
}

func loadUDPPorts() (uint16, uint16) {
	minPortStr, exists := os.LookupEnv("MEDIA_SERVER_UDP_PORT_MIN")
	if !exists {
		log.Print("MEDIA_SERVER_UDP_PORT_MIN is not set, using default 49152")
		minPortStr = "49152"
	}
	minPort, err := strconv.Atoi(minPortStr)
	if err != nil {
		log.Fatalf("Invalid value for MEDIA_SERVER_UDP_PORT_MIN: %v", err)
	}

	maxPortStr, exists := os.LookupEnv("MEDIA_SERVER_UDP_PORT_MAX")
	if !exists {
		log.Print("MEDIA_SERVER_UDP_PORT_MAX is not set, using default 65535")
		maxPortStr = "65535"
	}
	maxPort, err := strconv.Atoi(maxPortStr)
	if err != nil {
		log.Fatalf("Invalid value for MEDIA_SERVER_UDP_PORT_MAX: %v", err)
	}

	return uint16(minPort), uint16(maxPort)
}

func main() {

	// Load .env file
	loadEnv()

	// Load the configuration file.
	loadConfig()

	// Read environment variables
	port, exists := os.LookupEnv("MEDIA_SERVER_PORT")
	if !exists {
		port = "8080"
	}
	loadTURNServers()

	// Initialize the Redis client.
	redisHost, exists := os.LookupEnv("REDIS_HOST")
	if !exists {
		log.Print("REDIS_HOST not set, using default localhost")
		redisHost = "localhost"
	}
	redisPort, exists := os.LookupEnv("REDIS_PORT")
	if !exists {
		log.Print("REDIS_PORT not set, using default 6379")
		redisPort = "6379"
	}
	// Initialize Redis client
	rdb = redis.NewClient(&redis.Options{
		Addr: fmt.Sprintf("%s:%s", redisHost, redisPort),
	})

	registerHTTPHandlers()

	// Load UDP ports that will be used for WebRTC communication
	minUDPPort, maxUDPPort := loadUDPPorts()

	// Set up WebRTC API with custom UDP port range
	settingEngine := webrtc.SettingEngine{}
	settingEngine.SetEphemeralUDPPortRange(minUDPPort, maxUDPPort)
	if publicIP := strings.TrimSpace(os.Getenv("MEDIA_SERVER_PUBLIC_IP")); publicIP != "" {
		settingEngine.SetNAT1To1IPs([]string{publicIP}, webrtc.ICECandidateTypeHost)
		log.Printf("ICE host candidates use public IP: %s", publicIP)
	}
	fmt.Printf("Ephemeral UDP Port Range %d:%d\n", minUDPPort, maxUDPPort)

	// Create a new WebRTC API instance with the custom setting engine.
	api = webrtc.NewAPI(webrtc.WithSettingEngine(settingEngine))

	// Start the HTTP server.
	addr := fmt.Sprintf(":%s", port)
	log.Printf("Starting WebRTC signaling server on %s", addr)
	if err := http.ListenAndServe(addr, nil); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
