import { WebSocketServer, WebSocket } from "ws";
import ntpClient from "ntp-client";
import http from "http";
import { URL } from "url";

// Import your session and simulation managers.
// These modules should export functions like createSession, getSession, deleteSession, etc.
import {
  getOrCreateSessionDetails,
  DeviceConfig,
  getReadyForSimulation,
  pushData,
  getResponse,
  cleanUpSession,
  registerConnection,
  unregisterConnection,
} from "./session_utils";
import { connect, quit } from "./redis_client";
import { getPublicIP } from "./utils";

const logger = console;

interface ExtWebSocket extends WebSocket {
  isAlive: boolean;
}

/**
 * Send a JSON message over the WebSocket.
 */
function sendMessage(ws: WebSocket, messageType: string, message: any): void {
  ws.send(JSON.stringify({ type: messageType, data: message }));
}

/**
 * Calculate the time offset between the local clock and a standard NTP server.
 * Uses ntp-time to query 'time.google.com' 10 times and returns the average offset (in seconds).
 */
async function calculateTimeOffset(): Promise<number> {
  let totalOffset = 0;
  let actualRequests = 0;
  const numRequests = 10;

  for (let i = 0; i < numRequests; i++) {
    try {
      // ntpTime returns the server time in milliseconds.
      ntpClient.getNetworkTime("time.google.com", 123, function (err, date) {
        if (err) {
          console.error(err);
          return;
        }

        if (date) {
          const localTime = Date.now();
          // Offset in seconds.
          const offset = (date.getTime() - localTime) / 1000;
          totalOffset += offset;
          actualRequests++;
        }
      });
    } catch (err) {
      if (err instanceof Error) {
        logger.error("Error calculating time offset:", err.message);
      }
    }
    // Wait 50ms between requests.
    await new Promise((res) => setTimeout(res, 50));
  }

  const timeOffset = actualRequests > 0 ? totalOffset / actualRequests : 0;
  logger.info(
    `Network Clock is ahead of the System clock by: ${timeOffset.toFixed(
      5
    )} seconds`
  );
  return timeOffset;
}

/**
 * Main handler for incoming WebSocket connections.
 */
async function handleConnection(ws: WebSocket, req: http.IncomingMessage) {
  // Parse query parameters from the URL.
  const reqUrl = req.url || "";
  const parsedUrl = new URL(reqUrl, `http://${req.headers.host}`);
  let sessionId = parsedUrl.searchParams.get("session_id") || "";
  let deviceId = parsedUrl.searchParams.get("device_id") || "";

  // Handle connection close.
  // WE MUST REGISTER THESE HANDLERS BEFORE ANYTHING IS PUSHED TO REDIS IN ORDER TO AVOID MEMORY LEAKS.
  ws.on("close", async () => {
    logger.info(`Device ${deviceId} disconnected from session ${sessionId}`);
    unregisterConnection(sessionId, deviceId);
    await cleanUpSession(sessionId);
  });

  ws.on("error", async (err) => {
    logger.error(`WebSocket error for device ${deviceId}:`, err);
    unregisterConnection(sessionId, deviceId);
    await ws.close();
  });

  let config: DeviceConfig;
  const configStr = parsedUrl.searchParams.get("config") || "";

  try {
    config = JSON.parse(configStr);
    logger.info(
      `Received client config from HTTP payload: ${JSON.stringify(config)}`
    );
  } catch (err) {
    logger.error("Error parsing client config from HTTP payload:", err);
    throw err;
  }

  if (sessionId || deviceId) {
    logger.info(
      `Received session_id ${sessionId} and device_id ${deviceId} from client.`
    );
  }

  // Generate new IDs if not provided.
  [sessionId, deviceId] = await getOrCreateSessionDetails(
    sessionId,
    deviceId,
    config
  );

  // Calculate the time offset between server and NTP.
  const timeOffset = await calculateTimeOffset();

  // Get public IP address so that the user can use this to connect to the server for video streaming.
  const publicIP = await getPublicIP();

  // Accept the connection by sending an initialization message.
  sendMessage(ws, "init", { session_id: sessionId, device_id: deviceId, server_ip: publicIP });
  logger.info(
    `Sent session_id ${sessionId} and device_id ${deviceId} to client.`
  );

  // Register this connection for the session
  registerConnection(sessionId, deviceId, ws);

  // Start the simulation if applicable.
  const ready = await getReadyForSimulation(sessionId);
  sendMessage(ws, "status", { ready });

  if (!ready) {
    await ws.close();
    return;
  }

  logger.info(`Simulation Ready: ${ready} -- sent to ${deviceId}`);

  // Listen for further messages from the client.
  ws.on("message", async (data: string) => {
    try {
      const json = JSON.parse(data.toString());
      const { data: payload } = json;
      // Add a timestamp (in seconds) adjusted by the time offset.
      payload.received = Date.now() / 1000 + timeOffset;
      // Push data to the simulation.
      await pushData(sessionId, deviceId, payload);
      // Retrieve the simulation's response.
      const response = await getResponse(sessionId, deviceId);
      if (response) {
        sendMessage(ws, "response", response.data);
        if (response.sessionDone === "1") {
          await ws.close();
        }
      }
    } catch (err) {
      logger.error("Error processing message:", err);
      await ws.close();
    }
  });
}

/**
 * Start the WebSocket server.
 */
const port = Number(process.env.WEBSOCKET_SERVER_PORT) || 5000;
const wss = new WebSocketServer({ port });

// Connect to Redis when the server starts.
(async () => {
  try {
    await connect();
    logger.info("Connected to Redis successfully!");
  } catch (error) {
    console.error("Error connecting to Redis:", error);
    process.exit(1); // exit if unable to connect
  }
})();

function setConnectionTimeout(ws: WebSocket, timeoutMs: number) {
  let timer = setTimeout(() => {
    logger.warn("Closing inactive connection due to timeout.");
    ws.terminate();
  }, timeoutMs);

  ws.on("message", () => {
    // Reset the timer when any activity occurs.
    clearTimeout(timer);
    timer = setTimeout(() => {
      logger.warn("Closing inactive connection due to timeout.");
      ws.terminate();
    }, timeoutMs);
  });

  ws.on("close", () => {
    clearTimeout(timer);
  });
}

// Handle incoming WebSocket connections.
wss.on("connection", (ws: WebSocket, req: http.IncomingMessage) => {
  setConnectionTimeout(ws, 300000); // 5 minute inactivity timeout.
  (ws as ExtWebSocket).isAlive = true;

  // Set up a heartbeat mechanism to keep the connection alive.
  ws.on("pong", () => {
    (ws as ExtWebSocket).isAlive = true;
    // console.log("Received pong from client");
  });

  handleConnection(ws, req).catch(async (err) => {
    if (err instanceof Error) {
      logger.error("Error handling connection:", err);
    }
    await sendMessage(ws, "error", err.message);
    await ws.close();
  });
});

// Disconnect from Redis when the server closes.
wss.on("close", async () => {
  try {
    await quit();
    logger.info("Disconnected from Redis successfully!");
    clearInterval(interval);
  } catch (error) {
    console.error("Error disconnecting from Redis:", error);
  }
});

// Set up an interval (e.g., every 30 seconds) to check if clients are alive.
const interval = setInterval(() => {
  wss.clients.forEach((ws: WebSocket) => {
    if (!(ws as ExtWebSocket).isAlive) {
      logger.warn("Terminating a dead connection.");
      // Terminate the connection. This should trigger the 'close' event.
      return ws.terminate();
    }

    // Reset the flag and send a ping. If pong is received, then ws.isAlive will be set to true.
    (ws as ExtWebSocket).isAlive = false;
    ws.ping();
  });
}, 20000); // 20 second interval

process.on("SIGINT", async () => {
  logger.info("\nGracefully shutting down WebSocket server...");

  try {
    await quit();
    logger.info("Disconnected from Redis successfully!");
  } catch (error) {
    console.error("Error during shutdown:", error);
  }

  process.exit(0);
});

logger.info(`WebSocket server is running on ws://localhost:${port}/ws`);
