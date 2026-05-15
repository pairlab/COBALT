import { v4 as uuidv4, validate } from "uuid";
import { WebSocket } from "ws";
import {
  checkHashExists,
  checkHashKeyExists,
  deleteKey,
  getHashKey,
  incrementHashKey,
  setHashKey,
  addToPendingAdd,
  removeFromPendingAdd,
  removeFromPendingDelete,
  addToPendingDelete,
  isInPendingAdd,
  isInPendingDelete,
} from "./redis_client";

const logger = console;

// Global map to track WebSocket connections by session ID
// Map<sessionId, Map<deviceId, WebSocket>>
const sessionConnections = new Map<string, Map<string, WebSocket>>();

/**
 * Get or create the connections map for a session
 */
export function getSessionConnections(sessionId: string): Map<string, WebSocket> {
  if (!sessionConnections.has(sessionId)) {
    sessionConnections.set(sessionId, new Map());
  }
  return sessionConnections.get(sessionId)!;
}

/**
 * Register a WebSocket connection for a device in a session
 */
export function registerConnection(sessionId: string, deviceId: string, ws: WebSocket): void {
  const connections = getSessionConnections(sessionId);
  connections.set(deviceId, ws);
  logger.info(`Registered connection for device ${deviceId} in session ${sessionId}`);
}

/**
 * Unregister a WebSocket connection for a device in a session
 */
export function unregisterConnection(sessionId: string, deviceId: string): void {
  const connections = sessionConnections.get(sessionId);
  if (connections) {
    connections.delete(deviceId);
    logger.info(`Unregistered connection for device ${deviceId} in session ${sessionId}`);
    // Clean up empty session map
    if (connections.size === 0) {
      sessionConnections.delete(sessionId);
    }
  }
}

export interface DeviceConfig {
  sim: string;
  arm: string;
  sim_type: string;
  username: string;
}

export interface SessionConfig {
  session_id: string;
  sim_type: string;
  devices: { [deviceId: string]: DeviceConfig }; // Store devices in a separate object
  username: string;
}

export interface ResponseData {
  data: string;
  sessionDone: string;
}

/**
 * Generate a random alpha-numeric string of input length
 * @returns 
 */
export function generateRandomString(length: number): string {
  const chars = '0123456789';
  let result = '';
  for (let i = 0; i < length; i++) {
    const idx = Math.floor(Math.random() * chars.length);
    result += chars[idx];
  }
  return result;
}

/**
 * Retrieve the number of required devices for a session based on the config
 * @returns integer 
 */
export function getRequiredDeviceCount(sessionConfig: SessionConfig): number {
  switch (sessionConfig["sim_type"]) {
    case "single":
      return 1;
    case "bimanual":
      return 2;
    default:
      throw new Error(`Invalid sim type: ${sessionConfig["sim_type"]}.`);
  }
}

/**
 * Generate a unique session ID ensuring no duplicates with active sessions.
 */
export async function generateSessionId(): Promise<string> {
  let sessionId = generateRandomString(5);
  while (await checkHashExists(`sessions:${sessionId}`)) {
    sessionId = generateRandomString(5);
  }
  return sessionId;
}

export function validateSessionId(sessionId: string): boolean {
  // check if session Id is 5 digit string
  return /^\d{5}$/.test(sessionId);
}

/**
 * Generate a unique device ID ensuring no duplicates with active sessions.
 */
export async function generateDeviceId(sessionId: string): Promise<string> {
  let deviceId = uuidv4();
  while (await checkHashKeyExists(`sessions:${sessionId}`, deviceId)) {
    deviceId = uuidv4();
  }
  return deviceId;
}

/**
 * Retrieve the configuration for a given session.
 */
export async function getSessionConfig(
  sessionId: string
): Promise<SessionConfig> {
  const sessionConfigStr = await getHashKey(`sessions:${sessionId}`, "config");

  if (!sessionConfigStr) {
    throw new Error(`Session ${sessionId} does not exist.`);
  }

  const sessionConfig: SessionConfig = JSON.parse(sessionConfigStr);

  return sessionConfig;
}

/**
 * Create a new session or retrieve an existing session based on the provided session ID.
 * If a session ID is provided, the function will check if the device ID is already in use.
 * If a device ID is provided, the function will check if the device name is already in use.
 * If the session ID is not provided, a new session ID will be generated.
 * If the device ID is not provided, a new device ID will be generated.
 * @param sessionId unique session ID
 * @param deviceId unique device ID (does not have to be unique across sessions)
 * @param config device configuration
 * @returns a tuple containing the session ID and device ID
 */
export async function getOrCreateSessionDetails(sessionId: string, deviceId: string, config: DeviceConfig): Promise<[string, string]> {
  let sessionConfig: SessionConfig;
  let requiredNumDevices: number;

  // If device ID is provided and it is a number, ensure that it is positive.
  if (deviceId && validate(deviceId) === false) {
    throw new Error(`Invalid device ID: ${deviceId}`);
  }

  // Check session ID is a valid session ID
  if (sessionId && validateSessionId(sessionId) === false) {
    throw new Error(`Invalid session ID: ${sessionId}`);
  }

  if (sessionId) {
    sessionConfig = await getSessionConfig(sessionId);

    // Ensure that the session has the appropriate amount of devices
    const numActiveDevices = await getHashKey(
      `sessions:${sessionId}`,
      "device_count"
    );
    requiredNumDevices = getRequiredDeviceCount(sessionConfig);

    if (Number(numActiveDevices) >= requiredNumDevices) {
      throw new Error(
        `Session ${sessionId} has ${numActiveDevices} devices connected but requires ${requiredNumDevices}.`
      );
    }

    // Ensure that the device ID is unique in the session
    if (deviceId) {
      if (sessionConfig.devices[deviceId]) {
        throw new Error(
          `Device ID ${deviceId} already exists in session ${sessionId}.`
        );
      }
    }

    // Ensure that the device configuration is consistent with other devices in the session
    for (const existingDeviceId in sessionConfig.devices) {

      // For now, we can ignore the below two params. This will be necessary in future work with more automation and task routing features.
      // Currently, the sim is determined by whatever task/sim is running on the backend teleoperation server.
      // The arm that will be controlled will be determined by whichever device joins first.

      // if (sessionConfig.devices[existingDeviceId].sim !== config.sim) {
      //   throw new Error(
      //     `Device sim: ${config.sim} is different from other devices in sessions:${sessionId}.`
      //   );
      // }
      // if (sessionConfig.devices[existingDeviceId].arm === config.arm) {
      //   throw new Error(
      //     `Device arm ${config["arm"]} is already in use in sessions:${sessionId}.`
      //   );
      // }

      if (
        sessionConfig.devices[existingDeviceId].sim_type !== config.sim_type
      ) {
        throw new Error(
          `Device sim type: ${config["sim_type"]} is different from other devices in sessions:${sessionId}.`
        );
      }
      if (sessionConfig.devices[existingDeviceId].username !== config.username) {
        throw new Error(
          `Device username: ${config["username"]} is different from other devices in sessions:${sessionId}.`
        );
      }
    }
  } else {
    sessionId = await generateSessionId();
    sessionConfig = {
      session_id: sessionId,
      sim_type: config["sim_type"],
      devices: {},
      username: config["username"],
    };
  }

  if (!deviceId) {
    deviceId = await generateDeviceId(sessionId);
  }

  await setHashKey(
    `sessions:${sessionId}`,
    "config",
    JSON.stringify({
      ...sessionConfig,
      devices: {
        ...sessionConfig.devices,
        [deviceId]: config,
      },
    })
  );

  await incrementHashKey(`sessions:${sessionId}`, "device_count");

  // Refresh session config
  sessionConfig = await getSessionConfig(sessionId);
  requiredNumDevices = getRequiredDeviceCount(sessionConfig);

  // Add session to pending_add if it has the required number of devices
  // This logic assumes that when one device disconnects in a multi-device session, then the entire session is disconnected.
  // Hence, a new session must be recreated. Technically, if a device disconnects and tries to reconnect to the same session (which shouldn't be possible),
  // then a new session should still not be created because the same session ID will still be active in the teleoperation server.
  if (Object.keys(sessionConfig.devices).length === requiredNumDevices) {
    await addToPendingAdd(sessionId);

    // Check if pending_delete_set contains sessionId
    const existsInPendingDelete = await isInPendingDelete(sessionId);
    if (existsInPendingDelete) {
      // Remove from both pending_delete and pending_delete_set
      await removeFromPendingDelete(sessionId);
    }
  }

  return [sessionId, deviceId];
}

/**
 * Gets the ready status of a session.
 * @param sessionId unique session ID
 * @returns true if the session is ready, false otherwise
 */
async function getSessionStatus(sessionId: string): Promise<string> {
  return await getHashKey(`sessions:${sessionId}`, "status");
}

/**
 * Wait for the session to be ready for simulation.
 * Uses a polling mechanism to check the session status.
 * Polls every 1 second
 * Timeout after 60 seconds
 * @param sessionId unique session ID
 * @returns true if the session is ready, false if the timeout is exceeded
 */
export async function getReadyForSimulation(sessionId: string): Promise<boolean> {
  const checkInterval = 1000; // check every 1 second
  const timeout = 60000; // 1 minute timeout
  const startTime = Date.now();

  while (true) {
    const status = await getSessionStatus(sessionId);
    if (status === "1") {
      return true;
    }
    // Teleop server can explicitly reject invalid sessions (e.g. wrong device count).
    if (status === "-1") {
      return false;
    }
    // Check if we've exceeded the timeout.
    if (Date.now() - startTime >= timeout) {
      return false; // or throw an error if you prefer
    }
    await new Promise((resolve) => setTimeout(resolve, checkInterval));
  }
}

/**
 * Push incoming received device data to a session.
 * @param sessionId unique session ID
 * @param deviceId unique device ID
 * @param payload device data
 * @returns true if the data was successfully pushed, false otherwise
 */
export async function pushData(sessionId: string, deviceId: string, payload: any): Promise<boolean> {
  return await setHashKey(
    `sessions:${sessionId}`,
    deviceId,
    JSON.stringify(payload)
  );
}

/**
 * Retrieve the response data for a given session.
 * @param sessionId unique session ID
 * @returns the response data
 */
export async function getResponse(sessionId: string, deviceId: string): Promise<ResponseData | null> {
  const data = await getHashKey(`sessions:${sessionId}`, `response:${deviceId}`);
  const sessionDone = await getHashKey(`sessions:${sessionId}`, "done");
  if (data === "") {
    return null;
  }
  return { data, sessionDone } as ResponseData;
}

/**
 * Decrements device count. If device count is 0, deletes the session. Otherwise, deletes the device and corresponding config.
 * Also closes all WebSocket connections for the session.
 * @param sessionId unique session ID
 */
export async function cleanUpSession(sessionId: string): Promise<void> {
  // Close all WebSocket connections for this session
  const connections = sessionConnections.get(sessionId);
  if (connections) {
    connections.forEach((ws: WebSocket, devId: string) => {
      try {
        logger.info(`Closing WebSocket for device ${devId} in session ${sessionId}`);
        ws.close(1000, "Session ended");
      } catch (err) {
        logger.error(`Error closing WebSocket for device ${devId}:`, err);
      }
    });
    // Clear all connections for this session
    connections.clear();
    sessionConnections.delete(sessionId);
  }

  // Delete the session from Redis
  let success = await deleteKey(`sessions:${sessionId}`);
  console.log('deletion', success);

  await addToPendingDelete(sessionId);

  const existsInPendingAdd = await isInPendingAdd(sessionId);
  if (existsInPendingAdd) {
    await removeFromPendingAdd(sessionId);
  }

  logger.info(`Deleted session ${sessionId} and added to pending_delete`);
}
