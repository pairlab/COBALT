import { createClient, RedisClientType } from "redis";
import dotenv from "dotenv";
dotenv.config();

const REDIS_HOST = process.env.REDIS_HOST || "localhost";
const REDIS_PORT = process.env.REDIS_PORT || 6379;

const client: RedisClientType = createClient({
  url: `redis://${REDIS_HOST}:${REDIS_PORT}`,
});
client.on("error", (err) => console.log("Redis Client Error", err));

const isDebugMode = process.env.NODE_ENV === "development";

/**
 * Redis Requirements
 * Set - contains all session ids and device ids
 * Hash - 
 *      session id: {
 *          device_1 id: <pose data>
 *          device_2 id: <pose data>
 *          response_data: <data>
 *          ready: <boolean>
 *          config: <config data>
 *      }
 * Python file
 */

export async function connect(): Promise<boolean> {
  if (await client.connect()) {

    if (isDebugMode) {
      await client.flushDb();
    }

    return true;
  }

  return false;
}

export async function quit(): Promise<void> {

  if (isDebugMode) {
    await client.flushDb();
  }

  await client.quit();
}

export async function getHashKey(hash: string, key: string): Promise<string> {
  const data = await client.hGet(hash, key);
  return data || '';
}

export async function checkHashExists(hash: string): Promise<boolean> {
  if (await client.exists(hash)) {
    return true;
  }
  return false;
}

export async function checkHashKeyExists(hash: string, key: string): Promise<boolean> {
  if (await client.hExists(hash, key)) {
    return true;
  }
  return false;
}

export async function setHashKey(hash: string, key: string, value: string): Promise<boolean> {
  if (await client.hSet(hash, key, value)) {
    return true;
  }
  return false;
}

export async function incrementHashKey(hash: string, key: string): Promise<number> {
  return await client.hIncrBy(hash, key, 1);
}

export async function decrementHashKey(hash: string, key: string): Promise<number> {
  return await client.hIncrBy(hash, key, -1);
}

export async function deleteKey(key: string): Promise<boolean> {
  if (await client.del(key)) {
    return true;
  }
  return false;
}

export async function deleteHashKey(hash: string, key: string): Promise<boolean> {
  if (await client.hDel(hash, key)) {
    return true;
  }
  return false;
}

export async function addToPendingAdd(sessionId: string): Promise<void> {
  await client.lPush('pending_add', sessionId);
  await client.sAdd('pending_add_set', sessionId);
}

export async function addToPendingDelete(sessionId: string): Promise<void> {
  await client.lPush('pending_delete', sessionId);
  await client.sAdd('pending_delete_set', sessionId);
}

export async function removeFromPendingAdd(sessionId: string): Promise<void> {
  await client.sRem('pending_add_set', sessionId);
  await client.lRem('pending_add', 0, sessionId);
}

export async function removeFromPendingDelete(sessionId: string): Promise<void> {
  await client.sRem('pending_delete_set', sessionId);
  await client.lRem('pending_delete', 0, sessionId);
}

export async function isInPendingAdd(sessionId: string): Promise<boolean> {
  return await client.sIsMember('pending_add_set', sessionId);
}

export async function isInPendingDelete(sessionId: string): Promise<boolean> {
  return await client.sIsMember('pending_delete_set', sessionId);
}
