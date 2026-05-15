import https from "https";

/**
 * Retrieve the public IP address by querying an external service (ipify).
 * This should work in any environment with internet access.
 */
export function getPublicIP(): Promise<string> {
  return new Promise((resolve, reject) => {
    https
      .get("https://api.ipify.org?format=json", (res) => {
        let data = "";
        res.on("data", (chunk) => {
          data += chunk;
        });
        res.on("end", () => {
          try {
            const parsed = JSON.parse(data);
            if (parsed && parsed.ip) {
              resolve(parsed.ip);
            } else {
              reject(new Error("No IP found in response"));
            }
          } catch (err) {
            reject(err);
          }
        });
      })
      .on("error", (err) => {
        reject(err);
      });
  });
}
