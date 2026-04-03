/**
 * MAIA Server configuration.
 * Reads from Expo public env vars at build time.
 * Can be overridden at runtime (e.g. from SettingsScreen).
 */

export interface ServerConfig {
  /** Base HTTP/HTTPS URL, e.g. https://maia.railway.app */
  serverUrl: string;
  /** WebSocket URL derived from serverUrl */
  wsUrl: string;
  /** X-API-Key header value */
  apiKey: string;
  /** If true, skip server entirely and use on-device ONNX only */
  onDeviceOnly: boolean;
  /** If true, fall back to server when on-device confidence is low */
  fallbackToServer: boolean;
}

const _railwayUrl = process.env.EXPO_PUBLIC_RAILWAY_URL ?? '';
const _serverUrl = process.env.EXPO_PUBLIC_SERVER_URL ?? _railwayUrl ?? 'http://localhost:8000';
const _apiKey = process.env.EXPO_PUBLIC_API_KEY ?? 'dev-key-change-me';
const _fallback = (process.env.EXPO_PUBLIC_FALLBACK_TO_SERVER ?? 'true').toLowerCase() !== 'false';

/** Converts https:// → wss:// and http:// → ws:// */
function toWsUrl(httpUrl: string): string {
  return httpUrl.replace(/^https:\/\//, 'wss://').replace(/^http:\/\//, 'ws://');
}

let _config: ServerConfig = {
  serverUrl: _serverUrl,
  wsUrl: `${toWsUrl(_serverUrl)}/ws/emg`,
  apiKey: _apiKey,
  onDeviceOnly: false,
  fallbackToServer: _fallback,
};

/** Get the current server config. */
export function getServerConfig(): Readonly<ServerConfig> {
  return _config;
}

/** Override server config at runtime (e.g. from Settings screen). */
export function updateServerConfig(partial: Partial<ServerConfig>): void {
  _config = { ..._config, ...partial };
  if (partial.serverUrl && !partial.wsUrl) {
    _config.wsUrl = `${toWsUrl(partial.serverUrl)}/ws/emg`;
  }
}
