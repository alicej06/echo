/**
 * metro.config.js
 * Metro bundler configuration for Expo SDK 51.
 * Adds Buffer polyfill resolution and react-native-ble-plx compatibility.
 */

const { getDefaultConfig } = require('expo/metro-config');

/** @type {import('expo/metro-config').MetroConfig} */
const config = getDefaultConfig(__dirname);

// Resolve 'buffer' -> node_modules/buffer (BLE + socket.io need it)
config.resolver.extraNodeModules = {
  ...config.resolver.extraNodeModules,
  buffer: require.resolve('buffer'),
  stream: require.resolve('stream-browserify'),
  events: require.resolve('events'),
};

// CJS interop: ensure .cjs extensions are handled
config.resolver.sourceExts = [
  ...config.resolver.sourceExts,
  'cjs',
];

module.exports = config;
