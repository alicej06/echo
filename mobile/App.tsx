/**
 * App.tsx
 * Entry point for the EMG ASL mobile app.
 *
 * expo-router takes over routing via the app/ directory.  This file exists
 * primarily as the Expo entry point shim and to ensure global polyfills
 * (Buffer, etc.) are loaded before any module that needs them.
 *
 * NOTE: When using expo-router the "main" field in package.json points to
 * "expo-router/entry", so this file is only executed when running without
 * expo-router (e.g. bare workflow / unit tests).  It is kept here for
 * completeness and for teams that prefer an explicit entry point.
 */

// ---------------------------------------------------------------------------
// Polyfills — must come before any other import that uses Buffer
// ---------------------------------------------------------------------------
import { Buffer } from 'buffer';
global.Buffer = global.Buffer ?? Buffer;

// ---------------------------------------------------------------------------
// React / Expo Router entry
// ---------------------------------------------------------------------------
import 'expo-router/entry';
