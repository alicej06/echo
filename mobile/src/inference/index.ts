/**
 * inference/index.ts
 * Public surface of the on-device inference module.
 *
 * Typical usage:
 *
 *   import { onDeviceInference, EMGWindowBuffer } from '../inference';
 *
 *   // At app start / screen mount:
 *   await onDeviceInference.loadModel();
 *   const buffer = new EMGWindowBuffer();
 *
 *   // Inside the BLE data callback:
 *   const windows = buffer.ingestBytes(blePacket);
 *   for (const window of windows) {
 *     const result = await onDeviceInference.predict(window);
 *     if (result) console.log(result.label, result.confidence);
 *   }
 *
 *   // On unmount:
 *   onDeviceInference.dispose();
 *   buffer.reset();
 */

export { onDeviceInference } from './ONNXInference';
export { EMGWindowBuffer } from './EMGWindowBuffer';
