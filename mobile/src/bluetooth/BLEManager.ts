/**
 * BLEManager.ts
 * Manages Bluetooth Low Energy communication with the Thalmic MYO Armband.
 *
 * MYO DIRECT BLE PROTOCOL
 * -----------------------
 * The MYO armband uses a proprietary BLE GATT profile documented via
 * community reverse-engineering (https://github.com/thalmiclabs/myo-bluetooth).
 *
 * To enable raw EMG streaming from a mobile device:
 *   1. Connect to the MYO device (advertises with name "Myo").
 *   2. Write the "set EMG mode" command to the command characteristic.
 *   3. Subscribe to all four EMG notify characteristics.
 *   4. Reassemble 8 channels from the 4 interleaved characteristics.
 *
 * Each EMG characteristic delivers 2 samples × 4 channels (int8) per
 * notification at ~50 Hz. Together, the 4 characteristics reconstruct
 * 8 channels at 200 Hz.
 *
 * NOTE: The primary production path uses the Python myo-python SDK on a
 * laptop (via MyoConnect + USB dongle), forwarding data to this app via
 * WebSocket. Direct BLE is an optional alternative for fully untethered use.
 */

import {
  BleManager,
  Device,
  State,
  Subscription,
  BleError,
  Characteristic,
  LogLevel,
} from 'react-native-ble-plx';
import { Platform, PermissionsAndroid } from 'react-native';
import { Buffer } from 'buffer';

// ---------------------------------------------------------------------------
// MYO BLE Constants
// (Source: https://github.com/thalmiclabs/myo-bluetooth)
// ---------------------------------------------------------------------------

/** MYO control service — write commands here to configure the armband. */
export const MYO_CONTROL_SERVICE_UUID    = 'd5060001-a904-deb9-4748-2c7f4a124842';

/** MYO command characteristic — write SET_EMG_MODE command to start streaming. */
export const MYO_COMMAND_CHAR_UUID       = 'd5060401-a904-deb9-4748-2c7f4a124842';

/** MYO EMG data service. */
export const MYO_EMG_SERVICE_UUID        = 'd5060005-a904-deb9-4748-2c7f4a124842';

/**
 * Four EMG notify characteristics. Each delivers 2 samples × 4 channels (int8)
 * at ~50 Hz. Chars 0 & 2 carry channels 0-3; chars 1 & 3 carry channels 4-7.
 */
export const MYO_EMG_CHAR_UUIDS = [
  'd5060105-a904-deb9-4748-2c7f4a124842',
  'd5060205-a904-deb9-4748-2c7f4a124842',
  'd5060305-a904-deb9-4748-2c7f4a124842',
  'd5060405-a904-deb9-4748-2c7f4a124842',
] as const;

/** BLE advertised device name. */
export const BLE_DEVICE_NAME = 'Myo';

/**
 * SET_EMG_MODE command bytes.
 * Format: [command=0x01, payload_size=0x03, emg_mode=0x02, imu_mode=0x01, classifier_mode=0x01]
 * emg_mode=0x02 → raw filtered EMG (200 Hz, 8 ch, int8)
 */
const SET_EMG_MODE_CMD = new Uint8Array([0x01, 0x03, 0x02, 0x01, 0x01]);

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type ConnectionStatus =
  | 'disconnected'
  | 'scanning'
  | 'connecting'
  | 'connected'
  | 'error';

export type ConnectionStatusListener = (status: ConnectionStatus) => void;

/** Decoded 8-channel EMG sample (int8 values, one per channel). */
export type EMGSample = [number, number, number, number, number, number, number, number];

// ---------------------------------------------------------------------------
// EMGBLEManager
// ---------------------------------------------------------------------------

export class EMGBLEManager {
  private manager: BleManager;
  private connectedDevice: Device | null = null;
  private emgSubscriptions: Subscription[] = [];
  private scanSubscription: Subscription | null = null;
  private status: ConnectionStatus = 'disconnected';
  private statusListeners: Set<ConnectionStatusListener> = new Set();

  private reconnectAttempts = 0;
  private readonly MAX_RECONNECT_ATTEMPTS = 5;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private targetDeviceName: string | null = null;
  private onDataCallback: ((bytes: Uint8Array) => void) | null = null;

  constructor() {
    this.manager = new BleManager();
    if (__DEV__) {
      this.manager.setLogLevel(LogLevel.Verbose);
    }
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /** Request OS Bluetooth permissions then scan for and connect to the MYO. */
  async scanAndConnect(deviceName: string = BLE_DEVICE_NAME): Promise<Device> {
    this.targetDeviceName = deviceName;
    this.reconnectAttempts = 0;
    await this.ensurePermissions();
    await this.waitForBleReady();
    return this.doScanAndConnect(deviceName);
  }

  /**
   * Subscribe to raw EMG bytes from all four MYO EMG characteristics.
   * Each call to `onData` delivers 8 int8 values (one 8-channel sample).
   */
  async subscribeToEMG(
    device: Device,
    onData: (bytes: Uint8Array) => void,
  ): Promise<void> {
    this.onDataCallback = onData;

    // Enable raw EMG streaming via command write
    await this.enableEMGStreaming(device);

    // Subscribe to all 4 EMG characteristics
    for (const charUUID of MYO_EMG_CHAR_UUIDS) {
      const sub = device.monitorCharacteristicForService(
        MYO_EMG_SERVICE_UUID,
        charUUID,
        (error: BleError | null, characteristic: Characteristic | null) => {
          if (error) {
            console.warn('[MYO BLE] EMG char monitor error:', error.message);
            if (error.errorCode === 201 || error.errorCode === 205) {
              this.handleUnexpectedDisconnect();
            }
            return;
          }
          if (!characteristic?.value) return;

          const raw = Buffer.from(characteristic.value, 'base64');
          const bytes = new Uint8Array(raw.buffer, raw.byteOffset, raw.byteLength);

          // Each notification: [s0_ch0, s0_ch1, s0_ch2, s0_ch3, s1_ch0, s1_ch1, s1_ch2, s1_ch3]
          // We emit each 4-byte half-sample. The EMGStream processor reassembles 8-ch frames.
          onData(bytes);
        },
      );
      this.emgSubscriptions.push(sub);
    }
  }

  /** Disconnect from the MYO and cancel any pending reconnect. */
  async disconnect(): Promise<void> {
    this.cancelReconnect();
    this.targetDeviceName = null;
    this.onDataCallback = null;

    this.emgSubscriptions.forEach((s) => s.remove());
    this.emgSubscriptions = [];

    this.scanSubscription?.remove();
    this.scanSubscription = null;

    if (this.connectedDevice) {
      try {
        await this.connectedDevice.cancelConnection();
      } catch {
        // Intentional disconnect — ignore errors
      }
      this.connectedDevice = null;
    }

    this.setStatus('disconnected');
  }

  isConnected(): boolean {
    return this.status === 'connected' && this.connectedDevice !== null;
  }

  addStatusListener(listener: ConnectionStatusListener): () => void {
    this.statusListeners.add(listener);
    return () => this.statusListeners.delete(listener);
  }

  getStatus(): ConnectionStatus {
    return this.status;
  }

  destroy(): void {
    this.disconnect();
    this.manager.destroy();
  }

  // -------------------------------------------------------------------------
  // Private helpers
  // -------------------------------------------------------------------------

  /**
   * Write the SET_EMG_MODE command to the MYO command characteristic.
   * This activates raw filtered EMG streaming at 200 Hz.
   */
  private async enableEMGStreaming(device: Device): Promise<void> {
    try {
      const cmdBase64 = Buffer.from(SET_EMG_MODE_CMD).toString('base64');
      await device.writeCharacteristicWithResponseForService(
        MYO_CONTROL_SERVICE_UUID,
        MYO_COMMAND_CHAR_UUID,
        cmdBase64,
      );
      console.log('[MYO BLE] EMG streaming enabled (200 Hz, 8 ch)');
    } catch (err) {
      console.warn('[MYO BLE] Failed to write EMG mode command:', err);
    }
  }

  private async doScanAndConnect(deviceName: string): Promise<Device> {
    return new Promise<Device>((resolve, reject) => {
      this.setStatus('scanning');

      const timeout = setTimeout(() => {
        this.manager.stopDeviceScan();
        reject(new Error(`Scan timed out — MYO "${deviceName}" not found. Ensure armband is charged and within range.`));
        this.setStatus('error');
      }, 20_000);

      this.manager.startDeviceScan(
        null, // Scan all services — MYO may not advertise service UUID before connect
        { allowDuplicates: false },
        async (error: BleError | null, device: Device | null) => {
          if (error) {
            clearTimeout(timeout);
            this.manager.stopDeviceScan();
            reject(error);
            this.setStatus('error');
            return;
          }

          if (!device) return;

          const nameMatch =
            device.name?.startsWith('Myo') ||
            device.localName?.startsWith('Myo') ||
            device.name === deviceName ||
            device.id === deviceName;

          if (!nameMatch) return;

          clearTimeout(timeout);
          this.manager.stopDeviceScan();
          console.log(`[MYO BLE] Found device: ${device.name} [${device.id}]`);

          try {
            this.setStatus('connecting');
            const connected = await device.connect({ timeout: 10_000 });
            await connected.discoverAllServicesAndCharacteristics();

            this.connectedDevice = connected;
            this.reconnectAttempts = 0;
            this.setStatus('connected');

            this.connectedDevice.onDisconnected((_error, _dev) => {
              if (this.targetDeviceName !== null) {
                this.handleUnexpectedDisconnect();
              }
            });

            resolve(connected);
          } catch (connectError) {
            this.setStatus('error');
            reject(connectError);
          }
        },
      );
    });
  }

  private handleUnexpectedDisconnect(): void {
    this.emgSubscriptions.forEach((s) => s.remove());
    this.emgSubscriptions = [];
    this.connectedDevice = null;
    this.setStatus('disconnected');

    if (this.targetDeviceName && this.reconnectAttempts < this.MAX_RECONNECT_ATTEMPTS) {
      this.scheduleReconnect();
    } else {
      this.setStatus('error');
    }
  }

  private scheduleReconnect(): void {
    this.cancelReconnect();
    const backoffMs = Math.min(1_000 * 2 ** this.reconnectAttempts, 30_000);
    this.reconnectAttempts += 1;
    console.log(`[MYO BLE] Reconnect attempt ${this.reconnectAttempts}/${this.MAX_RECONNECT_ATTEMPTS} in ${backoffMs}ms`);

    this.reconnectTimer = setTimeout(async () => {
      if (!this.targetDeviceName) return;
      try {
        const device = await this.doScanAndConnect(this.targetDeviceName);
        if (this.onDataCallback) {
          await this.subscribeToEMG(device, this.onDataCallback);
        }
      } catch {
        if (this.reconnectAttempts < this.MAX_RECONNECT_ATTEMPTS) {
          this.scheduleReconnect();
        } else {
          this.setStatus('error');
        }
      }
    }, backoffMs);
  }

  private cancelReconnect(): void {
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private setStatus(status: ConnectionStatus): void {
    this.status = status;
    this.statusListeners.forEach((l) => l(status));
  }

  private async waitForBleReady(): Promise<void> {
    return new Promise((resolve, reject) => {
      const subscription = this.manager.onStateChange((state: State) => {
        if (state === State.PoweredOn) {
          subscription.remove();
          resolve();
        } else if (state === State.Unsupported || state === State.Unauthorized) {
          subscription.remove();
          reject(new Error(`Bluetooth state: ${state}`));
        }
      }, true);
    });
  }

  private async ensurePermissions(): Promise<void> {
    if (Platform.OS !== 'android') return;

    if (Platform.Version >= 31) {
      const results = await PermissionsAndroid.requestMultiple([
        PermissionsAndroid.PERMISSIONS.BLUETOOTH_SCAN,
        PermissionsAndroid.PERMISSIONS.BLUETOOTH_CONNECT,
        PermissionsAndroid.PERMISSIONS.BLUETOOTH_ADVERTISE,
        PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
      ]);
      const allGranted = Object.values(results).every(
        (r) => r === PermissionsAndroid.RESULTS.GRANTED,
      );
      if (!allGranted) {
        throw new Error('Required Bluetooth permissions were not granted. Enable them in Settings.');
      }
    } else {
      const granted = await PermissionsAndroid.request(
        PermissionsAndroid.PERMISSIONS.ACCESS_FINE_LOCATION,
        {
          title: 'Location Permission',
          message: 'MAIA EMG-ASL needs location access to scan for Bluetooth devices.',
          buttonNeutral: 'Ask Me Later',
          buttonNegative: 'Cancel',
          buttonPositive: 'OK',
        },
      );
      if (granted !== PermissionsAndroid.RESULTS.GRANTED) {
        throw new Error('Location permission required for BLE scanning.');
      }
    }
  }
}

// Singleton instance — import this across the app
export const bleManager = new EMGBLEManager();
