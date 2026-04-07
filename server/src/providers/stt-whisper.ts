/**
 * Whisper STT Provider
 *
 * Local/self-hosted STT via whisper.cpp server (HTTP batch API).
 * Free, no API key needed. Runs via Docker or existing instance.
 *
 * Uses client-side adaptive energy-based VAD to detect speech boundaries,
 * then sends buffered audio to whisper.cpp's /inference endpoint.
 *
 * Setup: docker run -d --name callme-whisper -p 127.0.0.1:8178:8080 \
 *   -v callme-whisper-models:/models ghcr.io/ggml-org/whisper.cpp:main \
 *   whisper-server --host 0.0.0.0 --port 8080 -m /models/ggml-large-v3-turbo.bin
 */

import { spawnSync, spawn, type ChildProcess } from 'child_process';
import { existsSync, mkdirSync } from 'fs';
import { join } from 'path';
import { homedir, platform } from 'os';
import type { RealtimeSTTProvider, RealtimeSTTSession, STTConfig } from './types.js';

// Docker constants
const DOCKER_IMAGE = 'ghcr.io/ggml-org/whisper.cpp:main';
const CONTAINER_NAME = 'callme-whisper';
const DEFAULT_PORT = 8178;
const HEALTH_TIMEOUT_MS = 120000;
const HEALTH_POLL_MS = 3000;
const VOLUME_NAME = 'callme-whisper-models';

// VAD constants
const SPEECH_CONFIRM_FRAMES = 3;   // Consecutive frames above threshold to confirm speech (60ms)
const PRE_BUFFER_FRAMES = 15;      // Rolling buffer before speech detection (~300ms)
const MIN_SPEECH_MS = 200;         // Minimum speech duration to transcribe
const NOISE_FLOOR_ALPHA = 0.01;    // EMA smoothing for noise floor
const SPEECH_MULTIPLIER = 3.0;     // Speech threshold = noiseFloor * this
const SILENCE_MULTIPLIER = 1.5;    // Silence threshold = noiseFloor * this
const MIN_SPEECH_THRESHOLD = 200;  // Minimum absolute speech threshold
const MIN_SILENCE_THRESHOLD = 100; // Minimum absolute silence threshold

// ─── Mu-law decode lookup table ────────────────────────────────────────────────
// Pre-compute all 256 mu-law byte → 16-bit PCM sample mappings.
const MU_LAW_TABLE = new Int16Array(256);
for (let i = 0; i < 256; i++) {
  // Mu-law is stored with all bits inverted
  const mu = ~i & 0xff;
  const sign = mu & 0x80;
  const exponent = (mu >> 4) & 0x07;
  const mantissa = mu & 0x0f;
  let sample = ((mantissa << 3) + 0x84) << exponent;
  sample -= 0x84;
  if (sign) sample = -sample;
  MU_LAW_TABLE[i] = sample;
}

// ─── Audio conversion utilities ────────────────────────────────────────────────

/**
 * Decode mu-law buffer to PCM 16-bit little-endian.
 * Each mu-law byte becomes 2 bytes (one 16-bit sample).
 */
function muLawBufferToPcm16(muLaw: Buffer): Buffer {
  const pcm = Buffer.alloc(muLaw.length * 2);
  for (let i = 0; i < muLaw.length; i++) {
    pcm.writeInt16LE(MU_LAW_TABLE[muLaw[i]], i * 2);
  }
  return pcm;
}

/**
 * Upsample PCM 16-bit from 8kHz to 16kHz using linear interpolation.
 * For each pair of adjacent samples, produces the original + one interpolated sample.
 */
function resample8kTo16k(pcm8k: Buffer): Buffer {
  const sampleCount = pcm8k.length / 2;
  if (sampleCount < 2) return pcm8k;

  const pcm16k = Buffer.alloc(sampleCount * 2 * 2); // 2x samples, 2 bytes each
  for (let i = 0; i < sampleCount; i++) {
    const sample = pcm8k.readInt16LE(i * 2);
    const nextSample = i < sampleCount - 1 ? pcm8k.readInt16LE((i + 1) * 2) : sample;
    const interpolated = Math.round((sample + nextSample) / 2);

    pcm16k.writeInt16LE(sample, i * 4);
    pcm16k.writeInt16LE(interpolated, i * 4 + 2);
  }
  return pcm16k;
}

/**
 * Wrap raw PCM data in a WAV header (mono, 16-bit).
 */
function encodeWav(pcm: Buffer, sampleRate: number): Buffer {
  const header = Buffer.alloc(44);
  const dataSize = pcm.length;
  const fileSize = 36 + dataSize;

  // RIFF header
  header.write('RIFF', 0);
  header.writeUInt32LE(fileSize, 4);
  header.write('WAVE', 8);

  // fmt chunk
  header.write('fmt ', 12);
  header.writeUInt32LE(16, 16);        // chunk size
  header.writeUInt16LE(1, 20);         // PCM format
  header.writeUInt16LE(1, 22);         // mono
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(sampleRate * 2, 28); // byte rate (sampleRate * channels * bitsPerSample/8)
  header.writeUInt16LE(2, 32);         // block align
  header.writeUInt16LE(16, 34);        // bits per sample

  // data chunk
  header.write('data', 36);
  header.writeUInt32LE(dataSize, 40);

  return Buffer.concat([header, pcm]);
}

/**
 * Compute RMS energy of mu-law audio chunk using lookup table.
 */
function computeRmsEnergy(muLaw: Buffer): number {
  if (muLaw.length === 0) return 0;
  let sumSquares = 0;
  for (let i = 0; i < muLaw.length; i++) {
    const sample = MU_LAW_TABLE[muLaw[i]];
    sumSquares += sample * sample;
  }
  return Math.sqrt(sumSquares / muLaw.length);
}

// ─── Auto-setup ────────────────────────────────────────────────────────────────

// Track spawned server process for cleanup
let whisperServerProcess: ChildProcess | null = null;

/**
 * Map model name (e.g. 'large-v3-turbo') to the ggml filename used by whisper.cpp.
 */
function modelToFilename(model: string): string {
  if (model.startsWith('ggml-') && model.endsWith('.bin')) return model;
  return `ggml-${model}.bin`;
}

/**
 * Get the models directory for storing whisper models.
 */
function getModelsDir(): string {
  const dir = join(homedir(), '.cache', 'callme', 'whisper-models');
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
  return dir;
}

/**
 * Download a whisper model from Hugging Face if not already present.
 */
async function ensureModelDownloaded(model: string): Promise<string> {
  const modelsDir = getModelsDir();
  const modelFile = modelToFilename(model);
  const modelPath = join(modelsDir, modelFile);

  if (existsSync(modelPath)) {
    console.error(`[Whisper] Model ${modelFile} found at ${modelPath}`);
    return modelPath;
  }

  // Download from Hugging Face
  const baseModel = model.replace(/^ggml-/, '').replace(/\.bin$/, '');
  const url = `https://huggingface.co/ggerganov/whisper.cpp/resolve/main/${modelFile}`;
  console.error(`[Whisper] Downloading model ${modelFile} from Hugging Face...`);
  console.error(`[Whisper] URL: ${url}`);
  console.error(`[Whisper] This may take a while for large models...`);

  // Use curl for download with progress
  const curlResult = spawnSync('curl', [
    '-L', '--progress-bar',
    '-o', modelPath,
    url,
  ], { stdio: ['pipe', 'pipe', 'inherit'], timeout: 600000 }); // 10 min

  if (curlResult.status !== 0) {
    throw new Error(`Failed to download whisper model from ${url}`);
  }

  console.error(`[Whisper] Model downloaded to ${modelPath}`);
  return modelPath;
}

/**
 * Start whisper-server as a native process (macOS/Linux with brew or local binary).
 */
async function startNativeServer(modelPath: string): Promise<string> {
  // Find whisper-server binary
  const whichResult = spawnSync('which', ['whisper-server'], { stdio: 'pipe', timeout: 5000 });
  const serverBin = whichResult.stdout?.toString().trim();

  if (!serverBin) {
    throw new Error(
      'whisper-server not found. Install via: brew install whisper-cpp\n' +
      'Or set CALLME_WHISPER_URL to point to an existing whisper.cpp server'
    );
  }

  // Check if port is already in use (avoid silently connecting to wrong server)
  const portCheck = spawnSync('lsof', ['-i', `:${DEFAULT_PORT}`, '-t'], { stdio: 'pipe', timeout: 5000 });
  if (portCheck.stdout?.toString().trim()) {
    throw new Error(
      `Port ${DEFAULT_PORT} already in use (PID ${portCheck.stdout.toString().trim()}).\n` +
      `Set CALLME_WHISPER_URL=http://127.0.0.1:${DEFAULT_PORT} if that's your whisper server,\n` +
      'or stop the process and retry.'
    );
  }

  console.error(`[Whisper] Starting native server (${serverBin})...`);
  console.error(`[Whisper] Model: ${modelPath}`);

  const serverProc = spawn(serverBin, [
    '--host', '127.0.0.1',
    '--port', String(DEFAULT_PORT),
    '-m', modelPath,
    '--convert',  // Accept any audio format via ffmpeg
  ], {
    stdio: ['pipe', 'pipe', 'pipe'],
    detached: false,
  });

  whisperServerProcess = serverProc;

  // Log server stderr for debugging
  serverProc.stderr?.on('data', (data: Buffer) => {
    const line = data.toString().trim();
    if (line) console.error(`[whisper-server] ${line}`);
  });

  serverProc.on('exit', (code) => {
    if (whisperServerProcess === serverProc) {
      console.error(`[Whisper] Server exited with code ${code}`);
      whisperServerProcess = null;
    }
  });

  // Register cleanup
  const cleanup = () => {
    if (whisperServerProcess) {
      console.error('[Whisper] Stopping server...');
      whisperServerProcess.kill('SIGTERM');
      whisperServerProcess = null;
    }
  };
  process.on('exit', cleanup);
  process.on('SIGINT', cleanup);
  process.on('SIGTERM', cleanup);

  return `http://127.0.0.1:${DEFAULT_PORT}`;
}

/**
 * Start whisper-server via Docker (Linux amd64).
 */
async function startDockerServer(modelPath: string): Promise<string> {
  const modelFile = modelPath.split('/').pop()!;

  // Check if Docker is available
  const dockerCheck = spawnSync('docker', ['info'], { stdio: 'pipe', timeout: 10000 });
  if (dockerCheck.status !== 0) {
    throw new Error(
      'Neither whisper-server binary nor Docker found.\n' +
      'Install whisper-server: brew install whisper-cpp (macOS) or build from source\n' +
      'Or set CALLME_WHISPER_URL to point to an existing whisper.cpp server'
    );
  }

  // Ensure volume
  const volResult = spawnSync('docker', ['volume', 'create', VOLUME_NAME], { stdio: 'pipe', timeout: 10000 });
  if (volResult.status !== 0) {
    const err = volResult.stderr?.toString() || 'unknown error';
    throw new Error(`Failed to create Docker volume '${VOLUME_NAME}': ${err.trim()}`);
  }

  // Copy model to volume
  const modelsDir = getModelsDir();
  console.error(`[Whisper] Copying model to Docker volume...`);
  const cpResult = spawnSync('docker', [
    'run', '--rm',
    '-v', `${VOLUME_NAME}:/models`,
    '-v', `${modelsDir}:/host-models:ro`,
    'alpine',
    'cp', `/host-models/${modelFile}`, `/models/${modelFile}`,
  ], { stdio: 'pipe', timeout: 60000 });
  if (cpResult.status !== 0) {
    const err = cpResult.stderr?.toString() || 'unknown error';
    throw new Error(`Failed to copy model to Docker volume: ${err.trim()}`);
  }

  // Check if container exists
  const psAll = spawnSync('docker', ['ps', '-a', '--filter', `name=^${CONTAINER_NAME}$`, '--format', '{{.Status}}\t{{.Command}}'], {
    stdio: 'pipe', timeout: 10000,
  });
  const containerInfo = psAll.stdout?.toString().trim();

  if (containerInfo) {
    // Check if existing container uses the correct model
    const usesCorrectModel = containerInfo.includes(modelFile);
    if (!usesCorrectModel) {
      console.error(`[Whisper] Container exists but uses different model — recreating...`);
      spawnSync('docker', ['rm', '-f', CONTAINER_NAME], { stdio: 'pipe', timeout: 15000 });
    } else if (containerInfo.startsWith('Up')) {
      console.error(`[Whisper] Container '${CONTAINER_NAME}' already running`);
      return `http://127.0.0.1:${DEFAULT_PORT}`;
    } else {
      console.error(`[Whisper] Starting stopped container '${CONTAINER_NAME}'...`);
      const startResult = spawnSync('docker', ['start', CONTAINER_NAME], { stdio: 'pipe', timeout: 30000 });
      if (startResult.status !== 0) {
        const err = startResult.stderr?.toString() || 'unknown error';
        throw new Error(`Failed to start container: ${err.trim()}`);
      }
      return `http://127.0.0.1:${DEFAULT_PORT}`;
    }
  }

  {
    console.error(`[Whisper] Starting new container '${CONTAINER_NAME}'...`);
    const run = spawnSync('docker', [
      'run', '-d',
      '--name', CONTAINER_NAME,
      '-p', `127.0.0.1:${DEFAULT_PORT}:8080`,
      '-v', `${VOLUME_NAME}:/models`,
      DOCKER_IMAGE,
      'whisper-server',
      '--host', '0.0.0.0',
      '--port', '8080',
      '-m', `/models/${modelFile}`,
    ], { stdio: 'pipe', timeout: 30000 });

    if (run.status !== 0) {
      const err = run.stderr?.toString() || 'unknown error';
      if (err.includes('port is already allocated') || err.includes('address already in use')) {
        console.error(`[Whisper] Port ${DEFAULT_PORT} in use. Set CALLME_WHISPER_URL to use a different port`);
      }
      throw new Error(`Failed to start Whisper container: ${err.trim()}`);
    }
  }

  return `http://127.0.0.1:${DEFAULT_PORT}`;
}

/**
 * Wait for whisper server to become healthy.
 */
async function waitForHealth(baseUrl: string): Promise<void> {
  console.error('[Whisper] Waiting for server to be ready...');
  const deadline = Date.now() + HEALTH_TIMEOUT_MS;
  while (Date.now() < deadline) {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 5000);
      const res = await fetch(`${baseUrl}/`, { signal: controller.signal });
      clearTimeout(timeout);
      if (res.ok) {
        console.error('[Whisper] Server ready!');
        return;
      }
    } catch {
      // Not ready yet
    }
    await new Promise(r => setTimeout(r, HEALTH_POLL_MS));
  }
  throw new Error(`Whisper server did not become ready within ${HEALTH_TIMEOUT_MS / 1000}s`);
}

/**
 * Ensure a whisper.cpp server is running.
 * - macOS: uses native whisper-server binary (brew install whisper-cpp)
 * - Linux: falls back to Docker
 * Call this BEFORE creating the provider when no CALLME_WHISPER_URL is set.
 */
export async function ensureWhisperServerRunning(model: string = 'large-v3-turbo'): Promise<string> {
  // Download model if needed
  const modelPath = await ensureModelDownloaded(model);

  // Check for native binary first (preferred — uses Metal/GPU acceleration on macOS)
  const whichResult = spawnSync('which', ['whisper-server'], { stdio: 'pipe', timeout: 5000 });
  const hasNativeBinary = whichResult.status === 0;

  let baseUrl: string;
  if (hasNativeBinary) {
    baseUrl = await startNativeServer(modelPath);
  } else {
    baseUrl = await startDockerServer(modelPath);
  }

  await waitForHealth(baseUrl);
  return baseUrl;
}

// ─── Provider ──────────────────────────────────────────────────────────────────

export class WhisperSTTProvider implements RealtimeSTTProvider {
  readonly name = 'whisper';
  private serverUrl: string = '';
  private silenceDurationMs: number = 800;

  initialize(config: STTConfig): void {
    this.serverUrl = (config.apiUrl || `http://localhost:${DEFAULT_PORT}`).replace(/\/+$/, '');
    this.silenceDurationMs = config.silenceDurationMs || 800;
    const model = config.model || 'large-v3-turbo';
    console.error(`STT provider: Whisper (model: ${model}, silence: ${this.silenceDurationMs}ms, url: ${this.serverUrl})`);
  }

  createSession(): RealtimeSTTSession {
    return new WhisperSTTSession(this.serverUrl, this.silenceDurationMs);
  }
}

// ─── Session ───────────────────────────────────────────────────────────────────

type VADState = 'idle' | 'speech';

class WhisperSTTSession implements RealtimeSTTSession {
  private serverUrl: string;
  private silenceDurationMs: number;
  private connected = false;

  // VAD state
  private vadState: VADState = 'idle';
  private speechFrameCount = 0;
  private silenceStartMs: number | null = null;
  private noiseFloor = 500; // Initial estimate, adapts quickly
  private speechStartMs = 0;

  // Audio buffers
  private audioBuffer: Buffer[] = [];
  private preBuffer: Buffer[] = []; // Rolling buffer for pre-speech audio

  // Transcript queue (producer/consumer)
  private transcriptQueue: string[] = [];
  private pendingResolve: ((transcript: string) => void) | null = null;
  private pendingReject: ((error: Error) => void) | null = null;
  private pendingTimeout: ReturnType<typeof setTimeout> | null = null;

  // Serialization: ensure transcriptions complete in speech order
  private transcriptionChain: Promise<void> = Promise.resolve();
  private closed = false;

  // Callbacks
  private onPartialCallback: ((partial: string) => void) | null = null;

  constructor(serverUrl: string, silenceDurationMs: number) {
    this.serverUrl = serverUrl;
    this.silenceDurationMs = silenceDurationMs;
  }

  async connect(): Promise<void> {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 10000);
      const res = await fetch(`${this.serverUrl}/`, { signal: controller.signal });
      clearTimeout(timeout);
      if (res.ok) {
        this.connected = true;
        this.closed = false;
        console.error('[WhisperSTT] Connected to server');
        return;
      }
      throw new Error(`Server returned ${res.status}`);
    } catch (error) {
      throw new Error(`Failed to connect to Whisper server at ${this.serverUrl}: ${error}`);
    }
  }

  sendAudio(muLawData: Buffer): void {
    if (!this.connected) return;

    const energy = computeRmsEnergy(muLawData);

    // Compute adaptive thresholds
    const speechThreshold = Math.max(this.noiseFloor * SPEECH_MULTIPLIER, MIN_SPEECH_THRESHOLD);
    const silenceThreshold = Math.max(this.noiseFloor * SILENCE_MULTIPLIER, MIN_SILENCE_THRESHOLD);

    if (this.vadState === 'idle') {
      // Update noise floor only during idle (EMA)
      this.noiseFloor = this.noiseFloor * (1 - NOISE_FLOOR_ALPHA) + energy * NOISE_FLOOR_ALPHA;

      // Maintain rolling pre-buffer
      this.preBuffer.push(muLawData);
      if (this.preBuffer.length > PRE_BUFFER_FRAMES) {
        this.preBuffer.shift();
      }

      // Check for speech onset
      if (energy > speechThreshold) {
        this.speechFrameCount++;
        if (this.speechFrameCount >= SPEECH_CONFIRM_FRAMES) {
          // Speech confirmed — transition to speech state
          this.vadState = 'speech';
          this.speechStartMs = Date.now();
          this.silenceStartMs = null;
          // Prepend pre-buffer so we don't clip speech onset
          this.audioBuffer = [...this.preBuffer];
          this.preBuffer = [];
          console.error('[WhisperSTT] Speech started');
        }
      } else {
        this.speechFrameCount = 0;
      }
    } else {
      // In speech state — buffer audio
      this.audioBuffer.push(muLawData);

      if (energy < silenceThreshold) {
        // Silence detected
        if (this.silenceStartMs === null) {
          this.silenceStartMs = Date.now();
        } else if (Date.now() - this.silenceStartMs >= this.silenceDurationMs) {
          // Silence duration exceeded — speech ended
          const speechDuration = Date.now() - this.speechStartMs;
          console.error(`[WhisperSTT] Speech stopped (${speechDuration}ms)`);

          if (speechDuration >= MIN_SPEECH_MS) {
            // Serialize transcription — chain ensures speech-order delivery
            const audioChunks = [...this.audioBuffer];
            this.transcriptionChain = this.transcriptionChain
              .then(() => this.transcribeBufferedAudio(audioChunks))
              .catch(() => {}); // errors logged inside transcribeBufferedAudio
          } else {
            console.error('[WhisperSTT] Speech too short, discarding');
          }

          // Reset state
          this.vadState = 'idle';
          this.audioBuffer = [];
          this.speechFrameCount = 0;
          this.silenceStartMs = null;
        }
      } else {
        // Speech continues
        this.silenceStartMs = null;
      }
    }
  }

  private async transcribeBufferedAudio(audioChunks: Buffer[]): Promise<void> {
    if (audioChunks.length === 0 || this.closed) return;

    const muLawConcat = Buffer.concat(audioChunks);
    const pcm8k = muLawBufferToPcm16(muLawConcat);
    const pcm16k = resample8kTo16k(pcm8k);
    const wav = encodeWav(pcm16k, 16000);

    console.error(`[WhisperSTT] Transcribing ${(muLawConcat.length / 8000).toFixed(1)}s of audio...`);

    try {
      const formData = new FormData();
      formData.append('file', new Blob([wav], { type: 'audio/wav' }), 'audio.wav');
      formData.append('temperature', '0.0');
      formData.append('response_format', 'json');

      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 30000);
      const response = await fetch(`${this.serverUrl}/inference`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });
      clearTimeout(timeout);

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`[WhisperSTT] Transcription failed (${response.status}): ${errorText}`);
        return;
      }

      const result = await response.json() as { text?: string };
      const transcript = (result.text || '').trim();

      if (this.closed) {
        console.error('[WhisperSTT] Discarding transcript — session closed');
        return;
      }

      if (transcript) {
        console.error(`[WhisperSTT] Transcript: ${transcript}`);
        this.enqueueTranscript(transcript);
      } else {
        console.error('[WhisperSTT] Empty transcript (non-speech audio)');
      }
    } catch (error) {
      console.error('[WhisperSTT] Transcription error:', error);
    }
  }

  private enqueueTranscript(transcript: string): void {
    if (this.pendingResolve) {
      // Consumer is waiting — resolve immediately
      if (this.pendingTimeout) {
        clearTimeout(this.pendingTimeout);
        this.pendingTimeout = null;
      }
      const resolve = this.pendingResolve;
      this.pendingResolve = null;
      this.pendingReject = null;
      resolve(transcript);
    } else {
      // No consumer yet — queue for later
      this.transcriptQueue.push(transcript);
    }
  }

  async waitForTranscript(timeoutMs: number = 30000): Promise<string> {
    // Check queue first
    if (this.transcriptQueue.length > 0) {
      return this.transcriptQueue.shift()!;
    }

    // Wait for next transcript
    return new Promise((resolve, reject) => {
      this.pendingTimeout = setTimeout(() => {
        this.pendingResolve = null;
        this.pendingReject = null;
        this.pendingTimeout = null;
        reject(new Error('Transcript timeout'));
      }, timeoutMs);

      this.pendingResolve = resolve;
      this.pendingReject = reject;
    });
  }

  onPartial(callback: (partial: string) => void): void {
    // whisper.cpp batch API does not support streaming partials.
    // Store callback to satisfy interface, but it will not be called.
    this.onPartialCallback = callback;
  }

  close(): void {
    this.closed = true;
    this.connected = false;
    this.audioBuffer = [];
    this.preBuffer = [];
    this.transcriptQueue = [];
    this.vadState = 'idle';
    this.speechFrameCount = 0;
    this.silenceStartMs = null;

    if (this.pendingTimeout) {
      clearTimeout(this.pendingTimeout);
      this.pendingTimeout = null;
    }
    if (this.pendingReject) {
      this.pendingReject(new Error('Session closed'));
      this.pendingResolve = null;
      this.pendingReject = null;
    }
  }

  isConnected(): boolean {
    return this.connected;
  }
}
