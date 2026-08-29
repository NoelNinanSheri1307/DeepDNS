import { DNSObservation, DetectionResult, InferenceMode, WebSocketEventMessage } from '../types';

const WS_BASE_URL = import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8000';

export interface StreamClientCallbacks {
  onConnected?: (data: { mode: string; status: string }) => void;
  onBuffered?: (count: number) => void;
  onEvidenceUpdate?: (data: {
    horizon: number;
    attack_probability: number;
    benign_probability: number;
    status: string;
    evidence_history: Record<string, number>;
  }) => void;
  onDecision?: (result: DetectionResult) => void;
  onError?: (error: string) => void;
  onClose?: () => void;
}

export class DeepDNSStreamClient {
  private socket: WebSocket | null = null;
  public streamId: string = '';
  private callbacks: StreamClientCallbacks = {};

  constructor(callbacks: StreamClientCallbacks = {}) {
    this.callbacks = callbacks;
  }

  public connect(streamId: string, mode: InferenceMode = 'in_distribution') {
    this.disconnect();
    this.streamId = streamId;

    const url = `${WS_BASE_URL}/ws/stream/${encodeURIComponent(streamId)}?mode=${encodeURIComponent(mode)}`;
    this.socket = new WebSocket(url);

    this.socket.onopen = () => {
      // WebSocket open
    };

    this.socket.onmessage = (event: MessageEvent) => {
      try {
        const msg: WebSocketEventMessage = JSON.parse(event.data);
        this.handleMessage(msg);
      } catch (err) {
        if (this.callbacks.onError) {
          this.callbacks.onError('Received malformed WebSocket frame.');
        }
      }
    };

    this.socket.onerror = () => {
      if (this.callbacks.onError) {
        this.callbacks.onError('WebSocket connection encountered an error.');
      }
    };

    this.socket.onclose = () => {
      if (this.callbacks.onClose) {
        this.callbacks.onClose();
      }
    };
  }

  private handleMessage(msg: WebSocketEventMessage) {
    switch (msg.event) {
      case 'CONNECTED':
        this.callbacks.onConnected?.(msg.data);
        break;
      case 'OBSERVATION_BUFFERED':
        this.callbacks.onBuffered?.(msg.data.observations_count);
        break;
      case 'EVIDENCE_UPDATE':
        this.callbacks.onEvidenceUpdate?.(msg.data);
        break;
      case 'DECISION':
        this.callbacks.onDecision?.(msg.data as DetectionResult);
        break;
      case 'ERROR':
        this.callbacks.onError?.(msg.data.message || 'Stream processing error');
        break;
    }
  }

  public sendObservation(obs: DNSObservation) {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify(obs));
    } else {
      if (this.callbacks.onError) {
        this.callbacks.onError('Cannot send query: WebSocket is not open.');
      }
    }
  }

  public disconnect() {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
  }
}
