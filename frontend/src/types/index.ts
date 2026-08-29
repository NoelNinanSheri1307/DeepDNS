/**
 * Canonical Types for DeepDNS Web Application
 */

export type InferenceMode = 'in_distribution' | 'ood' | 'behavioral_only' | 'lexical_only';

export interface DNSObservation {
  timestamp: string | number;
  domain_name: string;
  fqdn_count?: number;
  subdomain_length?: number;
  upper?: number;
  lower?: number;
  numeric?: number;
  entropy?: number;
  special?: number;
  labels?: number;
  labels_max?: number;
  labels_average?: number;
  len?: number;
  subdomain?: number;
}

export interface DetectionResult {
  decision: 'ATTACK' | 'BENIGN';
  confidence: number;
  attack_probability: number;
  benign_probability: number;
  stopping_horizon: number;
  max_horizon: number;
  observations_consumed: number;
  observations_saved: number;
  query_savings_pct: number;
  is_early_decision: boolean;
  stop_reason: string;
  evidence_history: Record<string, number>;
  model_name: string;
  mode: string;
}

export interface HealthResponse {
  status: string;
  service: string;
  engine: string;
  available_modes: string[];
}

export type StreamState =
  | 'IDLE'
  | 'CONNECTING'
  | 'READY'
  | 'RECEIVING'
  | 'EVALUATING'
  | 'DECISION_REACHED'
  | 'ERROR';

export interface WebSocketEventMessage {
  event: 'CONNECTED' | 'OBSERVATION_BUFFERED' | 'EVIDENCE_UPDATE' | 'DECISION' | 'ERROR';
  stream_id: string;
  data: any;
}
