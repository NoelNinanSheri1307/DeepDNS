import { DNSObservation, DetectionResult, HealthResponse, InferenceMode } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE_URL}/health`);
  if (!res.ok) {
    throw new Error(`Health check failed with status: ${res.status}`);
  }
  return res.json();
}

export async function detectBatch(
  observations: DNSObservation[],
  mode: InferenceMode = 'in_distribution',
  streamId: string = 'web_client_batch'
): Promise<DetectionResult> {
  const res = await fetch(`${API_BASE_URL}/detect`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      mode,
      stream_id: streamId,
      observations,
    }),
  });

  if (!res.ok) {
    const errBody = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(errBody.detail || `Inference request failed with code ${res.status}`);
  }

  return res.json();
}
