import React, { useRef } from 'react';
import { DNSObservation, StreamState } from '../types';
import { Play, RotateCcw, Upload, Activity, HelpCircle } from 'lucide-react';
import { SAMPLE_ATTACK_STREAM, SAMPLE_BENIGN_STREAM, SAMPLE_OOD_ATTACK_STREAM } from '../data/sampleStreams';

interface InputControllerProps {
  onLoadStream: (stream: DNSObservation[], name: string) => void;
  onStartStreaming: () => void;
  onBatchDetect: () => void;
  onReset: () => void;
  onOpenSchemaModal: () => void;
  streamState: StreamState;
  hasObservations: boolean;
  activeDatasetName: string;
}

export const InputController: React.FC<InputControllerProps> = ({
  onLoadStream,
  onStartStreaming,
  onBatchDetect,
  onReset,
  onOpenSchemaModal,
  streamState,
  hasObservations,
  activeDatasetName,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const isStreaming = streamState === 'RECEIVING' || streamState === 'EVALUATING';
  const isTerminated = streamState === 'DECISION_REACHED';

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const text = event.target?.result as string;
        if (file.name.endsWith('.json')) {
          const parsed = JSON.parse(text);
          const obsList: DNSObservation[] = Array.isArray(parsed) ? parsed : parsed.observations || [];
          if (obsList.length > 0) {
            onLoadStream(obsList, file.name);
          }
        } else if (file.name.endsWith('.csv')) {
          const lines = text.trim().split('\n');
          const headers = lines[0].split(',').map((h) => h.trim());
          const records: DNSObservation[] = [];
          for (let i = 1; i < lines.length && records.length < 30; i++) {
            const values = lines[i].split(',').map((v) => v.trim());
            const row: any = {};
            headers.forEach((h, idx) => {
              row[h] = values[idx];
            });
            records.push({
              timestamp: row.timestamp || `2026-08-28T12:00:00.${i}`,
              domain_name: row.domain_name || row.domain || row.sld || `query-${i}.com`,
              fqdn_count: parseFloat(row.FQDN_count || row.fqdn_count || 30),
              subdomain_length: parseFloat(row.subdomain_length || 10),
              entropy: parseFloat(row.entropy || 2.5),
              numeric: parseFloat(row.numeric || 2),
              upper: parseFloat(row.upper || 0),
              special: parseFloat(row.special || 2),
            });
          }
          if (records.length > 0) {
            onLoadStream(records, file.name);
          }
        }
      } catch (err) {
        alert('Failed to parse input file. Ensure format is valid JSON or CSV.');
      }
    };
    reader.readAsText(file);
  };

  return (
    <div className="card-panel" style={{ gap: '16px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px' }}>
        <div>
          <span className="text-label">Stream Controls & Benchmark Presets</span>
          <div style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
            Active Dataset in Memory: <strong style={{ color: 'var(--text-primary)' }}>{activeDatasetName || 'None'}</strong>
          </div>
        </div>

        {/* Preset Buttons */}
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          <button
            onClick={() => onLoadStream(SAMPLE_ATTACK_STREAM, 'CIC-Bell Light Text Attack')}
            className="btn-precision btn-crimson"
            style={{ fontSize: '12px', padding: '6px 12px' }}
            disabled={isStreaming}
          >
            Load Attack Stream (ID)
          </button>
          <button
            onClick={() => onLoadStream(SAMPLE_BENIGN_STREAM, 'Enterprise CDN Benign Traffic')}
            className="btn-precision btn-cyan"
            style={{ fontSize: '12px', padding: '6px 12px' }}
            disabled={isStreaming}
          >
            Load Benign Stream
          </button>
          <button
            onClick={() => onLoadStream(SAMPLE_OOD_ATTACK_STREAM, 'LOMO Video Exfiltration (OOD)')}
            className="btn-precision"
            style={{ fontSize: '12px', padding: '6px 12px', color: 'var(--accent-amber)' }}
            disabled={isStreaming}
          >
            Load OOD Attack
          </button>
        </div>
      </div>

      {/* Action Controls */}
      <div style={{ display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
        {/* Start Stream Button with Tooltip */}
        <div className="tooltip-wrapper">
          <button
            onClick={onStartStreaming}
            disabled={!hasObservations || isStreaming || isTerminated}
            className="btn-precision btn-primary"
            style={{ padding: '9px 18px', fontWeight: 700 }}
          >
            <Play size={14} />
            <span>START WEBSOCKET STREAM</span>
          </button>
          <div className="tooltip-box">
            Connects to the real-time WebSocket endpoint (<code>/ws/stream/&#123;id&#125;</code>) and feeds queries one-by-one to simulate real network traffic, triggering AEC evaluation at K=5, 10, 15, 20, 25, 30.
          </div>
        </div>

        {/* Instant Batch Detect Button with Tooltip */}
        <div className="tooltip-wrapper">
          <button
            onClick={onBatchDetect}
            disabled={!hasObservations || isStreaming}
            className="btn-precision"
          >
            <Activity size={14} />
            <span>INSTANT BATCH DETECT (/detect)</span>
          </button>
          <div className="tooltip-box tooltip-wide">
            <strong style={{ color: 'var(--accent-blue)', display: 'block', marginBottom: '3px' }}>
              Instant Batch Evaluation Pipeline (POST /detect)
            </strong>
            Sends the loaded observation sequence (5 to 30 queries) in a single HTTP POST request. The PyTorch Dual-View model runs forward passes sequentially across horizons K &isin; [5, 10, 15, 20, 25, 30]. If confidence crosses &tau;<sub>attack</sub> or &tau;<sub>benign</sub>, it returns an immediate early stopping result. If the stream is unknown/ambiguous and evidence remains uncertain through K=30, the AEC executes a bounded fallback decision at K=30 (P &ge; 0.5 &rarr; ATTACK, P &lt; 0.5 &rarr; BENIGN with reason FORCED_HORIZON_REACHED), guaranteeing the resolver never hangs indefinitely.
          </div>
        </div>

        {/* Upload Button */}
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={isStreaming}
          className="btn-precision"
        >
          <Upload size={14} />
          <span>UPLOAD CSV / JSON</span>
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".json,.csv"
          onChange={handleFileUpload}
          style={{ display: 'none' }}
        />

        {/* Schema & Download Help Button */}
        <button
          onClick={onOpenSchemaModal}
          className="btn-precision"
          style={{ color: 'var(--accent-blue)', borderColor: '#2b4b68' }}
        >
          <HelpCircle size={14} />
          <span>SCHEMA & SAMPLES</span>
        </button>

        {/* Reset Button */}
        <button
          onClick={onReset}
          className="btn-precision"
          style={{ marginLeft: 'auto', color: 'var(--text-muted)' }}
        >
          <RotateCcw size={14} />
          <span>RESET SESSION</span>
        </button>
      </div>
    </div>
  );
};
