import React from 'react';

export const WorkflowGuide: React.FC = () => {
  return (
    <div className="card-panel" style={{ gap: '14px', background: 'var(--bg-elevated)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span className="text-label" style={{ color: 'var(--accent-blue)', fontWeight: 'bold' }}>
          Interactive Workflow Guide — How to Use DeepDNS
        </span>
        <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
          4-Step Adaptive Detection Flow
        </span>
      </div>

      <div className="workflow-grid">
        <div className="workflow-card">
          <div className="workflow-number">STEP 1: SELECT TRAFFIC</div>
          <div style={{ fontSize: '13px', color: 'var(--text-primary)', fontWeight: 600 }}>
            Choose DNS Query Stream
          </div>
          <p style={{ fontSize: '12px', color: 'var(--text-secondary)', margin: 0 }}>
            Click one of the benchmark presets (<em>Attack</em>, <em>Benign</em>, <em>OOD</em>) or upload your own CSV/JSON network query capture.
          </p>
        </div>

        <div className="workflow-card">
          <div className="workflow-number">STEP 2: SELECT MODE</div>
          <div style={{ fontSize: '13px', color: 'var(--text-primary)', fontWeight: 600 }}>
            Configure Detection Boundary
          </div>
          <p style={{ fontSize: '12px', color: 'var(--text-secondary)', margin: 0 }}>
            Use <strong>In-Distribution</strong> (&tau;=0.95/0.15) for standard traffic or <strong>OOD / LOMO</strong> (&tau;=0.80/0.01) for unseen zero-shot modalities.
          </p>
        </div>

        <div className="workflow-card">
          <div className="workflow-number">STEP 3: RUN DETECTION</div>
          <div style={{ fontSize: '13px', color: 'var(--text-primary)', fontWeight: 600 }}>
            Start Stream or Batch
          </div>
          <p style={{ fontSize: '12px', color: 'var(--text-secondary)', margin: 0 }}>
            Click <strong>Start WebSocket Stream</strong> to observe queries arriving live in real-time, <strong>Step Query</strong> for manual inspection, or <strong>Instant Batch Detect</strong>.
          </p>
        </div>

        <div className="workflow-card">
          <div className="workflow-number">STEP 4: INSPECT AEC SAVINGS</div>
          <div style={{ fontSize: '13px', color: 'var(--text-primary)', fontWeight: 600 }}>
            Analyze Adaptive Decision
          </div>
          <p style={{ fontSize: '12px', color: 'var(--text-secondary)', margin: 0 }}>
            Watch DeepDNS terminate at early horizons (K=5, 10, 15...) as soon as confidence crosses threshold, saving 50% to 83.3% of network query volume.
          </p>
        </div>
      </div>
    </div>
  );
};
