import React from 'react';
import { DetectionResult } from '../types';
import { AlertTriangle, CheckCircle2, Clock, Zap } from 'lucide-react';

interface DecisionPanelProps {
  result: DetectionResult | null;
  evaluating: boolean;
}

export const DecisionPanel: React.FC<DecisionPanelProps> = ({
  result,
  evaluating,
}) => {
  if (!result) {
    return (
      <div
        className="card-panel"
        style={{
          justifyContent: 'center',
          alignItems: 'center',
          minHeight: '230px',
          borderStyle: 'dashed',
          borderColor: 'var(--border-medium)',
        }}
      >
        <div style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
          {evaluating ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--accent-blue)' }}>
              <Clock className="animate-spin" size={18} />
              <span>Evaluating AEC Confidence Bounds...</span>
            </div>
          ) : (
            <div>
              <div className="font-serif" style={{ fontSize: '15px', color: 'var(--text-secondary)', marginBottom: '4px' }}>
                Awaiting Sequential Stream Execution
              </div>
              <div style={{ fontSize: '12px', color: 'var(--text-dim)' }}>
                Start a WebSocket stream or click Instant Batch Detect to evaluate evidence
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }

  const isAttack = result.decision === 'ATTACK';
  const confidencePercent = (result.confidence * 100).toFixed(2);
  const queriesSaved = 30 - result.stopping_horizon;
  const reductionPercent = ((queriesSaved / 30) * 100).toFixed(1);

  return (
    <div
      className="card-panel-elevated"
      style={{
        borderColor: isAttack ? 'var(--accent-crimson)' : 'var(--accent-cyan)',
        background: isAttack ? 'rgba(255, 51, 85, 0.04)' : 'rgba(0, 229, 163, 0.04)',
        gap: '18px',
      }}
    >
      {/* Top Banner */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '12px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          {isAttack ? (
            <AlertTriangle size={26} color="var(--accent-crimson)" />
          ) : (
            <CheckCircle2 size={26} color="var(--accent-cyan)" />
          )}
          <div>
            <span
              className="text-label"
              style={{
                color: isAttack ? 'var(--accent-crimson)' : 'var(--accent-cyan)',
                fontWeight: 700,
              }}
            >
              AEC TERMINAL DECISION REACHED
            </span>
            <h2 style={{ fontSize: '22px', margin: '2px 0 0 0', fontWeight: 700 }}>
              {isAttack ? 'ATTACK EXFILTRATION DETECTED' : 'BENIGN TRAFFIC CONFIRMED'}
            </h2>
          </div>
        </div>

        {/* Model Confidence with Tooltip */}
        <div className="tooltip-wrapper" style={{ textAlign: 'right' }}>
          <div style={{ cursor: 'help' }}>
            <span className="text-label term-hover">MODEL CONFIDENCE</span>
            <div
              className="font-serif"
              style={{
                fontSize: '28px',
                fontWeight: 700,
                color: isAttack ? 'var(--accent-crimson)' : 'var(--accent-cyan)',
                lineHeight: 1.1,
              }}
            >
              {confidencePercent}%
            </div>
          </div>
          <div className="tooltip-box tooltip-wide tooltip-top">
            <strong style={{ color: isAttack ? 'var(--accent-crimson)' : 'var(--accent-cyan)', display: 'block', marginBottom: '3px' }}>
              How Model Confidence Is Calculated
            </strong>
            Computed via the Softmax activation over the Dual-View Fusion linear head logits at stopping horizon K*={result.stopping_horizon}:
            <br />
            <code>P(Attack) = exp(logit_1) / (exp(logit_0) + exp(logit_1)) = {(result.attack_probability * 100).toFixed(2)}%</code>.
            <br />
            {isAttack
              ? `Since verdict is ATTACK, Confidence = P(Attack) = ${confidencePercent}%.`
              : `Since verdict is BENIGN, Confidence = P(Benign) = 1 - P(Attack) = ${confidencePercent}%.`}
          </div>
        </div>
      </div>

      {/* 4 Telemetry Metrics with Tooltips */}
      <div className="telemetry-grid">
        {/* Metric 1: Stopping Horizon */}
        <div className="telemetry-cell">
          <div className="tooltip-wrapper">
            <span className="text-label term-hover">STOPPING HORIZON (K*)</span>
            <div className="tooltip-box tooltip-top">
              <strong style={{ color: 'var(--accent-blue)', display: 'block', marginBottom: '3px' }}>
                Stopping Horizon Calculation
              </strong>
              The exact sequential observation checkpoint K &isin; [5, 10, 15, 20, 25, 30] where the Adaptive Evidence Controller triggered terminal stopping.
            </div>
          </div>
          <div className="telemetry-value" style={{ color: 'var(--text-primary)' }}>
            K = {result.stopping_horizon}
          </div>
          <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            Terminated at query #{result.stopping_horizon}
          </span>
        </div>

        {/* Metric 2: Queries Consumed */}
        <div className="telemetry-cell">
          <div className="tooltip-wrapper">
            <span className="text-label term-hover">QUERIES CONSUMED</span>
            <div className="tooltip-box tooltip-top">
              <strong style={{ color: 'var(--accent-blue)', display: 'block', marginBottom: '3px' }}>
                Queries Consumed Metric
              </strong>
              Total number of DNS packet observations ingested and evaluated before the decision boundary was satisfied ({result.stopping_horizon} queries).
            </div>
          </div>
          <div className="telemetry-value">
            {result.stopping_horizon}
          </div>
          <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            Observed before stopping
          </span>
        </div>

        {/* Metric 3: Queries Saved */}
        <div className="telemetry-cell">
          <div className="tooltip-wrapper">
            <span className="text-label term-hover">QUERIES SAVED</span>
            <div className="tooltip-box tooltip-top">
              <strong style={{ color: 'var(--accent-cyan)', display: 'block', marginBottom: '3px' }}>
                Queries Saved Calculation
              </strong>
              Calculated as: <code>30 - K* = 30 - {result.stopping_horizon} = +{queriesSaved} queries saved</code> compared to a static 30-query buffer.
            </div>
          </div>
          <div className="telemetry-value" style={{ color: queriesSaved > 0 ? 'var(--accent-cyan)' : 'var(--text-muted)' }}>
            +{queriesSaved}
          </div>
          <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            Queries avoided vs K=30
          </span>
        </div>

        {/* Metric 4: Query Reduction */}
        <div className="telemetry-cell">
          <div className="tooltip-wrapper">
            <span className="text-label term-hover">QUERY REDUCTION</span>
            <div className="tooltip-box tooltip-top">
              <strong style={{ color: 'var(--accent-cyan)', display: 'block', marginBottom: '3px' }}>
                Query Reduction % Calculation
              </strong>
              Calculated as: <code>((30 - K*) / 30) * 100 = ({queriesSaved} / 30) * 100 = {reductionPercent}%</code> reduction in resolver buffering overhead.
            </div>
          </div>
          <div className="telemetry-value" style={{ color: queriesSaved > 0 ? 'var(--accent-cyan)' : 'var(--text-muted)' }}>
            {reductionPercent}%
          </div>
          <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            Network volume saved
          </span>
        </div>
      </div>

      {/* Stopping Rationale Footer with Tooltip */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          background: 'var(--bg-elevated)',
          padding: '12px 16px',
          borderRadius: '3px',
          border: '1px solid var(--border-subtle)',
          fontSize: '13px',
          flexWrap: 'wrap',
          gap: '8px',
        }}
      >
        <div className="tooltip-wrapper">
          <div>
            <span className="text-label term-hover" style={{ display: 'block', marginBottom: '2px' }}>
              STOPPING RATIONALE
            </span>
            <span className="font-serif" style={{ color: 'var(--text-primary)', fontWeight: 600 }}>
              {result.stop_reason}
            </span>
          </div>
          <div className="tooltip-box tooltip-wide tooltip-top">
            <strong style={{ color: 'var(--accent-blue)', display: 'block', marginBottom: '3px' }}>
              Stopping Rule Decision Logic
            </strong>
            {result.is_early_decision ? (
              <span>
                <strong>EARLY STOP TRIGGERED:</strong> Confidence crossed the validation-calibrated decision threshold before horizon K=30, allowing immediate automated containment.
              </span>
            ) : (
              <span>
                <strong>END-OF-STREAM / BOUNDED FALLBACK:</strong> Sequence reached its end without crossing early thresholds. DeepDNS classified by 50% boundary (p &ge; 0.5 &rarr; ATTACK, p &lt; 0.5 &rarr; BENIGN) to guarantee zero resolver blocking.
              </span>
            )}
          </div>
        </div>

        {result.is_early_decision ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--accent-cyan)', fontSize: '12px', fontWeight: 600 }}>
            <Zap size={14} />
            <span>EARLY STOPPING SAVED {reductionPercent}% LATENCY</span>
          </div>
        ) : (
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--text-muted)', fontSize: '12px' }}>
            <Clock size={14} />
            <span>BOUNDED AT K={result.stopping_horizon}</span>
          </div>
        )}
      </div>
    </div>
  );
};
