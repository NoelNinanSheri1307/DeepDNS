import React from 'react';
import { DetectionResult } from '../types';

interface ObservationHorizonBarProps {
  currentCount: number;
  result: DetectionResult | null;
}

const HORIZONS = [5, 10, 15, 20, 25, 30];

export const ObservationHorizonBar: React.FC<ObservationHorizonBarProps> = ({
  currentCount,
  result,
}) => {
  const stoppingHorizon = result ? result.stopping_horizon : null;

  return (
    <div className="card-panel" style={{ gap: '16px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px' }}>
        <div>
          <span className="text-label">Adaptive Observation Horizons</span>
          <div style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
            Sequential Evaluation Checkpoints (K &isin; [5, 30] queries)
          </div>
        </div>
        <div className="status-pill">
          <span className="text-label">Observed:</span>
          <span className="font-serif" style={{ color: 'var(--text-primary)', fontWeight: 700 }}>
            {currentCount} / 30 Queries
          </span>
        </div>
      </div>

      {/* Progress Track */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: '10px' }}>
        {HORIZONS.map((k) => {
          const isEvaluated = currentCount >= k;
          const isStopPoint = stoppingHorizon === k;
          const isSaved = stoppingHorizon !== null && k > stoppingHorizon;

          let badgeBg = 'var(--bg-elevated)';
          let borderColor = 'var(--border-subtle)';
          let statusText = 'PENDING';
          let textColor = 'var(--text-muted)';
          let tooltipDetail = `Evaluation checkpoint K=${k}. If evidence is uncertain at this point, observation continues.`;

          if (isStopPoint) {
            badgeBg = result?.decision === 'ATTACK' ? 'rgba(255, 51, 85, 0.15)' : 'rgba(0, 229, 163, 0.15)';
            borderColor = result?.decision === 'ATTACK' ? 'var(--accent-crimson)' : 'var(--accent-cyan)';
            textColor = result?.decision === 'ATTACK' ? 'var(--accent-crimson)' : 'var(--accent-cyan)';
            statusText = result?.is_early_decision ? 'STOPPED EARLY' : 'BOUNDED STOP';
            tooltipDetail = `Model confidence crossed operating threshold at K=${k}. Observation terminated early, saving ${30 - k} queries.`;
          } else if (isSaved) {
            badgeBg = 'rgba(56, 189, 248, 0.05)';
            borderColor = 'rgba(56, 189, 248, 0.2)';
            textColor = 'var(--accent-blue)';
            statusText = 'SAVED (SKIPPED)';
            tooltipDetail = `Horizon K=${k} was skipped and saved because a confident decision was already made earlier at K=${stoppingHorizon}.`;
          } else if (isEvaluated) {
            badgeBg = 'var(--bg-subtle)';
            borderColor = 'var(--border-medium)';
            textColor = 'var(--text-secondary)';
            statusText = 'UNCERTAIN -> CONT';
            tooltipDetail = `Evaluated at K=${k}: Posterior probability remained in the uncertain zone between thresholds. Observation continued.`;
          }

          return (
            <div key={k} className="tooltip-wrapper" style={{ width: '100%' }}>
              <div
                style={{
                  width: '100%',
                  background: badgeBg,
                  border: `1px solid ${borderColor}`,
                  borderRadius: '3px',
                  padding: '12px 10px',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: '4px',
                  cursor: 'help',
                  transition: 'all 0.2s ease',
                }}
              >
                <span className="font-serif" style={{ fontSize: '15px', fontWeight: 700, color: isEvaluated || isStopPoint ? 'var(--text-primary)' : 'var(--text-dim)' }}>
                  K = {k}
                </span>
                <span
                  className="font-serif"
                  style={{
                    fontSize: '11px',
                    letterSpacing: '0.04em',
                    color: textColor,
                    fontWeight: 600,
                    textAlign: 'center',
                  }}
                >
                  {statusText}
                </span>
              </div>
              <div className="tooltip-box">
                <strong style={{ color: textColor, display: 'block', marginBottom: '3px' }}>
                  Horizon K={k} Status: {statusText}
                </strong>
                {tooltipDetail}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
