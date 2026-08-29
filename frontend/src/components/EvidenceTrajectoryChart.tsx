import React from 'react';
import { DetectionResult, InferenceMode } from '../types';

interface EvidenceTrajectoryChartProps {
  evidenceHistory: Record<string, number>;
  result: DetectionResult | null;
  mode: InferenceMode;
}

export const EvidenceTrajectoryChart: React.FC<EvidenceTrajectoryChartProps> = ({
  evidenceHistory,
  result,
  mode,
}) => {
  const tauAttack = mode === 'ood' ? 0.80 : 0.95;
  const tauBenign = mode === 'ood' ? 0.01 : 0.15;

  const horizons = [5, 10, 15, 20, 25, 30];
  const svgWidth = 640;
  const svgHeight = 230;
  const padLeft = 50;
  const padRight = 30;
  const padTop = 24;
  const padBottom = 34;

  const plotWidth = svgWidth - padLeft - padRight;
  const plotHeight = svgHeight - padTop - padBottom;

  const getX = (k: number) => {
    const idx = horizons.indexOf(k);
    return padLeft + (idx / (horizons.length - 1)) * plotWidth;
  };

  const getY = (prob: number) => {
    return padTop + (1.0 - Math.min(Math.max(prob, 0.0), 1.0)) * plotHeight;
  };

  // Trajectory points
  const points: { k: number; p: number; x: number; y: number }[] = [];
  horizons.forEach((k) => {
    if (evidenceHistory[k.toString()] !== undefined) {
      const p = evidenceHistory[k.toString()];
      points.push({ k, p, x: getX(k), y: getY(p) });
    }
  });

  const pathD = points.length > 0
    ? points.map((pt, i) => (i === 0 ? `M ${pt.x} ${pt.y}` : `L ${pt.x} ${pt.y}`)).join(' ')
    : '';

  const stopK = result ? result.stopping_horizon : null;

  return (
    <div className="card-panel" style={{ gap: '14px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px' }}>
        <div>
          <div className="tooltip-wrapper">
            <span className="text-label term-hover">Sequential Evidence Trajectory</span>
            <div className="tooltip-box">
              <strong style={{ color: 'var(--accent-blue)', display: 'block', marginBottom: '3px' }}>
                Posterior Probability Curve P(Attack)
              </strong>
              Plots the sequential probability of exfiltration computed by the Dual-View Fusion network at each evaluation checkpoint.
            </div>
          </div>
          <div style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
            Posterior Probability P(Attack) vs Decision Thresholds
          </div>
        </div>

        <div style={{ display: 'flex', gap: '14px', alignItems: 'center' }}>
          {/* Attack Threshold Tooltip */}
          <div className="tooltip-wrapper">
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'help' }}>
              <span style={{ width: '12px', height: '2px', background: 'var(--accent-crimson)', display: 'inline-block' }} />
              <span className="font-serif term-hover" style={{ fontSize: '12px', color: 'var(--text-muted)' }}>&tau;<sub>atk</sub> ({tauAttack})</span>
            </div>
            <div className="tooltip-box tooltip-wide">
              <strong style={{ color: 'var(--accent-crimson)', display: 'block', marginBottom: '3px' }}>
                Attack Stopping Boundary (&tau;<sub>atk</sub> = {tauAttack})
              </strong>
              Calibrated strictly on validation data to achieve &ge; 98.17% exfiltration recall while maintaining an ultra-low False Positive Rate (FPR &le; 0.12%). If P(Attack) &ge; {tauAttack} at any horizon K &isin; [5, 10, 15, 20, 25, 30], observation terminates early to save query overhead and block the attack immediately. {mode === 'ood' ? 'In OOD mode, τ=0.80 provides optimal sensitivity to zero-shot unseen payload formats.' : ''}
            </div>
          </div>

          {/* Benign Threshold Tooltip */}
          <div className="tooltip-wrapper">
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'help' }}>
              <span style={{ width: '12px', height: '2px', background: 'var(--accent-cyan)', display: 'inline-block' }} />
              <span className="font-serif term-hover" style={{ fontSize: '12px', color: 'var(--text-muted)' }}>&tau;<sub>ben</sub> ({tauBenign})</span>
            </div>
            <div className="tooltip-box tooltip-wide">
              <strong style={{ color: 'var(--accent-cyan)', display: 'block', marginBottom: '3px' }}>
                Benign Stopping Boundary (&tau;<sub>ben</sub> = {tauBenign})
              </strong>
              Calibrated strictly on validation data to rapidly confirm legitimate corporate/CDN traffic within 5 to 10 queries, saving up to 83.3% of resolver query buffering without risk of missing subtle tunnels. {mode === 'ood' ? 'In OOD mode, τ=0.01 provides high stringency to prevent false negative leaks on stealthy video exfiltration.' : ''}
            </div>
          </div>
        </div>
      </div>

      <div style={{ width: '100%', overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
          {/* Background Grid & Axis */}
          <rect x={padLeft} y={padTop} width={plotWidth} height={plotHeight} fill="rgba(14, 16, 23, 0.6)" stroke="var(--border-subtle)" strokeWidth="1" />

          {/* Grid lines */}
          {[0.0, 0.25, 0.5, 0.75, 1.0].map((level) => {
            const y = getY(level);
            return (
              <g key={level}>
                <line x1={padLeft} y1={y} x2={svgWidth - padRight} y2={y} stroke="var(--border-subtle)" strokeDasharray="3 3" strokeWidth="1" />
                <text x={padLeft - 8} y={y + 4} textAnchor="end" fill="var(--text-dim)" fontSize="11" fontFamily="var(--font-serif)">
                  {level.toFixed(2)}
                </text>
              </g>
            );
          })}

          {/* X Axis Labels */}
          {horizons.map((k) => {
            const x = getX(k);
            return (
              <g key={k}>
                <line x1={x} y1={padTop} x2={x} y2={padTop + plotHeight} stroke="var(--border-subtle)" strokeDasharray="2 2" strokeWidth="1" />
                <text x={x} y={svgHeight - 12} textAnchor="middle" fill="var(--text-muted)" fontSize="12" fontFamily="var(--font-serif)">
                  K={k}
                </text>
              </g>
            );
          })}

          {/* Threshold Lines */}
          <line
            x1={padLeft}
            y1={getY(tauAttack)}
            x2={svgWidth - padRight}
            y2={getY(tauAttack)}
            stroke="var(--accent-crimson)"
            strokeWidth="1.5"
            strokeDasharray="4 4"
          />
          <line
            x1={padLeft}
            y1={getY(tauBenign)}
            x2={svgWidth - padRight}
            y2={getY(tauBenign)}
            stroke="var(--accent-cyan)"
            strokeWidth="1.5"
            strokeDasharray="4 4"
          />

          {/* Query Savings Zone */}
          {stopK && stopK < 30 && (
            <rect
              x={getX(stopK)}
              y={padTop}
              width={svgWidth - padRight - getX(stopK)}
              height={plotHeight}
              fill="rgba(56, 189, 248, 0.04)"
              stroke="rgba(56, 189, 248, 0.15)"
              strokeDasharray="2 2"
            />
          )}

          {/* Evidence Path */}
          {pathD && (
            <path
              d={pathD}
              fill="none"
              stroke="var(--accent-blue)"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )}

          {/* Observation Points */}
          {points.map((pt) => {
            const isTerminal = stopK === pt.k;
            const ptColor = isTerminal
              ? result?.decision === 'ATTACK' ? 'var(--accent-crimson)' : 'var(--accent-cyan)'
              : 'var(--accent-blue)';

            return (
              <g key={pt.k}>
                <circle cx={pt.x} cy={pt.y} r={isTerminal ? 6 : 4} fill={ptColor} stroke="#0e1017" strokeWidth="2" />
                <text
                  x={pt.x}
                  y={pt.y - 10}
                  textAnchor="middle"
                  fill="var(--text-primary)"
                  fontSize="11"
                  fontWeight="700"
                  fontFamily="var(--font-serif)"
                >
                  {(pt.p * 100).toFixed(1)}%
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
};
