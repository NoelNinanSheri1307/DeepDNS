import React from 'react';
import { InferenceMode } from '../types';

interface PipelineOverviewProps {
  mode: InferenceMode;
}

export const PipelineOverview: React.FC<PipelineOverviewProps> = ({ mode }) => {
  const getThresholds = () => {
    if (mode === 'ood') return { atk: '0.80', ben: '0.01', name: 'OOD Dual-View Fusion' };
    if (mode === 'behavioral_only') return { atk: '0.95', ben: '0.15', name: 'Behavioral-Only GRU' };
    if (mode === 'lexical_only') return { atk: '0.95', ben: '0.15', name: 'Lexical-Only CNN' };
    return { atk: '0.95', ben: '0.15', name: 'In-Distribution Dual-View Fusion' };
  };

  const { atk, ben, name } = getThresholds();

  return (
    <div className="card-panel" style={{ padding: '16px 22px', gap: '12px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span className="text-label">Active Architecture & Evidence Controller</span>
        <span className="font-serif" style={{ fontSize: '13px', color: 'var(--accent-blue)', fontWeight: 600 }}>
          {name}
        </span>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
          gap: '12px',
        }}
      >
        {/* Behavioral View */}
        <div className="tooltip-wrapper" style={{ width: '100%' }}>
          <div
            style={{
              width: '100%',
              background: 'var(--bg-elevated)',
              padding: '12px 14px',
              borderRadius: '3px',
              border: '1px solid var(--border-subtle)',
              display: 'flex',
              flexDirection: 'column',
              gap: '2px',
              cursor: 'help',
            }}
          >
            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>BEHAVIORAL VIEW</div>
            <div className="font-serif" style={{ fontSize: '14px', fontWeight: 700, color: 'var(--accent-blue)' }}>Temporal GRU (64)</div>
          </div>
          <div className="tooltip-box tooltip-wide">
            <strong style={{ color: 'var(--accent-blue)', display: 'block', marginBottom: '4px' }}>
              Behavioral Temporal GRU Encoder
            </strong>
            Takes the 12 causal numerical metrics (log inter-arrival time &Delta;t = log(1+&delta;t), FQDN length, label entropy, digit ratio) and updates recurrent hidden state h<sub>t</sub> &isin; R<sup>64</sup> chronologically without future leakage.
          </div>
        </div>

        {/* Lexical View */}
        <div className="tooltip-wrapper" style={{ width: '100%' }}>
          <div
            style={{
              width: '100%',
              background: 'var(--bg-elevated)',
              padding: '12px 14px',
              borderRadius: '3px',
              border: '1px solid var(--border-subtle)',
              display: 'flex',
              flexDirection: 'column',
              gap: '2px',
              cursor: 'help',
            }}
          >
            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>LEXICAL VIEW</div>
            <div className="font-serif" style={{ fontSize: '14px', fontWeight: 700, color: 'var(--accent-cyan)' }}>Char-CNN (128)</div>
          </div>
          <div className="tooltip-box tooltip-wide">
            <strong style={{ color: 'var(--accent-cyan)', display: 'block', marginBottom: '4px' }}>
              Character-level Lexical CNN
            </strong>
            Encodes raw domain query strings into character tokens using a 45-character ASCII vocabulary and 1D convolutions (kernel sizes 3, 4, 5) to generate a 128-dimensional embedding of domain randomness and chunking patterns.
          </div>
        </div>

        {/* Dual-View Fusion */}
        <div className="tooltip-wrapper" style={{ width: '100%' }}>
          <div
            style={{
              width: '100%',
              background: 'var(--bg-elevated)',
              padding: '12px 14px',
              borderRadius: '3px',
              border: '1px solid var(--border-subtle)',
              display: 'flex',
              flexDirection: 'column',
              gap: '2px',
              cursor: 'help',
            }}
          >
            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>DUAL-VIEW FUSION</div>
            <div className="font-serif" style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-primary)' }}>MLP Head (64)</div>
          </div>
          <div className="tooltip-box tooltip-wide">
            <strong style={{ color: 'var(--text-primary)', display: 'block', marginBottom: '4px' }}>
              Dual-View Fusion MLP
            </strong>
            Concatenates behavioral [z<sub>beh</sub>] and lexical [z<sub>lex</sub>] representations into z<sub>fuse</sub> &isin; R<sup>64</sup>, passing into a linear classification head to produce sequential classification logits.
          </div>
        </div>

        {/* AEC Boundaries */}
        <div className="tooltip-wrapper" style={{ width: '100%' }}>
          <div
            style={{
              width: '100%',
              background: 'var(--bg-elevated)',
              padding: '12px 14px',
              borderRadius: '3px',
              border: '1px solid var(--border-subtle)',
              display: 'flex',
              flexDirection: 'column',
              gap: '2px',
              cursor: 'help',
            }}
          >
            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>AEC BOUNDARIES</div>
            <div className="font-serif" style={{ fontSize: '14px', fontWeight: 700, color: 'var(--accent-amber)' }}>
              &tau;<sub>atk</sub>={atk} | &tau;<sub>ben</sub>={ben}
            </div>
          </div>
          <div className="tooltip-box tooltip-wide">
            <strong style={{ color: 'var(--accent-amber)', display: 'block', marginBottom: '4px' }}>
              Adaptive Evidence Controller (AEC)
            </strong>
            Evaluates posterior probabilities P(Attack) at horizons K &isin; [5, 10, 15, 20, 25, 30]. If P(Attack) &ge; &tau;<sub>atk</sub> &rarr; Stop (Attack). If P(Attack) &le; &tau;<sub>ben</sub> &rarr; Stop (Benign). Else &rarr; Continue to next horizon.
          </div>
        </div>
      </div>
    </div>
  );
};
