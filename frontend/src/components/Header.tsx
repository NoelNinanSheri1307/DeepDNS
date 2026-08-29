import React from 'react';
import { InferenceMode } from '../types';

export type ActiveTab = 'landing' | 'instrument';

interface HeaderProps {
  activeTab: ActiveTab;
  onSelectTab: (tab: ActiveTab) => void;
  mode: InferenceMode;
  onSelectMode: (mode: InferenceMode) => void;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  onSelectTab,
  mode,
  onSelectMode,
}) => {
  const modes: { id: InferenceMode; label: string; desc: string }[] = [
    {
      id: 'in_distribution',
      label: 'In-Distribution',
      desc: 'Standard Dual-View Fusion (Behavioral GRU + Lexical Char-CNN) calibrated on known attacks with τ_attack=0.95, τ_benign=0.15.',
    },
    {
      id: 'ood',
      label: 'OOD / LOMO',
      desc: 'Leave-One-Modality-Out (LOMO) model evaluated on 100% unseen video & compressed attack tunneling with τ_attack=0.80, τ_benign=0.01.',
    },
    {
      id: 'behavioral_only',
      label: 'Behavioral Only',
      desc: 'Ablation mode testing isolated 12 causal numerical features without domain character text analysis.',
    },
    {
      id: 'lexical_only',
      label: 'Lexical Only',
      desc: 'Ablation mode testing isolated Character-CNN text features without inter-arrival burst modeling.',
    },
  ];

  return (
    <header className="header-bar">
      {/* Brand & Tab Navigation */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '28px', flexWrap: 'wrap' }}>
        <div>
          <h1
            style={{
              fontFamily: 'var(--font-serif)',
              fontSize: '24px',
              letterSpacing: '0.04em',
              fontWeight: 700,
              color: 'var(--text-primary)',
              margin: 0,
            }}
          >
            DeepDNS
          </h1>
          <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            Sequential Multi-View DNS Tunneling Detection
          </span>
        </div>

        {/* Top Tab Switcher */}
        <div style={{ display: 'flex', gap: '6px', background: 'var(--bg-elevated)', padding: '4px', borderRadius: '4px', border: '1px solid var(--border-subtle)' }}>
          <button
            onClick={() => onSelectTab('landing')}
            className="btn-precision"
            style={{
              padding: '6px 14px',
              fontSize: '13px',
              background: activeTab === 'landing' ? 'var(--bg-subtle)' : 'transparent',
              borderColor: activeTab === 'landing' ? 'var(--accent-blue)' : 'transparent',
              color: activeTab === 'landing' ? 'var(--accent-blue)' : 'var(--text-muted)',
              fontWeight: activeTab === 'landing' ? 700 : 500,
            }}
          >
            Methodology & Science
          </button>
          <button
            onClick={() => onSelectTab('instrument')}
            className="btn-precision"
            style={{
              padding: '6px 14px',
              fontSize: '13px',
              background: activeTab === 'instrument' ? 'var(--bg-subtle)' : 'transparent',
              borderColor: activeTab === 'instrument' ? 'var(--accent-cyan)' : 'transparent',
              color: activeTab === 'instrument' ? 'var(--accent-cyan)' : 'var(--text-muted)',
              fontWeight: activeTab === 'instrument' ? 700 : 500,
            }}
          >
            Live Detection
          </button>
        </div>
      </div>

      {/* Mode Switcher: Only displayed when activeTab === 'instrument' */}
      {activeTab === 'instrument' && (
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
          <span className="text-label" style={{ marginRight: '4px' }}>Inference Mode:</span>
          {modes.map((m) => {
            const isSelected = mode === m.id;
            return (
              <div key={m.id} className="tooltip-wrapper">
                <button
                  onClick={() => onSelectMode(m.id)}
                  className="btn-precision"
                  style={{
                    padding: '6px 12px',
                    fontSize: '12px',
                    background: isSelected ? 'var(--bg-elevated)' : 'transparent',
                    borderColor: isSelected ? 'var(--accent-blue)' : 'var(--border-subtle)',
                    color: isSelected ? 'var(--accent-blue)' : 'var(--text-secondary)',
                    fontWeight: isSelected ? 700 : 500,
                  }}
                >
                  {m.label}
                </button>
                <div className="tooltip-box">
                  <strong style={{ color: 'var(--accent-blue)', display: 'block', marginBottom: '3px' }}>
                    {m.label} Architecture
                  </strong>
                  {m.desc}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </header>
  );
};
