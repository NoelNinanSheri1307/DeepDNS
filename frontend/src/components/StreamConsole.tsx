import React from 'react';
import { DNSObservation } from '../types';
import { Terminal } from 'lucide-react';

interface StreamConsoleProps {
  observations: DNSObservation[];
  stoppingHorizon: number | null;
}

export const StreamConsole: React.FC<StreamConsoleProps> = ({
  observations,
  stoppingHorizon,
}) => {
  return (
    <div className="card-panel" style={{ gap: '14px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Terminal size={16} color="var(--accent-blue)" />
          <div className="tooltip-wrapper">
            <span className="text-label term-hover">Live DNS Query Stream Buffer</span>
            <div className="tooltip-box tooltip-wide">
              <strong style={{ color: 'var(--accent-blue)', display: 'block', marginBottom: '3px' }}>
                Query Stream Buffer & Data Origin
              </strong>
              Holds the sequential DNS queries for the active session. When loading benchmark presets, queries are extracted directly from genuine CIC-Bell-DNS-Exf-2021 PCAP captures (e.g. <code>stateless_features-light_text.pcap.csv</code>, <code>stateless_features-benign_heavy_2.pcap.csv</code>). When building custom streams or uploading files, queries reflect the client-defined sequence.
            </div>
          </div>
        </div>
        <span className="font-serif" style={{ fontSize: '13px', color: 'var(--text-muted)' }}>
          {observations.length} queries in active buffer
        </span>
      </div>

      <div
        style={{
          background: 'var(--bg-primary)',
          border: '1px solid var(--border-subtle)',
          borderRadius: '3px',
          maxHeight: '260px',
          overflowY: 'auto',
          fontFamily: 'var(--font-serif)',
          fontSize: '13px',
        }}
      >
        {observations.length === 0 ? (
          <div style={{ padding: '32px', textAlign: 'center', color: 'var(--text-dim)' }}>
            No queries in buffer. Load a benchmark preset or upload CSV/JSON to begin.
          </div>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
            <thead>
              <tr style={{ background: 'var(--bg-elevated)', borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)', fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                <th style={{ padding: '8px 12px', width: '45px' }}>
                  <div className="tooltip-wrapper">
                    <span className="term-hover">#</span>
                    <div className="tooltip-box">
                      Sequential query index within this host session (1 to 30).
                    </div>
                  </div>
                </th>
                <th style={{ padding: '8px 12px' }}>
                  <div className="tooltip-wrapper">
                    <span className="term-hover">TIMESTAMP</span>
                    <div className="tooltip-box">
                      Arrival timestamp used to compute causal inter-arrival time &Delta;t = log(1 + &delta;t).
                    </div>
                  </div>
                </th>
                <th style={{ padding: '8px 12px' }}>
                  <div className="tooltip-wrapper">
                    <span className="term-hover">DOMAIN NAME / FQDN</span>
                    <div className="tooltip-box">
                      Fully Qualified Domain Name tokenized by the Character-CNN into character indices.
                    </div>
                  </div>
                </th>
                <th style={{ padding: '8px 12px', width: '70px' }}>
                  <div className="tooltip-wrapper">
                    <span className="term-hover">LEN</span>
                    <div className="tooltip-box">
                      Total string length of the FQDN query.
                    </div>
                  </div>
                </th>
                <th style={{ padding: '8px 12px', width: '80px' }}>
                  <div className="tooltip-wrapper">
                    <span className="term-hover">ENTROPY</span>
                    <div className="tooltip-box">
                      Shannon Character Entropy: H(X) = -&sum; p(x) log<sub>2</sub> p(x). High values (&gt; 2.5) indicate randomized exfiltration payloads.
                    </div>
                  </div>
                </th>
                <th style={{ padding: '8px 12px', width: '110px' }}>
                  <div className="tooltip-wrapper">
                    <span className="term-hover">STATUS</span>
                    <div className="tooltip-box">
                      Observation status: BUFFERED (waiting), HORIZON (AEC evaluation checkpoint), AEC STOP (terminal decision), or SKIPPED (saved early).
                    </div>
                  </div>
                </th>
              </tr>
            </thead>
            <tbody>
              {observations.map((obs, idx) => {
                const qNum = idx + 1;
                const isPastStop = stoppingHorizon !== null && qNum > stoppingHorizon;
                const isStopPoint = stoppingHorizon === qNum;

                let statusBadge = 'BUFFERED';
                let statusColor = 'var(--text-muted)';
                if (isStopPoint) {
                  statusBadge = 'AEC STOP';
                  statusColor = 'var(--accent-crimson)';
                } else if (isPastStop) {
                  statusBadge = 'SKIPPED';
                  statusColor = 'var(--accent-blue)';
                } else if (qNum % 5 === 0) {
                  statusBadge = 'HORIZON';
                  statusColor = 'var(--accent-amber)';
                }

                return (
                  <tr
                    key={idx}
                    style={{
                      borderBottom: '1px solid rgba(31, 36, 50, 0.4)',
                      background: isStopPoint ? 'rgba(255, 51, 85, 0.08)' : isPastStop ? 'rgba(56, 189, 248, 0.02)' : 'transparent',
                      opacity: isPastStop ? 0.35 : 1.0,
                    }}
                  >
                    <td style={{ padding: '8px 12px', color: 'var(--text-dim)' }}>{qNum}</td>
                    <td style={{ padding: '8px 12px', color: 'var(--text-muted)' }}>
                      {typeof obs.timestamp === 'string' ? obs.timestamp.split('T')[1] || obs.timestamp : obs.timestamp}
                    </td>
                    <td style={{ padding: '8px 12px', color: 'var(--text-primary)', wordBreak: 'break-all' }}>
                      {obs.domain_name}
                    </td>
                    <td style={{ padding: '8px 12px', color: 'var(--text-secondary)' }}>
                      {obs.fqdn_count || obs.domain_name.length}
                    </td>
                    <td style={{ padding: '8px 12px', color: 'var(--text-secondary)' }}>
                      {obs.entropy ? obs.entropy.toFixed(2) : '-'}
                    </td>
                    <td style={{ padding: '8px 12px' }}>
                      <span style={{ fontSize: '11px', fontWeight: 700, color: statusColor }}>
                        {statusBadge}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};
