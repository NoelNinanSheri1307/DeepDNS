import React, { useState } from 'react';
import { DNSObservation } from '../types';
import {
  SAMPLE_ATTACK_1,
  SAMPLE_ATTACK_2,
  SAMPLE_ATTACK_3,
  SAMPLE_BENIGN_1,
  SAMPLE_BENIGN_2,
  SAMPLE_BENIGN_3,
} from '../data/sampleStreams';

interface CustomStreamBuilderProps {
  onApplyCustomStream: (observations: DNSObservation[], name: string) => void;
}

// Compute Shannon Character Entropy
const computeEntropy = (str: string): number => {
  if (!str) return 0;
  const counts: Record<string, number> = {};
  for (const ch of str) {
    counts[ch] = (counts[ch] || 0) + 1;
  }
  let entropy = 0;
  const len = str.length;
  for (const ch in counts) {
    const p = counts[ch] / len;
    entropy -= p * Math.log2(p);
  }
  return parseFloat(entropy.toFixed(3));
};

// Compute lexical breakdown from domain name
const extractDomainStats = (domain: string) => {
  const fqdnLen = domain.length;
  const parts = domain.split('.');
  const sub = parts.length > 2 ? parts.slice(0, -2).join('.') : '';
  const subLen = sub.length;

  let upper = 0;
  let lower = 0;
  let numeric = 0;
  let special = 0;

  for (const ch of domain) {
    if (ch >= 'A' && ch <= 'Z') upper++;
    else if (ch >= 'a' && ch <= 'z') lower++;
    else if (ch >= '0' && ch <= '9') numeric++;
    else special++;
  }

  const entropy = computeEntropy(domain);
  const labels = parts.length;
  const labelsMax = Math.max(...parts.map((p) => p.length), 0);
  const labelsAvg = parseFloat((fqdnLen / Math.max(1, labels)).toFixed(2));

  return {
    fqdn_count: fqdnLen,
    subdomain_length: subLen,
    upper,
    lower,
    numeric,
    special,
    entropy,
    labels,
    labels_max: labelsMax,
    labels_average: labelsAvg,
    len: fqdnLen,
    subdomain: subLen > 0 ? 1 : 0,
  };
};

export const CustomStreamBuilder: React.FC<CustomStreamBuilderProps> = ({
  onApplyCustomStream,
}) => {
  const [isOpen, setIsOpen] = useState<boolean>(false);
  const [builderTab, setBuilderTab] = useState<'form' | 'json'>('form');
  const [randomCount, setRandomCount] = useState<number>(15);

  // Working state of custom observations
  const [customRows, setCustomRows] = useState<DNSObservation[]>(() =>
    JSON.parse(JSON.stringify(SAMPLE_ATTACK_1.slice(0, 10)))
  );

  const [rawJsonText, setRawJsonText] = useState<string>(() =>
    JSON.stringify(SAMPLE_ATTACK_1.slice(0, 10), null, 2)
  );
  const [jsonError, setJsonError] = useState<string | null>(null);

  const handleUpdateDomain = (idx: number, newDomain: string) => {
    const updated = [...customRows];
    const stats = extractDomainStats(newDomain);
    updated[idx] = {
      ...updated[idx],
      domain_name: newDomain,
      ...stats,
    };
    setCustomRows(updated);
    setRawJsonText(JSON.stringify(updated, null, 2));
  };

  const handleUpdateNumericField = (idx: number, field: keyof DNSObservation, value: string) => {
    const updated = [...customRows];
    updated[idx] = {
      ...updated[idx],
      [field]: parseFloat(value) || 0,
    };
    setCustomRows(updated);
    setRawJsonText(JSON.stringify(updated, null, 2));
  };

  const handleUpdateTimestamp = (idx: number, value: string) => {
    const updated = [...customRows];
    updated[idx] = {
      ...updated[idx],
      timestamp: value,
    };
    setCustomRows(updated);
    setRawJsonText(JSON.stringify(updated, null, 2));
  };

  const handleAddRow = () => {
    if (customRows.length >= 30) return;
    const lastRow = customRows[customRows.length - 1] || SAMPLE_ATTACK_1[0];
    const newIdx = customRows.length + 1;
    const baseTime = new Date(lastRow.timestamp || Date.now());
    const newTime = new Date(baseTime.getTime() + 300).toISOString();
    const domain = `chunk-${newIdx}.payload-tunnel.attacker-dns.org`;
    const stats = extractDomainStats(domain);

    const newRow: DNSObservation = {
      timestamp: newTime,
      domain_name: domain,
      ...stats,
    };

    const updated = [...customRows, newRow];
    setCustomRows(updated);
    setRawJsonText(JSON.stringify(updated, null, 2));
  };

  const handleRemoveRow = (idx: number) => {
    if (customRows.length <= 5) {
      alert('DeepDNS requires at least 5 observations in a stream sequence.');
      return;
    }
    const updated = customRows.filter((_, i) => i !== idx);
    setCustomRows(updated);
    setRawJsonText(JSON.stringify(updated, null, 2));
  };

  const handleLoadTemplate = (template: 'attack' | 'benign') => {
    const src = template === 'attack' ? SAMPLE_ATTACK_1.slice(0, 10) : SAMPLE_BENIGN_1.slice(0, 10);
    const copied = JSON.parse(JSON.stringify(src));
    setCustomRows(copied);
    setRawJsonText(JSON.stringify(copied, null, 2));
    setJsonError(null);
  };

  // Generate Realistic Random Sequences within Dataset Bounds
  const handleGenerateRealisticRandom = (type: 'attack' | 'benign') => {
    const count = Math.min(30, Math.max(5, randomCount));
    const generated: DNSObservation[] = [];
    const baseTime = new Date();

    const attackPool = [...SAMPLE_ATTACK_1, ...SAMPLE_ATTACK_2, ...SAMPLE_ATTACK_3];
    const benignPool = [...SAMPLE_BENIGN_1, ...SAMPLE_BENIGN_2, ...SAMPLE_BENIGN_3];

    const sourcePool = type === 'attack' ? attackPool : benignPool;
    const maxStart = Math.max(0, sourcePool.length - count);
    const startIdx = Math.floor(Math.random() * (maxStart + 1));

    for (let i = 0; i < count; i++) {
      const srcRow = sourcePool[(startIdx + i) % sourcePool.length];
      const interval = type === 'attack' ? 120 : 600;
      const timeStr = new Date(baseTime.getTime() + i * interval).toISOString();

      generated.push({
        ...srcRow,
        timestamp: timeStr,
      });
    }

    setCustomRows(generated);
    setRawJsonText(JSON.stringify(generated, null, 2));
    setJsonError(null);
  };

  const handleApply = () => {
    if (builderTab === 'json') {
      try {
        const parsed = JSON.parse(rawJsonText);
        const obsList: DNSObservation[] = Array.isArray(parsed) ? parsed : parsed.observations || [];
        if (!Array.isArray(obsList) || obsList.length < 5) {
          setJsonError('JSON must contain an array of at least 5 DNS observation objects.');
          return;
        }
        setJsonError(null);
        setCustomRows(obsList);
        onApplyCustomStream(obsList, `Custom JSON Stream (${obsList.length} Queries)`);
      } catch (err: any) {
        setJsonError(`Invalid JSON format: ${err.message}`);
        return;
      }
    } else {
      if (customRows.length < 5) {
        alert('DeepDNS requires at least 5 observations.');
        return;
      }
      onApplyCustomStream(customRows, `Custom Form Stream (${customRows.length} Queries)`);
    }
  };

  return (
    <div className="card-panel" style={{ gap: '16px', background: 'var(--bg-elevated)' }}>
      {/* Header Toggle */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px' }}>
        <div>
          <span className="text-label" style={{ color: 'var(--accent-cyan)' }}>
            Custom DNS Stream & Query Editor
          </span>
          <div style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
            Inspect, edit, or customize all fields (Domain Name, Timestamps, Upper, Lower, Numeric, Entropy)
          </div>
        </div>

        <button
          onClick={() => setIsOpen(!isOpen)}
          className="btn-precision"
          style={{ fontSize: '12px', padding: '6px 14px', color: isOpen ? 'var(--accent-amber)' : 'var(--text-primary)' }}
        >
          {isOpen ? '▲ Hide Custom Stream Editor' : '▼ Open Custom Stream Editor'}
        </button>
      </div>

      {isOpen && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', marginTop: '4px' }}>
          {/* Generator Controls */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              flexWrap: 'wrap',
              gap: '12px',
              background: 'var(--bg-primary)',
              padding: '10px 14px',
              borderRadius: '4px',
              border: '1px solid var(--border-subtle)',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span className="text-label" style={{ color: 'var(--accent-blue)', fontSize: '11px' }}>
                Realistic Generator:
              </span>
              <label style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>Length:</label>
              <input
                type="number"
                min={5}
                max={30}
                value={randomCount}
                onChange={(e) => setRandomCount(Math.min(30, Math.max(5, parseInt(e.target.value) || 5)))}
                style={{
                  width: '55px',
                  background: 'var(--bg-elevated)',
                  border: '1px solid var(--border-medium)',
                  color: 'var(--text-primary)',
                  padding: '3px 6px',
                  fontSize: '12px',
                  borderRadius: '3px',
                  fontFamily: 'var(--font-serif)',
                }}
              />
              <button
                onClick={() => handleGenerateRealisticRandom('attack')}
                className="btn-precision"
                style={{ fontSize: '11px', padding: '4px 10px', color: 'var(--accent-crimson)' }}
              >
                🎲 Generate Realistic Attack ({randomCount})
              </button>
              <button
                onClick={() => handleGenerateRealisticRandom('benign')}
                className="btn-precision"
                style={{ fontSize: '11px', padding: '4px 10px', color: 'var(--accent-cyan)' }}
              >
                🎲 Generate Realistic Benign ({randomCount})
              </button>
            </div>

            <div style={{ display: 'flex', gap: '6px' }}>
              <button
                onClick={() => handleLoadTemplate('attack')}
                className="btn-precision"
                style={{ fontSize: '11px', padding: '4px 8px' }}
              >
                Attack Preset (10)
              </button>
              <button
                onClick={() => handleLoadTemplate('benign')}
                className="btn-precision"
                style={{ fontSize: '11px', padding: '4px 8px' }}
              >
                Benign Preset (10)
              </button>
            </div>
          </div>

          {/* Sub-nav */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px' }}>
            <div style={{ display: 'flex', gap: '6px' }}>
              <button
                onClick={() => setBuilderTab('form')}
                className="btn-precision"
                style={{
                  fontSize: '12px',
                  padding: '4px 10px',
                  background: builderTab === 'form' ? 'var(--bg-subtle)' : 'transparent',
                  color: builderTab === 'form' ? 'var(--accent-blue)' : 'var(--text-muted)',
                  fontWeight: builderTab === 'form' ? 700 : 500,
                }}
              >
                Full Multi-Field Table ({customRows.length} Queries)
              </button>
              <button
                onClick={() => setBuilderTab('json')}
                className="btn-precision"
                style={{
                  fontSize: '12px',
                  padding: '4px 10px',
                  background: builderTab === 'json' ? 'var(--bg-subtle)' : 'transparent',
                  color: builderTab === 'json' ? 'var(--accent-blue)' : 'var(--text-muted)',
                  fontWeight: builderTab === 'json' ? 700 : 500,
                }}
              >
                Raw JSON Editor
              </button>
            </div>

            {builderTab === 'form' && customRows.length < 30 && (
              <button
                onClick={handleAddRow}
                className="btn-precision btn-primary"
                style={{ fontSize: '11px', padding: '4px 10px', fontWeight: 700 }}
              >
                + Add Query Row
              </button>
            )}
          </div>

          {/* Tab 1: Comprehensive Multi-Field Table Form */}
          {builderTab === 'form' ? (
            <div style={{ maxHeight: '360px', overflowX: 'auto', overflowY: 'auto', border: '1px solid var(--border-subtle)', borderRadius: '3px' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px', fontFamily: 'var(--font-serif)', minWidth: '950px' }}>
                <thead>
                  <tr style={{ background: 'var(--bg-primary)', borderBottom: '1px solid var(--border-medium)', color: 'var(--text-muted)' }}>
                    <th style={{ padding: '6px 8px', width: '35px' }}>
                      <div className="tooltip-wrapper">
                        <span className="term-hover">#</span>
                        <div className="tooltip-box">Sequential query index within the active stream flow (1 to 30).</div>
                      </div>
                    </th>
                    <th style={{ padding: '6px 8px', width: '140px' }}>
                      <div className="tooltip-wrapper">
                        <span className="term-hover">Timestamp</span>
                        <div className="tooltip-box">
                          <strong style={{ color: 'var(--accent-blue)', display: 'block', marginBottom: '3px' }}>Packet Arrival Timestamp</strong>
                          Used to compute causal inter-arrival gap &Delta;t = log(1 + &delta;t) between consecutive queries for temporal burst tracking in the GRU.
                        </div>
                      </div>
                    </th>
                    <th style={{ padding: '6px 8px', minWidth: '220px' }}>
                      <div className="tooltip-wrapper">
                        <span className="term-hover">Domain Name / FQDN</span>
                        <div className="tooltip-box tooltip-wide">
                          <strong style={{ color: 'var(--accent-cyan)', display: 'block', marginBottom: '3px' }}>Fully Qualified Domain Name</strong>
                          Fed into the Character-CNN Tokenizer (ASCII vocabulary V=45, length L=128) to inspect subdomain text orthography, Base64/Hex encoding, and chunking patterns.
                        </div>
                      </div>
                    </th>
                    <th style={{ padding: '6px 8px', width: '60px' }}>
                      <div className="tooltip-wrapper">
                        <span className="term-hover">Len</span>
                        <div className="tooltip-box">Total character string length of the FQDN query. Tunneling payloads typically inflate overall query length.</div>
                      </div>
                    </th>
                    <th style={{ padding: '6px 8px', width: '60px' }}>
                      <div className="tooltip-wrapper">
                        <span className="term-hover">SubLen</span>
                        <div className="tooltip-box">Length of the leftmost subdomain label carrying the exfiltrated payload chunk.</div>
                      </div>
                    </th>
                    <th style={{ padding: '6px 8px', width: '55px' }}>
                      <div className="tooltip-wrapper">
                        <span className="term-hover">Upper</span>
                        <div className="tooltip-box">Count of uppercase characters [A-Z]. Often elevated in Base64 encoded tunneling.</div>
                      </div>
                    </th>
                    <th style={{ padding: '6px 8px', width: '55px' }}>
                      <div className="tooltip-wrapper">
                        <span className="term-hover">Lower</span>
                        <div className="tooltip-box">Count of lowercase characters [a-z] in the domain name.</div>
                      </div>
                    </th>
                    <th style={{ padding: '6px 8px', width: '55px' }}>
                      <div className="tooltip-wrapper">
                        <span className="term-hover">Numeric</span>
                        <div className="tooltip-box">Count of numeric digits [0-9]. Elevated in hexadecimal, chunked, and timestamped payloads.</div>
                      </div>
                    </th>
                    <th style={{ padding: '6px 8px', width: '55px' }}>
                      <div className="tooltip-wrapper">
                        <span className="term-hover">Special</span>
                        <div className="tooltip-box">Count of special characters (dots, hyphens, underscores) delimiting domain and chunk boundaries.</div>
                      </div>
                    </th>
                    <th style={{ padding: '6px 8px', width: '65px' }}>
                      <div className="tooltip-wrapper">
                        <span className="term-hover">Entropy</span>
                        <div className="tooltip-box tooltip-wide">
                          <strong style={{ color: 'var(--accent-amber)', display: 'block', marginBottom: '3px' }}>Shannon Character Entropy</strong>
                          H(X) = -&sum; p(x) log<sub>2</sub> p(x). Measures character unpredictability; values &gt; 2.5 indicate encrypted/compressed exfiltration.
                        </div>
                      </div>
                    </th>
                    <th style={{ padding: '6px 8px', width: '55px' }}>
                      <div className="tooltip-wrapper">
                        <span className="term-hover">Labels</span>
                        <div className="tooltip-box">Count of dot-separated domain labels (e.g. chunk.data.tunnel.net = 4 labels).</div>
                      </div>
                    </th>
                    <th style={{ padding: '6px 8px', width: '45px' }}>Act</th>
                  </tr>
                </thead>
                <tbody>
                  {customRows.map((row, idx) => (
                    <tr key={idx} style={{ borderBottom: '1px solid rgba(31, 36, 50, 0.5)' }}>
                      <td style={{ padding: '4px 8px', color: 'var(--text-dim)' }}>{idx + 1}</td>
                      <td style={{ padding: '4px 8px' }}>
                        <input
                          type="text"
                          value={row.timestamp}
                          onChange={(e) => handleUpdateTimestamp(idx, e.target.value)}
                          style={{ width: '100%', background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)', color: 'var(--text-secondary)', padding: '3px 4px', fontSize: '11px', fontFamily: 'monospace' }}
                        />
                      </td>
                      <td style={{ padding: '4px 8px' }}>
                        <input
                          type="text"
                          value={row.domain_name}
                          onChange={(e) => handleUpdateDomain(idx, e.target.value)}
                          style={{ width: '100%', background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)', color: 'var(--text-primary)', padding: '3px 6px', fontSize: '12px', fontFamily: 'monospace' }}
                        />
                      </td>
                      <td style={{ padding: '4px 8px' }}>
                        <input
                          type="number"
                          value={row.fqdn_count ?? row.domain_name.length}
                          onChange={(e) => handleUpdateNumericField(idx, 'fqdn_count', e.target.value)}
                          style={{ width: '100%', background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)', color: 'var(--text-primary)', padding: '3px 4px', fontSize: '11px' }}
                        />
                      </td>
                      <td style={{ padding: '4px 8px' }}>
                        <input
                          type="number"
                          value={row.subdomain_length ?? 0}
                          onChange={(e) => handleUpdateNumericField(idx, 'subdomain_length', e.target.value)}
                          style={{ width: '100%', background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)', color: 'var(--text-primary)', padding: '3px 4px', fontSize: '11px' }}
                        />
                      </td>
                      <td style={{ padding: '4px 8px' }}>
                        <input
                          type="number"
                          value={row.upper ?? 0}
                          onChange={(e) => handleUpdateNumericField(idx, 'upper', e.target.value)}
                          style={{ width: '100%', background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)', color: 'var(--text-primary)', padding: '3px 4px', fontSize: '11px' }}
                        />
                      </td>
                      <td style={{ padding: '4px 8px' }}>
                        <input
                          type="number"
                          value={row.lower ?? 0}
                          onChange={(e) => handleUpdateNumericField(idx, 'lower', e.target.value)}
                          style={{ width: '100%', background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)', color: 'var(--text-primary)', padding: '3px 4px', fontSize: '11px' }}
                        />
                      </td>
                      <td style={{ padding: '4px 8px' }}>
                        <input
                          type="number"
                          value={row.numeric ?? 0}
                          onChange={(e) => handleUpdateNumericField(idx, 'numeric', e.target.value)}
                          style={{ width: '100%', background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)', color: 'var(--text-primary)', padding: '3px 4px', fontSize: '11px' }}
                        />
                      </td>
                      <td style={{ padding: '4px 8px' }}>
                        <input
                          type="number"
                          value={row.special ?? 0}
                          onChange={(e) => handleUpdateNumericField(idx, 'special', e.target.value)}
                          style={{ width: '100%', background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)', color: 'var(--text-primary)', padding: '3px 4px', fontSize: '11px' }}
                        />
                      </td>
                      <td style={{ padding: '4px 8px' }}>
                        <input
                          type="number"
                          step={0.01}
                          value={row.entropy ?? 0}
                          onChange={(e) => handleUpdateNumericField(idx, 'entropy', e.target.value)}
                          style={{ width: '100%', background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)', color: 'var(--text-primary)', padding: '3px 4px', fontSize: '11px' }}
                        />
                      </td>
                      <td style={{ padding: '4px 8px' }}>
                        <input
                          type="number"
                          value={row.labels ?? 0}
                          onChange={(e) => handleUpdateNumericField(idx, 'labels', e.target.value)}
                          style={{ width: '100%', background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)', color: 'var(--text-primary)', padding: '3px 4px', fontSize: '11px' }}
                        />
                      </td>
                      <td style={{ padding: '4px 8px' }}>
                        <button
                          onClick={() => handleRemoveRow(idx)}
                          style={{ background: 'none', border: 'none', color: 'var(--accent-crimson)', cursor: 'pointer', fontSize: '13px' }}
                          title="Delete observation"
                        >
                          ✕
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            /* Tab 2: Raw JSON Editor */
            <div>
              <textarea
                value={rawJsonText}
                onChange={(e) => {
                  setRawJsonText(e.target.value);
                  setJsonError(null);
                }}
                style={{
                  width: '100%',
                  height: '220px',
                  background: 'var(--bg-primary)',
                  border: `1px solid ${jsonError ? 'var(--accent-crimson)' : 'var(--border-subtle)'}`,
                  color: 'var(--accent-blue)',
                  fontFamily: 'monospace',
                  fontSize: '12px',
                  padding: '12px',
                  borderRadius: '3px',
                }}
              />
              {jsonError && (
                <div style={{ color: 'var(--accent-crimson)', fontSize: '12px', marginTop: '4px' }}>
                  {jsonError}
                </div>
              )}
            </div>
          )}

          {/* Action Button */}
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px', alignItems: 'center' }}>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
              Sequence size: {customRows.length} queries
            </span>
            <button
              onClick={handleApply}
              className="btn-precision btn-cyan"
              style={{ padding: '8px 20px', fontWeight: 700 }}
            >
              Apply & Load Custom Stream into DeepDNS
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
