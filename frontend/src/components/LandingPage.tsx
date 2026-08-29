import React, { useState } from 'react';
import { ArrowRight, Cpu, Network, FastForward, Calculator, Award, ChevronDown, ChevronUp, Layers, Database } from 'lucide-react';

interface LandingPageProps {
  onNavigateToInstrument: () => void;
}

export const LandingPage: React.FC<LandingPageProps> = ({ onNavigateToInstrument }) => {
  const [showFullNovelty, setShowFullNovelty] = useState<boolean>(false);

  const causalFeatures = [
    { name: 'delta_t', desc: 'Causal log inter-arrival time between query i and i-1', formula: 'Δt_i = log(1 + (t_i - t_{i-1}))', role: 'Temporal burst cadence (GRU recurrence)' },
    { name: 'FQDN_count', desc: 'Total character length of the complete domain string', formula: 'Length(FQDN)', role: 'Payload inflation indicator' },
    { name: 'subdomain_length', desc: 'Character length of the leftmost subdomain label', formula: 'Length(Subdomain)', role: 'Payload chunk capacity' },
    { name: 'upper', desc: 'Count of uppercase ASCII letters [A-Z]', formula: 'Count(c ∈ [A-Z])', role: 'Base64/Base32 encoding detection' },
    { name: 'lower', desc: 'Count of lowercase ASCII letters [a-z]', formula: 'Count(c ∈ [a-z])', role: 'Standard domain text balance' },
    { name: 'numeric', desc: 'Count of digit characters [0-9]', formula: 'Count(c ∈ [0-9])', role: 'Hex/Base16 chunking and sequence indexing' },
    { name: 'entropy', desc: 'Shannon Character Entropy of the domain string', formula: 'H(X) = -Σ p(x) log₂(p(x))', role: 'Randomness of compressed/encrypted payloads' },
    { name: 'special', desc: 'Count of delimiter characters (dots, hyphens, underscores)', formula: 'Count(c ∈ {., -, _})', role: 'Label fragmentation count' },
    { name: 'labels', desc: 'Total count of dot-delimited labels in the FQDN', formula: 'Count(dots) + 1', role: 'Domain hierarchy depth' },
    { name: 'labels_max', desc: 'Maximum character length among all individual labels', formula: 'max_{l ∈ Labels} Length(l)', role: 'Single-label exfiltration chunk size' },
    { name: 'labels_average', desc: 'Mean character length across all domain labels', formula: 'FQDN_count / labels', role: 'Structural text density' },
    { name: 'subdomain', desc: 'Binary flag indicating presence of a non-empty subdomain', formula: '1 if SubLen > 0 else 0', role: 'Presence of hierarchical encapsulation' },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '36px', maxWidth: '1280px', margin: '0 auto' }}>
      {/* Hero Section */}
      <div className="card-panel-elevated" style={{ padding: '48px 40px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
        <div className="text-label" style={{ color: 'var(--accent-blue)', fontSize: '13px' }}>
          SCIENTIFIC OVERVIEW & NOVELTY
        </div>
        <h1 style={{ fontSize: '38px', lineHeight: 1.15, fontWeight: 700, margin: 0 }}>
          DeepDNS: Sequential Multi-View DNS Tunneling Detection with Adaptive Evidence Control
        </h1>
        <p style={{ fontSize: '16px', lineHeight: 1.6, color: 'var(--text-secondary)', maxWidth: '960px', margin: 0 }}>
          Modern data exfiltration attacks exploit the Domain Name System (DNS) by encoding stolen data into sub-domain labels and packet inter-arrival patterns. DeepDNS is an enterprise detection instrument that combines <strong>Temporal GRU sequence modeling</strong> and <strong>Character-level Convolutional feature extraction</strong> with an <strong>Adaptive Evidence Controller (AEC)</strong> to stop exfiltration attacks in as few as 5 to 10 queries, saving 50% to 83% of network query volume.
        </p>

        <div style={{ display: 'flex', gap: '14px', marginTop: '12px', alignItems: 'center' }}>
          <button
            onClick={onNavigateToInstrument}
            className="btn-precision btn-primary"
            style={{ padding: '12px 24px', fontSize: '15px', fontWeight: 700 }}
          >
            <span>Launch Live Detection</span>
            <ArrowRight size={16} />
          </button>
        </div>
      </div>

      {/* Core Scientific Novelty Card */}
      <div className="card-panel" style={{ padding: '32px', gap: '16px', borderLeft: '3px solid var(--accent-cyan)' }}>
        <h2 style={{ fontSize: '22px', margin: 0 }}>Core Scientific Novelty of DeepDNS</h2>

        <p style={{ fontSize: '14.5px', lineHeight: 1.7, color: 'var(--text-primary)', margin: 0 }}>
          The core novelty of DeepDNS is <strong>adaptive evidence-based DNS detection</strong>. Rather than always waiting for a fixed number of DNS queries, DeepDNS continuously combines temporal behavioral and lexical domain evidence to estimate attack probability, then uses an <strong>Adaptive Evidence Controller (AEC)</strong> to determine whether enough evidence exists to stop early or whether additional observations are required. Decisions are evaluated sequentially at <strong>K &isin; &#123;5, 10, 15, 20, 25, 30&#125;</strong> against calibrated attack and benign thresholds, producing the earliest sufficient stopping point while maintaining a bounded maximum observation horizon. The resulting stopping horizon directly translates into measurable query and telemetry savings, making DeepDNS not only a detection system, but an <strong>evidence-efficient sequential detection framework</strong>.
        </p>

        {/* Expandable Deep Technical Novelty */}
        <div>
          <button
            onClick={() => setShowFullNovelty(!showFullNovelty)}
            className="btn-precision"
            style={{ fontSize: '12px', padding: '6px 14px', color: 'var(--accent-cyan)' }}
          >
            {showFullNovelty ? (
              <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <ChevronUp size={14} /> Hide Formal Mathematical Formulation
              </span>
            ) : (
              <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                <ChevronDown size={14} /> Read Comprehensive Research Novelty & Formulation
              </span>
            )}
          </button>

          {showFullNovelty && (
            <div
              style={{
                marginTop: '14px',
                background: 'var(--bg-elevated)',
                border: '1px solid var(--border-medium)',
                borderRadius: '4px',
                padding: '20px',
                fontSize: '13.5px',
                lineHeight: 1.7,
                color: 'var(--text-secondary)',
              }}
            >
              <strong style={{ color: 'var(--text-primary)', display: 'block', marginBottom: '8px' }}>
                Formal Sequential Decision Process:
              </strong>
              DeepDNS introduces an adaptive, evidence-driven approach to DNS exfiltration detection that determines when sufficient evidence has been accumulated to make a reliable decision, rather than relying on a fixed observation window. The system jointly models two complementary views of DNS traffic: temporal-behavioral evidence captured through causal sequential features and a Temporal GRU, and lexical evidence extracted from domain-name character patterns using a Character-CNN. These representations are fused to produce sequential attack probabilities as DNS queries arrive. An Adaptive Evidence Controller (AEC) then evaluates the accumulated evidence at predefined horizons <em>K &isin; &#123;5, 10, 15, 20, 25, 30&#125;</em> against calibrated attack and benign thresholds, allowing the system to terminate observation as soon as sufficient evidence is available while retaining a bounded maximum horizon. This transforms detection from a fixed-length classification task into an adaptive sequential decision process, with the resulting stopping horizon directly quantifying how many DNS observations can be avoided. Thus, DeepDNS simultaneously targets detection reliability and evidence-efficient telemetry consumption, with query savings emerging as a measurable consequence of the adaptive decision mechanism.
            </div>
          )}
        </div>
      </div>

      {/* Benchmark Dataset Section */}
      <div className="card-panel" style={{ padding: '32px', gap: '18px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Database size={22} color="var(--accent-blue)" />
          <div>
            <div className="text-label" style={{ color: 'var(--accent-blue)', fontSize: '13px' }}>
              DATASET & EXPERIMENTAL CORPUS
            </div>
            <h2 style={{ fontSize: '22px', margin: '2px 0 0 0' }}>Canadian Institute for Cybersecurity (CIC-Bell) Dataset</h2>
          </div>
        </div>

        <p style={{ fontSize: '14px', color: 'var(--text-secondary)', lineHeight: 1.65, margin: 0 }}>
          DeepDNS was developed and scientifically evaluated using the authentic <strong>CIC-Bell DNS Exfiltration Dataset (2021)</strong>, created jointly by the Canadian Institute for Cybersecurity (CIC) and Bell Canada. Synthetic datasets (like generic "DNS Threat" rule-lists) were explicitly rejected because they lack realistic packet inter-arrival jitter, natural resolver delays, and genuine CDN subdomain fragmentation.
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '14px' }}>
          <div className="card-panel-elevated" style={{ padding: '18px', gap: '8px' }}>
            <span className="text-label" style={{ color: 'var(--accent-crimson)', fontWeight: 700 }}>
              Attack Modalities & Tunneling Tools
            </span>
            <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.5, margin: 0 }}>
              Includes <strong>Light & Heavy Text exfiltration</strong>, <strong>Audio streams</strong>, <strong>Video exfiltration</strong>, <strong>Compressed binary archives</strong>, and active <strong>DNSCat / Iodine</strong> tunnels executing real command-and-control exfiltration sessions.
            </p>
          </div>

          <div className="card-panel-elevated" style={{ padding: '18px', gap: '8px' }}>
            <span className="text-label" style={{ color: 'var(--accent-cyan)', fontWeight: 700 }}>
              Benign Enterprise Flows
            </span>
            <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.5, margin: 0 }}>
              Captured across live corporate networks containing normal user web browsing, cloud API calls, high-volume CDN traffic (Akamai, Cloudflare, AWS), and background OS telemetry.
            </p>
          </div>

          <div className="card-panel-elevated" style={{ padding: '18px', gap: '8px' }}>
            <span className="text-label" style={{ color: 'var(--accent-amber)', fontWeight: 700 }}>
              Data Preprocessing & Causal Isolation
            </span>
            <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.5, margin: 0 }}>
              Raw PCAPs were parsed to extract packet-level arrival timestamps and domain names. All stateful future aggregates were pruned, leaving 12 strictly causal features + raw domain strings.
            </p>
          </div>
        </div>
      </div>

      {/* The 12 Causal Behavioral Features Table */}
      <div className="card-panel" style={{ padding: '32px', gap: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Layers size={22} color="var(--accent-blue)" />
          <div>
            <div className="text-label" style={{ color: 'var(--accent-blue)', fontSize: '13px' }}>
              CAUSAL FEATURE SPECIFICATION
            </div>
            <h2 style={{ fontSize: '22px', margin: '2px 0 0 0' }}>The 12 Behavioral Traffic Metrics (Temporal GRU View)</h2>
          </div>
        </div>

        <p style={{ fontSize: '13.5px', color: 'var(--text-secondary)', lineHeight: 1.6, margin: 0 }}>
          To ensure <strong>zero future data leakage</strong> and sub-millisecond computability at live DNS resolvers, DeepDNS strictly eliminated all non-causal session totals and post-flow metrics. The Temporal GRU models the following 12 causal metrics normalized by training scaler parameters (<code>feature_scaler.json</code>):
        </p>

        <div style={{ width: '100%', overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '13px', fontFamily: 'var(--font-serif)' }}>
            <thead>
              <tr style={{ background: 'var(--bg-elevated)', borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)', fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                <th style={{ padding: '10px 12px', width: '40px' }}>#</th>
                <th style={{ padding: '10px 12px', width: '160px' }}>Feature Name</th>
                <th style={{ padding: '10px 12px', width: '220px' }}>Mathematical Definition</th>
                <th style={{ padding: '10px 12px', minWidth: '240px' }}>Description & Intuition</th>
                <th style={{ padding: '10px 12px' }}>Role in Neural Network</th>
              </tr>
            </thead>
            <tbody>
              {causalFeatures.map((f, idx) => (
                <tr key={idx} style={{ borderBottom: '1px solid rgba(31, 36, 50, 0.4)' }}>
                  <td style={{ padding: '9px 12px', color: 'var(--text-dim)' }}>{idx + 1}</td>
                  <td style={{ padding: '9px 12px', fontWeight: 600, color: 'var(--accent-blue)', fontFamily: 'monospace' }}>{f.name}</td>
                  <td style={{ padding: '9px 12px', color: 'var(--accent-cyan)', fontFamily: 'monospace', fontSize: '12px' }}>{f.formula}</td>
                  <td style={{ padding: '9px 12px', color: 'var(--text-secondary)' }}>{f.desc}</td>
                  <td style={{ padding: '9px 12px', color: 'var(--text-primary)' }}>{f.role}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Developed Models & Deep Architectural Specification */}
      <div className="card-panel" style={{ padding: '32px', gap: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Layers size={22} color="var(--accent-cyan)" />
          <div>
            <div className="text-label" style={{ color: 'var(--accent-cyan)', fontSize: '13px' }}>
              NEURAL ARCHITECTURE & PIPELINE SPECIFICATION
            </div>
            <h2 style={{ fontSize: '22px', margin: '2px 0 0 0' }}>Core Deep Learning Models & Decision Mechanisms</h2>
          </div>
        </div>

        {/* Section A: The 2 Deep Learning Models */}
        <div>
          <span className="text-label" style={{ color: 'var(--accent-blue)', display: 'block', marginBottom: '10px' }}>
            Primary Deep Learning Encoders (Trained Models)
          </span>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '16px' }}>
            {/* GRU */}
            <div className="card-panel-elevated" style={{ padding: '22px', gap: '10px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Network size={20} color="var(--accent-blue)" />
                <h3 style={{ fontSize: '17px', margin: 0 }}>1. Behavioral Temporal GRU Model</h3>
              </div>
              <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.55, margin: 0 }}>
                <strong>Model Type:</strong> Recurrent Neural Network (2-layer GRU, hidden size = 64, dropout = 0.2, bidirectional = False).<br />
                <strong>Input Tensor:</strong> Causal sequential feature matrix X<sub>beh</sub> &isin; R<sup>K &times; 12</sup>.<br />
                <strong>Function:</strong> Models inter-arrival cadence (&Delta;t) and traffic bursts chronologically with strictly zero future leakage, emitting latent embedding z<sub>beh</sub> = h<sub>K</sub> &isin; R<sup>64</sup>.
              </p>
            </div>

            {/* Char-CNN */}
            <div className="card-panel-elevated" style={{ padding: '22px', gap: '10px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Cpu size={20} color="var(--accent-cyan)" />
                <h3 style={{ fontSize: '17px', margin: 0 }}>2. Lexical Character-CNN Model</h3>
              </div>
              <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.55, margin: 0 }}>
                <strong>Model Type:</strong> Convolutional Neural Network (1D Filter Banks with kernels 3, 4, 5 and 64 filters each).<br />
                <strong>Input Tensor:</strong> ASCII character token matrix X<sub>lex</sub> &isin; R<sup>K &times; 128</sup> (Vocabulary V=45).<br />
                <strong>Function:</strong> Extracts sub-word orthographic patterns, Base64/Hex chunks, and character entropy artifacts into latent embedding z<sub>lex</sub> &isin; R<sup>128</sup>.
              </p>
            </div>
          </div>
        </div>

        {/* Section B: Integration Layer & Decision Mechanism */}
        <div style={{ marginTop: '8px' }}>
          <span className="text-label" style={{ color: 'var(--accent-amber)', display: 'block', marginBottom: '10px' }}>
            Integration Layer & Sequential Decision Mechanism
          </span>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '16px' }}>
            {/* Fusion MLP Head */}
            <div className="card-panel-elevated" style={{ padding: '22px', gap: '10px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Layers size={20} color="var(--text-primary)" />
                <h3 style={{ fontSize: '17px', margin: 0 }}>Integration Layer: Dual-View MLP Fusion Head</h3>
              </div>
              <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.55, margin: 0 }}>
                <strong>Layer Type:</strong> Linear Projection [z<sub>beh</sub> || z<sub>lex</sub>] (192 &rarr; 64) + ReLU + Dropout (0.2) + Linear Output (64 &rarr; 2) + Softmax.<br />
                <strong>Function:</strong> Jointly fuses behavioral burst dynamics and domain text representations into unified vector z<sub>fuse</sub> &isin; R<sup>64</sup>, computing step posterior probability p<sub>K</sub> = P(Attack | x<sub>1:K</sub>).
              </p>
            </div>

            {/* AEC */}
            <div className="card-panel-elevated" style={{ padding: '22px', gap: '10px', borderColor: 'rgba(245, 158, 11, 0.4)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <FastForward size={20} color="var(--accent-amber)" />
                <h3 style={{ fontSize: '17px', margin: 0 }}>Decision Algorithm: Adaptive Evidence Controller (AEC)</h3>
              </div>
              <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.55, margin: 0 }}>
                <strong>Algorithm Type:</strong> Dynamic Sequential Evidence Stopping Rule.<br />
                <strong>Function:</strong> Evaluates p<sub>K</sub> at discrete checkpoints K &isin; &#123;5, 10, 15, 20, 25, 30&#125;. Emits immediate terminal stopping when p<sub>K</sub> &ge; &tau;<sub>atk</sub> (Attack) or p<sub>K</sub> &le; &tau;<sub>ben</sub> (Benign), with a bounded 50% fallback at K=30.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* 4 Operating Modes Explained */}
      <div className="card-panel" style={{ padding: '32px', gap: '20px' }}>
        <div>
          <div className="text-label" style={{ color: 'var(--accent-blue)', fontSize: '13px' }}>
            OPERATIONAL CONFIGURATIONS & EVALUATION SPLITS
          </div>
          <h2 style={{ fontSize: '22px', margin: '2px 0 0 0' }}>Understanding the 4 Operating Modes</h2>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px' }}>
          {/* Mode 1 */}
          <div className="card-panel-elevated" style={{ padding: '20px', gap: '10px' }}>
            <span className="text-label" style={{ color: 'var(--accent-blue)', fontWeight: 700 }}>
              1. In-Distribution Dual-View
            </span>
            <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.55, margin: 0 }}>
              The primary enterprise operational mode. Combines the Temporal GRU (12 causal behavioral metrics) and the Lexical Char-CNN (domain text) to detect known tunneling families (Text, Audio, DNSCat, Iodine) with calibrated bounds (&tau;<sub>atk</sub>=0.95, &tau;<sub>ben</sub>=0.15).
            </p>
          </div>

          {/* Mode 2 */}
          <div className="card-panel-elevated" style={{ padding: '20px', gap: '10px', borderColor: 'rgba(245, 158, 11, 0.4)' }}>
            <span className="text-label" style={{ color: 'var(--accent-amber)', fontWeight: 700 }}>
              2. Out-of-Distribution (OOD / LOMO)
            </span>
            <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.55, margin: 0 }}>
              Evaluates Leave-One-Modality-Out generalization. Entire attack families (Video exfiltration, Compressed archives, Executable payloads) were completely withheld during training to prove that DeepDNS reliably detects unseen, novel zero-shot tunneling attacks (&tau;<sub>atk</sub>=0.80, &tau;<sub>ben</sub>=0.01).
            </p>
          </div>

          {/* Mode 3 */}
          <div className="card-panel-elevated" style={{ padding: '20px', gap: '10px' }}>
            <span className="text-label" style={{ color: 'var(--text-primary)', fontWeight: 700 }}>
              3. Behavioral-Only (GRU Ablation)
            </span>
            <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.55, margin: 0 }}>
              Isolates the network traffic timing branch. Evaluates exfiltration purely from inter-arrival burst patterns and causal statistics (&Delta;t, entropy, length, digit counts) without inspecting the domain string characters.
            </p>
          </div>

          {/* Mode 4 */}
          <div className="card-panel-elevated" style={{ padding: '20px', gap: '10px' }}>
            <span className="text-label" style={{ color: 'var(--text-primary)', fontWeight: 700 }}>
              4. Lexical-Only (CNN Ablation)
            </span>
            <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.55, margin: 0 }}>
              Isolates domain text orthography. Evaluates exfiltration purely from character-level subdomain randomness using 1D convolutions, proving why text alone suffers from high false alarms (38.26% FPR) on legitimate CDNs without temporal context.
            </p>
          </div>
        </div>
      </div>

      {/* Complete Master Scientific Ablation Matrix */}
      <div className="card-panel" style={{ padding: '32px', gap: '20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px' }}>
          <div>
            <div className="text-label" style={{ color: 'var(--accent-cyan)', fontSize: '13px' }}>
              EMPIRICAL BENCHMARKS (N_ID = 13,084, N_OOD = 20,683)
            </div>
            <h2 style={{ fontSize: '24px', margin: '4px 0 0 0' }}>Master Architectural & Ablation Performance Matrix</h2>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-muted)', fontSize: '12px' }}>
            <Award size={16} color="var(--accent-cyan)" />
            <span>1,000-Iteration Bootstrap 95% Confidence Intervals</span>
          </div>
        </div>

        <div style={{ width: '100%', overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '13px', fontFamily: 'var(--font-serif)' }}>
            <thead>
              <tr style={{ background: 'var(--bg-elevated)', borderBottom: '1px solid var(--border-subtle)', color: 'var(--text-muted)', fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                <th style={{ padding: '10px 14px' }}>Model / Algorithm</th>
                <th style={{ padding: '10px 14px' }}>Evaluation Split</th>
                <th style={{ padding: '10px 14px' }}>Recall (Detection)</th>
                <th style={{ padding: '10px 14px' }}>False Alarm Rate (FPR)</th>
                <th style={{ padding: '10px 14px' }}>F1-Score</th>
                <th style={{ padding: '10px 14px' }}>Mean Horizon (K*)</th>
                <th style={{ padding: '10px 14px' }}>Query Savings</th>
              </tr>
            </thead>
            <tbody>
              {/* In-Distribution Rows */}
              <tr style={{ borderBottom: '1px solid rgba(31, 36, 50, 0.4)' }}>
                <td style={{ padding: '10px 14px', fontWeight: 600, color: 'var(--accent-blue)' }}>Dual-View Fixed K=5</td>
                <td style={{ padding: '10px 14px' }}>In-Distribution</td>
                <td style={{ padding: '10px 14px' }}>98.00% [97.58, 98.41]</td>
                <td style={{ padding: '10px 14px' }}>5.49%</td>
                <td style={{ padding: '10px 14px' }}>0.9350</td>
                <td style={{ padding: '10px 14px' }}>5.00</td>
                <td style={{ padding: '10px 14px', color: 'var(--accent-cyan)', fontWeight: 700 }}>83.3%</td>
              </tr>
              <tr style={{ borderBottom: '1px solid rgba(31, 36, 50, 0.4)' }}>
                <td style={{ padding: '10px 14px', fontWeight: 600, color: 'var(--accent-blue)' }}>Dual-View Fixed K=10</td>
                <td style={{ padding: '10px 14px' }}>In-Distribution</td>
                <td style={{ padding: '10px 14px' }}>99.52% [99.31, 99.72]</td>
                <td style={{ padding: '10px 14px' }}>1.37%</td>
                <td style={{ padding: '10px 14px' }}>0.9833</td>
                <td style={{ padding: '10px 14px' }}>10.00</td>
                <td style={{ padding: '10px 14px', color: 'var(--accent-cyan)', fontWeight: 700 }}>66.7%</td>
              </tr>
              <tr style={{ borderBottom: '1px solid rgba(31, 36, 50, 0.4)' }}>
                <td style={{ padding: '10px 14px', fontWeight: 600, color: 'var(--accent-blue)' }}>Dual-View Fixed K=15</td>
                <td style={{ padding: '10px 14px' }}>In-Distribution</td>
                <td style={{ padding: '10px 14px' }}>99.76% [99.60, 99.90]</td>
                <td style={{ padding: '10px 14px' }}>0.52%</td>
                <td style={{ padding: '10px 14px' }}>0.9934</td>
                <td style={{ padding: '10px 14px' }}>15.00</td>
                <td style={{ padding: '10px 14px', color: 'var(--accent-cyan)', fontWeight: 700 }}>50.0%</td>
              </tr>
              <tr style={{ borderBottom: '1px solid rgba(31, 36, 50, 0.4)' }}>
                <td style={{ padding: '10px 14px', fontWeight: 600, color: 'var(--accent-blue)' }}>Dual-View Fixed K=30</td>
                <td style={{ padding: '10px 14px' }}>In-Distribution</td>
                <td style={{ padding: '10px 14px' }}>99.52% [99.31, 99.71]</td>
                <td style={{ padding: '10px 14px' }}>0.23%</td>
                <td style={{ padding: '10px 14px' }}>0.9952</td>
                <td style={{ padding: '10px 14px' }}>30.00</td>
                <td style={{ padding: '10px 14px', color: 'var(--text-muted)' }}>0.0%</td>
              </tr>
              <tr style={{ borderBottom: '1px solid rgba(31, 36, 50, 0.4)' }}>
                <td style={{ padding: '10px 14px', fontWeight: 600, color: 'var(--text-primary)' }}>Behavioral-Only (GRU)</td>
                <td style={{ padding: '10px 14px' }}>In-Distribution</td>
                <td style={{ padding: '10px 14px' }}>98.88%</td>
                <td style={{ padding: '10px 14px' }}>0.10%</td>
                <td style={{ padding: '10px 14px' }}>0.9933</td>
                <td style={{ padding: '10px 14px' }}>30.00</td>
                <td style={{ padding: '10px 14px', color: 'var(--text-muted)' }}>0.0%</td>
              </tr>
              <tr style={{ borderBottom: '1px solid rgba(31, 36, 50, 0.4)' }}>
                <td style={{ padding: '10px 14px', fontWeight: 600, color: 'var(--text-primary)' }}>Lexical-Only (CNN)</td>
                <td style={{ padding: '10px 14px' }}>In-Distribution</td>
                <td style={{ padding: '10px 14px' }}>98.45%</td>
                <td style={{ padding: '10px 14px', color: 'var(--accent-crimson)' }}>38.26% (High False Alarms)</td>
                <td style={{ padding: '10px 14px' }}>0.7048</td>
                <td style={{ padding: '10px 14px' }}>30.00</td>
                <td style={{ padding: '10px 14px', color: 'var(--text-muted)' }}>0.0%</td>
              </tr>
              <tr style={{ borderBottom: '1px solid rgba(31, 36, 50, 0.6)', background: 'rgba(0, 229, 163, 0.06)' }}>
                <td style={{ padding: '10px 14px', fontWeight: 700, color: 'var(--accent-cyan)' }}>Dual-View AEC (Adaptive)</td>
                <td style={{ padding: '10px 14px', fontWeight: 600 }}>In-Distribution</td>
                <td style={{ padding: '10px 14px', fontWeight: 700 }}>98.17% [97.74, 98.56]</td>
                <td style={{ padding: '10px 14px', color: 'var(--accent-cyan)', fontWeight: 700 }}>0.12%</td>
                <td style={{ padding: '10px 14px', fontWeight: 700 }}>0.9894</td>
                <td style={{ padding: '10px 14px', fontWeight: 700, color: 'var(--accent-blue)' }}>13.13 queries</td>
                <td style={{ padding: '10px 14px', color: 'var(--accent-cyan)', fontWeight: 700, fontSize: '14px' }}>56.2% SAVED</td>
              </tr>
              <tr style={{ borderBottom: '1px solid rgba(31, 36, 50, 0.4)' }}>
                <td style={{ padding: '10px 14px', fontWeight: 600, color: 'var(--text-secondary)' }}>Dual-View CUSUM</td>
                <td style={{ padding: '10px 14px' }}>In-Distribution</td>
                <td style={{ padding: '10px 14px' }}>99.12%</td>
                <td style={{ padding: '10px 14px' }}>0.59%</td>
                <td style={{ padding: '10px 14px' }}>0.9894</td>
                <td style={{ padding: '10px 14px' }}>9.22</td>
                <td style={{ padding: '10px 14px', color: 'var(--accent-cyan)' }}>69.3%</td>
              </tr>

              {/* OOD Rows */}
              <tr style={{ borderBottom: '1px solid rgba(31, 36, 50, 0.4)', background: 'rgba(245, 158, 11, 0.04)' }}>
                <td style={{ padding: '10px 14px', fontWeight: 600, color: 'var(--accent-amber)' }}>Dual-View Fixed K=10</td>
                <td style={{ padding: '10px 14px', color: 'var(--accent-amber)' }}>Zero-Shot OOD (LOMO)</td>
                <td style={{ padding: '10px 14px' }}>94.58%</td>
                <td style={{ padding: '10px 14px' }}>0.81%</td>
                <td style={{ padding: '10px 14px' }}>0.9691</td>
                <td style={{ padding: '10px 14px' }}>10.00</td>
                <td style={{ padding: '10px 14px', color: 'var(--accent-cyan)' }}>66.7%</td>
              </tr>
              <tr style={{ borderBottom: '1px solid rgba(31, 36, 50, 0.4)', background: 'rgba(245, 158, 11, 0.04)' }}>
                <td style={{ padding: '10px 14px', fontWeight: 600, color: 'var(--accent-amber)' }}>Dual-View Fixed K=30</td>
                <td style={{ padding: '10px 14px', color: 'var(--accent-amber)' }}>Zero-Shot OOD (LOMO)</td>
                <td style={{ padding: '10px 14px' }}>99.44%</td>
                <td style={{ padding: '10px 14px' }}>0.20%</td>
                <td style={{ padding: '10px 14px' }}>0.9964</td>
                <td style={{ padding: '10px 14px' }}>30.00</td>
                <td style={{ padding: '10px 14px', color: 'var(--text-muted)' }}>0.0%</td>
              </tr>
              <tr style={{ borderBottom: '1px solid rgba(31, 36, 50, 0.6)', background: 'rgba(245, 158, 11, 0.1)' }}>
                <td style={{ padding: '10px 14px', fontWeight: 700, color: 'var(--accent-amber)' }}>Dual-View AEC (Adaptive OOD)</td>
                <td style={{ padding: '10px 14px', fontWeight: 700, color: 'var(--accent-amber)' }}>Zero-Shot OOD (LOMO)</td>
                <td style={{ padding: '10px 14px', fontWeight: 700, color: 'var(--accent-cyan)' }}>99.21% [99.05, 99.36]</td>
                <td style={{ padding: '10px 14px', fontWeight: 700 }}>0.51%</td>
                <td style={{ padding: '10px 14px', fontWeight: 700 }}>0.9941</td>
                <td style={{ padding: '10px 14px', fontWeight: 700, color: 'var(--accent-blue)' }}>10.30 queries</td>
                <td style={{ padding: '10px 14px', color: 'var(--accent-cyan)', fontWeight: 700, fontSize: '14px' }}>65.7% SAVED</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Mathematical Formulas & Rigorous Definitions */}
      <div className="card-panel" style={{ padding: '32px', gap: '20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Calculator size={22} color="var(--accent-blue)" />
          <h2 style={{ fontSize: '24px', margin: 0 }}>Mathematical Formulations & Scoring Metrics</h2>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '16px' }}>
          {/* Classification Formulas */}
          <div className="card-panel-elevated" style={{ padding: '20px', gap: '12px' }}>
            <span className="text-label" style={{ color: 'var(--accent-blue)' }}>Detection & Error Metrics</span>
            <div style={{ fontSize: '13px', lineHeight: 1.8, color: 'var(--text-secondary)' }}>
              <div><strong>Recall / True Positive Rate (TPR):</strong> <code>TPR = TP / (TP + FN)</code> (Sensitivity to exfiltration attacks).</div>
              <div><strong>False Positive Rate (FPR):</strong> <code>FPR = FP / (FP + TN)</code> (False alarms on benign traffic).</div>
              <div><strong>Precision (Positive Predictive Value):</strong> <code>PPV = TP / (TP + FP)</code>.</div>
              <div><strong>F1-Score:</strong> <code>F1 = 2 * (Precision * Recall) / (Precision + Recall)</code> (Harmonic mean).</div>
            </div>
          </div>

          {/* Feature Engineering Formulas */}
          <div className="card-panel-elevated" style={{ padding: '20px', gap: '12px' }}>
            <span className="text-label" style={{ color: 'var(--accent-cyan)' }}>Causal Feature & Entropy Formulas</span>
            <div style={{ fontSize: '13px', lineHeight: 1.8, color: 'var(--text-secondary)' }}>
              <div><strong>Log Causal Inter-Arrival Time:</strong> <code>&Delta;t = log(1 + (t_i - t_(i-1)))</code> where &Delta;t &ge; 0.</div>
              <div><strong>Shannon Character Entropy:</strong> <code>H(X) = -&sum; p(x) log2 p(x)</code> (Measures subdomain randomness).</div>
              <div><strong>Digit Ratio:</strong> <code>Ratio_numeric = Count(digits) / Length(FQDN)</code>.</div>
              <div><strong>Feature Scaling:</strong> <code>z = (x - &mu;_train) / &sigma;_train</code> (Strict training scaler isolation).</div>
            </div>
          </div>

          {/* Adaptive Stopping Rule */}
          <div className="card-panel-elevated" style={{ padding: '20px', gap: '12px' }}>
            <span className="text-label" style={{ color: 'var(--accent-amber)' }}>AEC Dynamic Stopping Decision Rule</span>
            <div style={{ fontSize: '13px', lineHeight: 1.8, color: 'var(--text-secondary)' }}>
              <div><strong>AEC Stopping Horizon (K*):</strong></div>
              <code style={{ display: 'block', background: 'var(--bg-primary)', padding: '8px', borderRadius: '3px', color: 'var(--accent-amber)', margin: '4px 0' }}>
                K* = min &#123; K &isin; [5,10,15,20,25,30] | P(Attack) &ge; &tau;_atk &or; P(Attack) &le; &tau;_ben &#125;
              </code>
              <div><strong>Query Volume Savings:</strong> <code>Savings % = ((30 - K*) / 30) * 100</code>.</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
