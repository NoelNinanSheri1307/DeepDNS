import { DNSObservation } from '../types';
import realSamplesData from './real_samples.json';

export const SAMPLE_ATTACK_1: DNSObservation[] = (realSamplesData as any).attack_1_light_text;
export const SAMPLE_ATTACK_2: DNSObservation[] = (realSamplesData as any).attack_2_heavy_audio;
export const SAMPLE_ATTACK_3: DNSObservation[] = (realSamplesData as any).attack_3_light_audio;

export const SAMPLE_BENIGN_1: DNSObservation[] = (realSamplesData as any).benign_1_normal;
export const SAMPLE_BENIGN_2: DNSObservation[] = (realSamplesData as any).benign_2_normal;
export const SAMPLE_BENIGN_3: DNSObservation[] = (realSamplesData as any).benign_3_heavy;

export const SAMPLE_OOD_1: DNSObservation[] = (realSamplesData as any).ood_1_video;
export const SAMPLE_OOD_2: DNSObservation[] = (realSamplesData as any).ood_2_compressed;
export const SAMPLE_OOD_3: DNSObservation[] = (realSamplesData as any).ood_3_exe;

// Default aliases for primary buttons
export const SAMPLE_ATTACK_STREAM = SAMPLE_ATTACK_1;
export const SAMPLE_BENIGN_STREAM = SAMPLE_BENIGN_1;
export const SAMPLE_OOD_ATTACK_STREAM = SAMPLE_OOD_1;
