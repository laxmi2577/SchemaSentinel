from __future__ import annotations

import hashlib
import struct
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple
from pydantic import BaseModel, ConfigDict, Field
from src.data_engine.ingestion import RawTextToSQLSample


class MinHashConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    num_perm: int = 128
    threshold: float = 0.85
    num_bands: int = 16
    shingle_k: int = 3
    seed: int = 42


class LeakageViolation(BaseModel):
    model_config = ConfigDict(frozen=True)

    train_sample_id: str
    benchmark_sample_id: str
    similarity: float
    train_question: str
    benchmark_question: str


class DeduplicationSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_records: int
    unique_records: int
    internal_duplicates_dropped: int
    quarantined_benchmark_leakages: int
    violations: List[LeakageViolation] = Field(default_factory=list)


class MinHashSignature:
    PRIME_61: int = (1 << 61) - 1

    def __init__(self, config: MinHashConfig) -> None:
        self.config = config
        self.num_perm = config.num_perm
        self.rows_per_band = config.num_perm // config.num_bands
        self._a_params, self._b_params = self._generate_hash_params(config.seed)

    def _generate_hash_params(self, seed: int) -> Tuple[List[int], List[int]]:
        a_vals: List[int] = []
        b_vals: List[int] = []
        current = seed
        for _ in range(self.num_perm):
            current = (current * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            a = (current % (self.PRIME_61 - 1)) + 1
            current = (current * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            b = current % self.PRIME_61
            a_vals.append(a)
            b_vals.append(b)
        return a_vals, b_vals

    def build_shingles(self, text: str) -> Set[int]:
        normalized = " ".join(text.lower().strip().split())
        words = normalized.split()
        shingle_hashes: Set[int] = set()

        if len(words) < self.config.shingle_k:
            raw_bytes = normalized.encode("utf-8")
            h_int = struct.unpack("<Q", hashlib.md5(raw_bytes).digest()[:8])[0]
            shingle_hashes.add(h_int)
            return shingle_hashes

        for i in range(len(words) - self.config.shingle_k + 1):
            shingle = " ".join(words[i : i + self.config.shingle_k])
            raw_bytes = shingle.encode("utf-8")
            h_int = struct.unpack("<Q", hashlib.md5(raw_bytes).digest()[:8])[0]
            shingle_hashes.add(h_int)

        return shingle_hashes

    def compute_signature(self, text: str) -> List[int]:
        shingles = self.build_shingles(text)
        if not shingles:
            return [0] * self.num_perm

        signature: List[int] = [self.PRIME_61] * self.num_perm
        for shingle_val in shingles:
            for i in range(self.num_perm):
                val = (self._a_params[i] * shingle_val + self._b_params[i]) % self.PRIME_61
                if val < signature[i]:
                    signature[i] = val
        return signature

    def estimate_jaccard(self, sig1: List[int], sig2: List[int]) -> float:
        if len(sig1) != self.num_perm or len(sig2) != self.num_perm:
            return 0.0
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / self.num_perm


class LSHIndex:
    def __init__(self, config: MinHashConfig) -> None:
        self.config = config
        self.num_bands = config.num_bands
        self.rows_per_band = config.num_perm // config.num_bands
        self.buckets: Dict[Tuple[int, int], List[str]] = {}
        self.sample_signatures: Dict[str, List[int]] = {}

    def insert(self, key: str, signature: List[int]) -> None:
        self.sample_signatures[key] = signature
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_slice = tuple(signature[start:end])
            bucket_key = (band_idx, hash(band_slice))
            if bucket_key not in self.buckets:
                self.buckets[bucket_key] = []
            self.buckets[bucket_key].append(key)

    def query_candidates(self, signature: List[int]) -> Set[str]:
        candidates: Set[str] = set()
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_slice = tuple(signature[start:end])
            bucket_key = (band_idx, hash(band_slice))
            if bucket_key in self.buckets:
                candidates.update(self.buckets[bucket_key])
        return candidates


class BenchmarkContaminationDetector:
    def __init__(self, config: Optional[MinHashConfig] = None) -> None:
        self.config = config or MinHashConfig()
        self.hasher = MinHashSignature(self.config)
        self.benchmark_index = LSHIndex(self.config)
        self.benchmark_samples: Dict[str, RawTextToSQLSample] = {}

    def register_benchmark_samples(
        self,
        samples: Iterable[RawTextToSQLSample],
    ) -> int:
        registered = 0
        for sample in samples:
            sig = self.hasher.compute_signature(sample.question)
            self.benchmark_index.insert(sample.sample_id, sig)
            self.benchmark_samples[sample.sample_id] = sample
            registered += 1
        return registered

    def check_leakage(
        self,
        train_sample: RawTextToSQLSample,
    ) -> Optional[LeakageViolation]:
        sig = self.hasher.compute_signature(train_sample.question)
        candidates = self.benchmark_index.query_candidates(sig)

        highest_sim = 0.0
        worst_candidate_id: Optional[str] = None

        for cand_id in candidates:
            cand_sig = self.benchmark_index.sample_signatures[cand_id]
            sim = self.hasher.estimate_jaccard(sig, cand_sig)
            if sim > highest_sim:
                highest_sim = sim
                worst_candidate_id = cand_id

        if worst_candidate_id and highest_sim >= self.config.threshold:
            bench_sample = self.benchmark_samples[worst_candidate_id]
            return LeakageViolation(
                train_sample_id=train_sample.sample_id,
                benchmark_sample_id=bench_sample.sample_id,
                similarity=round(highest_sim, 4),
                train_question=train_sample.question,
                benchmark_question=bench_sample.question,
            )
        return None

    def sanitize_training_stream(
        self,
        train_samples: Iterable[RawTextToSQLSample],
        deduplicate_internal: bool = True,
    ) -> Tuple[List[RawTextToSQLSample], DeduplicationSummary]:
        sanitized: List[RawTextToSQLSample] = []
        violations: List[LeakageViolation] = []
        train_lsh = LSHIndex(self.config) if deduplicate_internal else None

        total = 0
        internal_dupes = 0
        quarantined = 0

        for sample in train_samples:
            total += 1

            leakage = self.check_leakage(sample)
            if leakage:
                violations.append(leakage)
                quarantined += 1
                continue

            if train_lsh:
                sample_sig = self.hasher.compute_signature(sample.question)
                internal_candidates = train_lsh.query_candidates(sample_sig)
                is_dupe = False
                for cand_id in internal_candidates:
                    cand_sig = train_lsh.sample_signatures[cand_id]
                    if self.hasher.estimate_jaccard(sample_sig, cand_sig) >= self.config.threshold:
                        is_dupe = True
                        break
                if is_dupe:
                    internal_dupes += 1
                    continue
                train_lsh.insert(sample.sample_id, sample_sig)

            sanitized.append(sample)

        summary = DeduplicationSummary(
            total_records=total,
            unique_records=len(sanitized),
            internal_duplicates_dropped=internal_dupes,
            quarantined_benchmark_leakages=quarantined,
            violations=violations,
        )
        return sanitized, summary
