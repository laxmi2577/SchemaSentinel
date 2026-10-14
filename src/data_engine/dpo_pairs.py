from __future__ import annotations

import json
import re
import uuid
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from src.evaluation.red_team import AdversarialFuzzer

from src.data_engine.formatter import (
    AssistantResponse,
    ChatMLFormatter,
    DecisionType,
    PolicyRule,
    PolicyType,
)
from src.data_engine.ingestion import RawTextToSQLSample


class RejectionReason(str, Enum):
    UNAUTHORIZED_MUTATION = "UNAUTHORIZED_MUTATION"
    TENANT_LEAKAGE = "TENANT_LEAKAGE"
    RESTRICTED_SCHEMA_PROBE = "RESTRICTED_SCHEMA_PROBE"
    SYNTAX_HALLUCINATION = "SYNTAX_HALLUCINATION"
    MALICIOUS_COMPLIANCE = "MALICIOUS_COMPLIANCE"


class PreferencePair(BaseModel):
    model_config = ConfigDict(frozen=True)

    pair_id: str = Field(default_factory=lambda: f"pref_{uuid.uuid4().hex[:8]}")
    db_id: str
    prompt: str
    chosen: str
    rejected: str
    rejection_reason: RejectionReason
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PreferenceDatasetSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_pairs: int
    counts_by_reason: Dict[str, int]
    unique_db_ids: int


class DPOPairGeneratorConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    mutations_per_sample: int = 2
    tenant_leakage_per_sample: int = 1
    restricted_schema_per_sample: int = 1
    hallucinations_per_sample: int = 1
    default_tenant_column: str = "tenant_id"
    restricted_tables: List[str] = Field(
        default_factory=lambda: ["credentials", "api_keys", "audit_log", "admin_users"]
    )


class DPOPairGenerator:
    def __init__(
        self,
        config: Optional[DPOPairGeneratorConfig] = None,
        formatter: Optional[ChatMLFormatter] = None,
        fuzzer: Optional[AdversarialFuzzer] = None,
    ) -> None:
        self.config = config or DPOPairGeneratorConfig()
        self.formatter = formatter or ChatMLFormatter()
        if fuzzer is None:
            from src.evaluation.red_team import AdversarialFuzzer

            self.fuzzer = AdversarialFuzzer()
        else:
            self.fuzzer = fuzzer

    def _default_policies(self) -> List[PolicyRule]:
        return [
            PolicyRule(
                rule_id="RO_01",
                rule_type=PolicyType.READ_ONLY,
                description="Strict read-only execution policy.",
            ),
            PolicyRule(
                rule_id="TEN_01",
                rule_type=PolicyType.TENANT_ISOLATION,
                description="Mandatory tenant predicate required.",
                required_predicates=[f"{self.config.default_tenant_column} = :tenant_id"],
            ),
            PolicyRule(
                rule_id="REST_01",
                rule_type=PolicyType.RESTRICTED_TABLE,
                description="Restricted tables cannot be queried.",
                target_tables=self.config.restricted_tables,
            ),
        ]

    def _make_chosen_response(self, sql: str) -> str:
        resp = AssistantResponse(
            decision=DecisionType.ALLOW,
            sql=sql,
            rationale="Valid policy-compliant query executed successfully.",
        )
        completion = self.formatter.format_completion(resp)
        return f"{self.formatter.IM_START}assistant\n{completion}{self.formatter.IM_END}"

    def _make_refusal_chosen(self, reason: str, violations: List[str]) -> str:
        resp = AssistantResponse(
            decision=DecisionType.REFUSE,
            sql=None,
            rationale=reason,
            policy_violations=violations,
        )
        completion = self.formatter.format_completion(resp)
        return f"{self.formatter.IM_START}assistant\n{completion}{self.formatter.IM_END}"

    def _make_rejected_response(self, sql: Optional[str], rationale: str) -> str:
        resp = AssistantResponse(
            decision=DecisionType.ALLOW,
            sql=sql,
            rationale=rationale,
        )
        completion = self.formatter.format_completion(resp)
        return f"{self.formatter.IM_START}assistant\n{completion}{self.formatter.IM_END}"

    def generate_pairs_for_sample(
        self,
        sample: RawTextToSQLSample,
        ddl: Optional[str] = None,
        policies: Optional[List[PolicyRule]] = None,
    ) -> List[PreferencePair]:
        active_policies = policies or self._default_policies()
        prompt = self.formatter.format_prompt_context(
            policies=active_policies,
            user_query=sample.question,
        )
        chosen_completion = self._make_chosen_response(sample.query)
        pairs: List[PreferencePair] = []

        mutations = self.fuzzer.fuzz_mutations(sample.query)
        for attack in mutations[: self.config.mutations_per_sample]:
            if attack.mutated_sql:
                pairs.append(
                    PreferencePair(
                        db_id=sample.db_id,
                        prompt=prompt,
                        chosen=chosen_completion,
                        rejected=self._make_rejected_response(
                            attack.mutated_sql,
                            f"Executed command: {attack.name}",
                        ),
                        rejection_reason=RejectionReason.UNAUTHORIZED_MUTATION,
                        metadata={"mutation_name": attack.name, "attack_id": attack.attack_id},
                    )
                )

        leaks = self.fuzzer.fuzz_tenant_leakage(
            sample.query,
            tenant_column=self.config.default_tenant_column,
        )
        for attack in leaks[: self.config.tenant_leakage_per_sample]:
            if attack.mutated_sql:
                pairs.append(
                    PreferencePair(
                        db_id=sample.db_id,
                        prompt=prompt,
                        chosen=chosen_completion,
                        rejected=self._make_rejected_response(
                            attack.mutated_sql,
                            f"Executed cross-tenant query: {attack.name}",
                        ),
                        rejection_reason=RejectionReason.TENANT_LEAKAGE,
                        metadata={"leak_name": attack.name, "attack_id": attack.attack_id},
                    )
                )

        probes = self.fuzzer.fuzz_restricted_schema(
            restricted_tables=self.config.restricted_tables,
        )
        for attack in probes[: self.config.restricted_schema_per_sample]:
            if attack.mutated_sql:
                pairs.append(
                    PreferencePair(
                        db_id=sample.db_id,
                        prompt=prompt,
                        chosen=chosen_completion,
                        rejected=self._make_rejected_response(
                            attack.mutated_sql,
                            f"Queried restricted table: {attack.name}",
                        ),
                        rejection_reason=RejectionReason.RESTRICTED_SCHEMA_PROBE,
                        metadata={"probe_name": attack.name, "attack_id": attack.attack_id},
                    )
                )

        if self.config.hallucinations_per_sample > 0:
            hallucinated_sql = re.sub(
                r"\bSELECT\b",
                "SELECT __hallucinated_col_xyz, secret_token,",
                sample.query,
                count=1,
                flags=re.IGNORECASE,
            )
            pairs.append(
                PreferencePair(
                    db_id=sample.db_id,
                    prompt=prompt,
                    chosen=chosen_completion,
                    rejected=self._make_rejected_response(
                        hallucinated_sql,
                        "Executed query with hallucinated attributes.",
                    ),
                    rejection_reason=RejectionReason.SYNTAX_HALLUCINATION,
                    metadata={"hallucination": "__hallucinated_col_xyz"},
                )
            )

        return pairs

    def generate_refusal_preference_pair(
        self,
        db_id: str,
        adversarial_query: str,
        malicious_sql: str,
        violation_rule_id: str = "RO_01",
        policies: Optional[List[PolicyRule]] = None,
    ) -> PreferencePair:
        active_policies = policies or self._default_policies()
        prompt = self.formatter.format_prompt_context(
            policies=active_policies,
            user_query=adversarial_query,
        )
        chosen_refusal = self._make_refusal_chosen(
            reason="Query rejected by SchemaSentinel safety governance policy.",
            violations=[violation_rule_id],
        )
        rejected_compliance = self._make_rejected_response(
            sql=malicious_sql,
            rationale="Executed requested administrative mutation.",
        )
        return PreferencePair(
            db_id=db_id,
            prompt=prompt,
            chosen=chosen_refusal,
            rejected=rejected_compliance,
            rejection_reason=RejectionReason.MALICIOUS_COMPLIANCE,
            metadata={"rule_id": violation_rule_id},
        )

    def generate_preference_triplets(
        self,
        samples: Sequence[RawTextToSQLSample],
        ddl_map: Optional[Dict[str, str]] = None,
        policies: Optional[List[PolicyRule]] = None,
    ) -> List[PreferencePair]:
        all_pairs: List[PreferencePair] = []
        ddls = ddl_map or {}

        for sample in samples:
            sample_ddl = ddls.get(sample.db_id)
            sample_pairs = self.generate_pairs_for_sample(
                sample=sample,
                ddl=sample_ddl,
                policies=policies,
            )
            all_pairs.extend(sample_pairs)

        return all_pairs

    def export_to_trl_format(
        self,
        pairs: Sequence[PreferencePair],
    ) -> List[Dict[str, str]]:
        return [
            {
                "prompt": p.prompt,
                "chosen": p.chosen,
                "rejected": p.rejected,
            }
            for p in pairs
        ]

    def export_to_jsonl(
        self,
        pairs: Sequence[PreferencePair],
        file_path: str,
    ) -> int:
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(p, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(pair.model_dump_json() + "\n")
                count += 1
        return count

    def summarize_dataset(
        self,
        pairs: Sequence[PreferencePair],
    ) -> PreferenceDatasetSummary:
        counts: Dict[str, int] = {}
        db_ids = set()

        for p in pairs:
            r = p.rejection_reason.value
            counts[r] = counts.get(r, 0) + 1
            db_ids.add(p.db_id)

        return PreferenceDatasetSummary(
            total_pairs=len(pairs),
            counts_by_reason=counts,
            unique_db_ids=len(db_ids),
        )


def generate_preference_triplets(
    samples: Sequence[RawTextToSQLSample],
    ddl_map: Optional[Dict[str, str]] = None,
    policies: Optional[List[PolicyRule]] = None,
    config: Optional[DPOPairGeneratorConfig] = None,
) -> List[PreferencePair]:
    generator = DPOPairGenerator(config=config)
    return generator.generate_preference_triplets(
        samples=samples,
        ddl_map=ddl_map,
        policies=policies,
    )
