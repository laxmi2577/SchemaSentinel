from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import uuid

from src.agentic_ops import ActiveLearningPair, AgentConfig, SelfHealAttempt
from src.data_engine.dpo_pairs import PreferencePair, RejectionReason


class ActiveLearningFlywheel:
    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        export_path: Optional[str] = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.export_path = export_path or self.config.active_learning_export_path
        self._pairs: List[ActiveLearningPair] = []

    def add_pair(self, pair: ActiveLearningPair) -> None:
        self._pairs.append(pair)

    def get_pairs(self) -> List[ActiveLearningPair]:
        return list(self._pairs)

    def mine_from_self_healing(
        self,
        attempts: Sequence[SelfHealAttempt],
        tenant_id: str,
        prompt: Optional[str] = None,
    ) -> List[ActiveLearningPair]:
        mined: List[ActiveLearningPair] = []
        user_prompt = prompt or f"Generate safe analytical query for tenant {tenant_id}"

        successful_attempts = [a for a in attempts if a.execution_success]
        if not successful_attempts:
            return mined

        winning_query = successful_attempts[-1].proposed_query

        for att in attempts:
            if not att.execution_success:
                pair = ActiveLearningPair(
                    pair_id=f"alp_{uuid.uuid4().hex[:8]}",
                    tenant_id=tenant_id,
                    prompt=user_prompt,
                    chosen_query=winning_query,
                    rejected_query=att.proposed_query if att.proposed_query != winning_query else att.original_query,
                    rejection_reason=f"SCHEMA_DRIFT_{att.detected_drift_type}",
                    estimated_reward_margin=1.5,
                    source="self_healing_loop",
                )
                self.add_pair(pair)
                mined.append(pair)

        return mined

    def mine_from_dlq(
        self,
        dlq_records: Sequence[Any],
        default_chosen_template: str = "SELECT tenant_id, COUNT(*) FROM fact_query_events WHERE tenant_id = '{tenant_id}' GROUP BY tenant_id;",
    ) -> List[ActiveLearningPair]:
        mined: List[ActiveLearningPair] = []

        for rec in dlq_records:
            tenant_id = getattr(rec, "tenant_id", "default_tenant")
            failing_sql = getattr(rec, "failing_sql", getattr(rec, "payload", {}).get("generated_sql", ""))
            error_class = getattr(rec, "error_class", "UNKNOWN_ERROR")

            if not failing_sql:
                continue

            chosen = default_chosen_template.format(tenant_id=tenant_id)
            pair = ActiveLearningPair(
                pair_id=f"alp_{uuid.uuid4().hex[:8]}",
                tenant_id=tenant_id,
                prompt=f"Generate compliant read-only query for tenant {tenant_id}",
                chosen_query=chosen,
                rejected_query=failing_sql,
                rejection_reason=error_class,
                estimated_reward_margin=2.0 if "MUTATION" in error_class else 1.2,
                source="dlq_quarantine",
            )
            self.add_pair(pair)
            mined.append(pair)

        return mined

    def export_dataset(self, target_path: Optional[str] = None) -> Tuple[str, int]:
        out_file = Path(target_path or self.export_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        count = 0
        with open(out_file, "w", encoding="utf-8") as fh:
            for pair in self._pairs:
                dpo_dict = {
                    "pair_id": pair.pair_id,
                    "prompt": pair.prompt,
                    "chosen": pair.chosen_query,
                    "rejected": pair.rejected_query,
                    "rejection_reason": pair.rejection_reason,
                    "metadata": {
                        "tenant_id": pair.tenant_id,
                        "source": pair.source,
                        "estimated_reward_margin": pair.estimated_reward_margin,
                    },
                }
                fh.write(json.dumps(dpo_dict) + "\n")
                count += 1

        return str(out_file), count

    def get_dataset_summary(self) -> Dict[str, Any]:
        reason_counts: Dict[str, int] = {}
        for p in self._pairs:
            reason_counts[p.rejection_reason] = reason_counts.get(p.rejection_reason, 0) + 1

        margins = [p.estimated_reward_margin for p in self._pairs]
        avg_margin = float(sum(margins) / len(margins)) if margins else 0.0

        return {
            "total_pairs": len(self._pairs),
            "rejection_breakdown": reason_counts,
            "average_estimated_reward_margin": round(avg_margin, 3),
            "export_path": self.export_path,
        }
