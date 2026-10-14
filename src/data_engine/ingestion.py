from __future__ import annotations

import json
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional
from pydantic import BaseModel, ConfigDict, Field
from src.data_engine.schema_parser import ASTSchemaParser, DatabaseCatalog


class BenchmarkSplit(str, Enum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"


class BenchmarkSource(str, Enum):
    SPIDER = "spider"
    BIRD = "bird"


class RawTextToSQLSample(BaseModel):
    model_config = ConfigDict(frozen=True)

    sample_id: str
    db_id: str
    question: str
    query: str
    source: BenchmarkSource = BenchmarkSource.SPIDER
    split: BenchmarkSplit = BenchmarkSplit.TRAIN
    difficulty: Optional[str] = None
    evidence: Optional[str] = None


class DatasetStreamConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    batch_size: int = 100
    max_samples: Optional[int] = None
    cloud_storage_base_uri: Optional[str] = None
    filter_db_ids: Optional[List[str]] = None


class SpiderDatasetStreamer:
    def __init__(
        self,
        config: Optional[DatasetStreamConfig] = None,
        schema_parser: Optional[ASTSchemaParser] = None,
    ) -> None:
        self.config = config or DatasetStreamConfig()
        self.schema_parser = schema_parser or ASTSchemaParser(default_dialect="sqlite")

    def parse_raw_item(
        self,
        item: Dict[str, Any],
        split: BenchmarkSplit,
        index: int,
    ) -> RawTextToSQLSample:
        db_id = item.get("db_id", "unknown_db")
        question = item.get("question", "").strip()
        query = item.get("query", "").strip()

        sample_id = f"spider_{split.value}_{db_id}_{index}"

        return RawTextToSQLSample(
            sample_id=sample_id,
            db_id=db_id,
            question=question,
            query=query,
            source=BenchmarkSource.SPIDER,
            split=split,
            difficulty=item.get("difficulty"),
        )

    def stream_from_records(
        self,
        records: List[Dict[str, Any]],
        split: BenchmarkSplit,
    ) -> Iterator[RawTextToSQLSample]:
        emitted_count = 0
        for idx, item in enumerate(records):
            if self.config.max_samples and emitted_count >= self.config.max_samples:
                break

            db_id = item.get("db_id")
            if self.config.filter_db_ids and db_id not in self.config.filter_db_ids:
                continue

            sample = self.parse_raw_item(item, split, idx)
            yield sample
            emitted_count += 1

    def parse_schemas_from_dict_list(
        self,
        schemas_data: List[Dict[str, Any]],
    ) -> Dict[str, DatabaseCatalog]:
        catalogs: Dict[str, DatabaseCatalog] = {}
        for db_meta in schemas_data:
            db_id = db_meta.get("db_id", "")
            if not db_id:
                continue
            catalog = self.schema_parser.from_spider_tables_dict(db_meta, db_id=db_id)
            catalogs[db_id] = catalog
        return catalogs


class BIRDDatasetStreamer:
    def __init__(
        self,
        config: Optional[DatasetStreamConfig] = None,
        schema_parser: Optional[ASTSchemaParser] = None,
    ) -> None:
        self.config = config or DatasetStreamConfig()
        self.schema_parser = schema_parser or ASTSchemaParser(default_dialect="sqlite")

    def parse_raw_item(
        self,
        item: Dict[str, Any],
        split: BenchmarkSplit,
        index: int,
    ) -> RawTextToSQLSample:
        db_id = item.get("db_id", "unknown_db")
        question = item.get("question", "").strip()
        query = item.get("SQL", "").strip() or item.get("query", "").strip()
        evidence = item.get("evidence", "").strip() or None
        difficulty = item.get("difficulty")

        sample_id = f"bird_{split.value}_{db_id}_{index}"

        return RawTextToSQLSample(
            sample_id=sample_id,
            db_id=db_id,
            question=question,
            query=query,
            source=BenchmarkSource.BIRD,
            split=split,
            difficulty=difficulty,
            evidence=evidence,
        )

    def stream_from_records(
        self,
        records: List[Dict[str, Any]],
        split: BenchmarkSplit,
    ) -> Iterator[RawTextToSQLSample]:
        emitted_count = 0
        for idx, item in enumerate(records):
            if self.config.max_samples and emitted_count >= self.config.max_samples:
                break

            db_id = item.get("db_id")
            if self.config.filter_db_ids and db_id not in self.config.filter_db_ids:
                continue

            sample = self.parse_raw_item(item, split, idx)
            yield sample
            emitted_count += 1

    def parse_schemas_from_dict_list(
        self,
        schemas_data: List[Dict[str, Any]],
    ) -> Dict[str, DatabaseCatalog]:
        catalogs: Dict[str, DatabaseCatalog] = {}
        for db_meta in schemas_data:
            db_id = db_meta.get("db_id", "")
            if not db_id:
                continue
            catalog = self.schema_parser.from_spider_tables_dict(db_meta, db_id=db_id)
            catalogs[db_id] = catalog
        return catalogs


class UnifiedDataEngineStreamer:
    def __init__(
        self,
        config: Optional[DatasetStreamConfig] = None,
        schema_parser: Optional[ASTSchemaParser] = None,
    ) -> None:
        self.config = config or DatasetStreamConfig()
        self.schema_parser = schema_parser or ASTSchemaParser(default_dialect="sqlite")
        self.spider_streamer = SpiderDatasetStreamer(self.config, self.schema_parser)
        self.bird_streamer = BIRDDatasetStreamer(self.config, self.schema_parser)

    def stream_mixed_samples(
        self,
        spider_records: List[Dict[str, Any]],
        bird_records: List[Dict[str, Any]],
        split: BenchmarkSplit,
    ) -> Iterator[RawTextToSQLSample]:
        emitted = 0
        spider_iter = self.spider_streamer.stream_from_records(spider_records, split)
        bird_iter = self.bird_streamer.stream_from_records(bird_records, split)

        while True:
            has_yielded = False
            if self.config.max_samples and emitted >= self.config.max_samples:
                break

            try:
                spider_sample = next(spider_iter)
                yield spider_sample
                emitted += 1
                has_yielded = True
                if self.config.max_samples and emitted >= self.config.max_samples:
                    break
            except StopIteration:
                pass

            try:
                bird_sample = next(bird_iter)
                yield bird_sample
                emitted += 1
                has_yielded = True
            except StopIteration:
                pass

            if not has_yielded:
                break
