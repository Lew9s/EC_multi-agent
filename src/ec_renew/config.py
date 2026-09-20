"""Configuration. The only place in the codebase that reads environment vars.

The API key is held in a ``SecretStr`` and handed straight to the HTTP adapter.
Business code (``workflow`` / ``experts`` / ``rag``) never receives it.

Note: this uses ``python-dotenv`` rather than ``pydantic-settings`` because the
lab machine has no network access and ``pydantic-settings`` is not installed.
The observable contract (a frozen ``settings`` object, secrets masked) is the
same.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, SecretStr

load_dotenv(".env", override=False)


def _s(key: str, default: str = "") -> str:
    raw = os.getenv(key)
    return default if raw is None or raw.strip() == "" else raw.strip()


def _f(key: str, default: float) -> float:
    try:
        return float(_s(key, str(default)))
    except ValueError:
        return default


def _i(key: str, default: int) -> int:
    try:
        return int(float(_s(key, str(default))))
    except ValueError:
        return default


def _b(key: str, default: bool) -> bool:
    raw = _s(key, "true" if default else "false").lower()
    return raw in {"1", "true", "yes", "on"}


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    # ---- LLM (DeepSeek, OpenAI-compatible) -------------------------------
    deepseek_api_key: SecretStr = SecretStr("")  # 允许为空：FakeLLM 不需要 key
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-flash"
    deepseek_thinking: str = "disabled"  # disabled | enabled
    llm_temperature: float = 0.0
    llm_max_tokens: int = 8192  # 见 D-101：方案是系统里最长的产物，输出上限需留足余量
    llm_timeout_s: float = 120.0

    # ---- Embedding (智谱 Embedding-3, OpenAI 兼容) ------------------------
    zhipu_api_key: SecretStr = SecretStr("")  # 允许为空：FakeEmbedding 不需要 key
    zhipu_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    embedding_model: str = "embedding-3"
    # embedding-3 支持 2048 / 1024 / 512 / 256；改这里必须同步重建集合。
    embedding_dimensions: int = 2048
    embedding_batch_size: int = 16
    embedding_timeout_s: float = 60.0

    # ---- Qdrant（向量库，docker compose 启动） ----------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr = SecretStr("")
    qdrant_collection: str = "ec_renew_change_orders"

    # ---- Neo4j (社区版默认端口) -------------------------------------------
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("")

    # ---- Retrieval -------------------------------------------------------
    top_k: int = 12
    # auto = 探测可达性后选最完整的一档；降级一律显式告警，绝不静默。
    rag_backend: str = "auto"  # auto | llamaindex | graph | memory
    rrf_k: int = 60  # Reciprocal Rank Fusion 平滑常数
    graph_top_k: int = 12
    # 纯向量命中要多高的相似度才算「真实历史案例」。0 = 关闭（默认）：
    # 向量检索总会返回 top-k，仅凭此宣称有历史依据会夸大保证等级。
    # 用智谱 embedding-3 时可设 0.5 左右让语义命中也能支撑 history_backed。
    vector_min_score: float = 0.0
    # 领域三元组抽取器：rule（确定性规则，默认）| llm（LlamaIndex SchemaLLMPathExtractor）
    kg_extractor: str = "rule"

    # ---- Corpus ----------------------------------------------------------
    data_dir: Path = Path("data")
    corpus_file: str = "corpus.txt"
    # 变更单之间的分隔符（旧数据格式，正则）
    corpus_separator: str = r"!@#\$%\^&\*"

    # ---- Workflow --------------------------------------------------------
    max_rounds: int = 3
    consensus_threshold: float = 0.6
    min_effective_experts: int = 3

    # ---- Cache / events --------------------------------------------------
    cache_enabled: bool = True
    cache_dir: Path = Path(".cache/llm")
    events_path: Path = Path("logs/events.jsonl")

    @property
    def thinking_payload(self) -> dict[str, dict[str, str]]:
        """Extra request body field controlling reasoning mode."""
        return {"thinking": {"type": self.deepseek_thinking}}

    def require_api_key(self) -> str:
        """Only the real DeepSeek adapter calls this."""
        key = self.deepseek_api_key.get_secret_value()
        if not key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY 未设置。请复制 .env.example 为 .env 并填写；"
                "或使用 FakeLLM 运行（不需要 key）。"
            )
        return key

    def require_zhipu_api_key(self) -> str:
        """Only the real Zhipu embedding adapter calls this."""
        key = self.zhipu_api_key.get_secret_value()
        if not key:
            raise RuntimeError(
                "ZHIPU_API_KEY 未设置。请复制 .env.example 为 .env 并填写；"
                "或使用 FakeEmbedding 运行（不需要 key）。"
            )
        return key

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            deepseek_api_key=SecretStr(_s("DEEPSEEK_API_KEY")),
            deepseek_base_url=_s("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            deepseek_model=_s("DEEPSEEK_MODEL", "deepseek-flash"),
            deepseek_thinking=_s("DEEPSEEK_THINKING", "disabled"),
            llm_temperature=_f("LLM_TEMPERATURE", 0.0),
            llm_max_tokens=_i("LLM_MAX_TOKENS", 8192),
            llm_timeout_s=_f("LLM_TIMEOUT_S", 120.0),
            zhipu_api_key=SecretStr(_s("ZHIPU_API_KEY")),
            zhipu_base_url=_s("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
            embedding_model=_s("EMBEDDING_MODEL", "embedding-3"),
            embedding_dimensions=_i("EMBEDDING_DIMENSIONS", 2048),
            embedding_batch_size=_i("EMBEDDING_BATCH_SIZE", 16),
            embedding_timeout_s=_f("EMBEDDING_TIMEOUT_S", 60.0),
            qdrant_url=_s("QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=SecretStr(_s("QDRANT_API_KEY")),
            qdrant_collection=_s("QDRANT_COLLECTION", "ec_renew_change_orders"),
            neo4j_uri=_s("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=_s("NEO4J_USER", "neo4j"),
            neo4j_password=SecretStr(_s("NEO4J_PASSWORD")),
            top_k=_i("TOP_K", 12),
            rag_backend=_s("RAG_BACKEND", "auto"),
            rrf_k=_i("RRF_K", 60),
            graph_top_k=_i("GRAPH_TOP_K", 12),
            vector_min_score=_f("VECTOR_MIN_SCORE", 0.0),
            kg_extractor=_s("KG_EXTRACTOR", "rule"),
            data_dir=Path(_s("DATA_DIR", "data")),
            corpus_file=_s("CORPUS_FILE", "corpus.txt"),
            corpus_separator=_s("CORPUS_SEPARATOR", r"!@#\$%\^&\*"),
            max_rounds=_i("MAX_ROUNDS", 3),
            consensus_threshold=_f("CONSENSUS_THRESHOLD", 0.6),
            min_effective_experts=_i("MIN_EFFECTIVE_EXPERTS", 3),
            cache_enabled=_b("CACHE_ENABLED", True),
            cache_dir=Path(_s("CACHE_DIR", ".cache/llm")),
            events_path=Path(_s("EVENTS_PATH", "logs/events.jsonl")),
        )


settings = Settings.from_env()