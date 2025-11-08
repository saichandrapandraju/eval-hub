"""Data models for NeMo Evaluator API integration."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EndpointType(str, Enum):
    """Endpoint types supported by NeMo Evaluator."""

    UNDEFINED = "undefined"
    CHAT = "chat"
    COMPLETIONS = "completions"
    VLM = "vlm"
    EMBEDDING = "embedding"


class NemoApiEndpoint(BaseModel):
    """API endpoint configuration for NeMo Evaluator."""

    model_config = ConfigDict(extra="allow")

    api_key: str | None = Field(None, description="API key environment variable name")
    model_id: str | None = Field(None, description="Model identifier")
    stream: bool | None = Field(None, description="Whether to stream responses")
    type: EndpointType | None = Field(None, description="Endpoint type")
    url: str | None = Field(None, description="API endpoint URL")


class NemoConfigParams(BaseModel):
    """Configuration parameters for NeMo Evaluator execution."""

    model_config = ConfigDict(extra="allow")

    limit_samples: int | float | None = Field(
        None, description="Limit evaluation samples"
    )
    max_new_tokens: int | None = Field(None, description="Maximum tokens to generate")
    max_retries: int | None = Field(None, description="Number of REST request retries")
    parallelism: int | None = Field(None, description="Execution parallelism")
    task: str | None = Field(None, description="Task name")
    temperature: float | None = Field(None, description="Generation temperature")
    request_timeout: int | None = Field(None, description="REST response timeout")
    top_p: float | None = Field(None, description="Top-p sampling parameter")
    extra: dict[str, Any] | None = Field(
        default_factory=dict, description="Framework specific parameters"
    )


class NemoEvaluationTarget(BaseModel):
    """Target configuration for NeMo Evaluator API endpoints."""

    api_endpoint: NemoApiEndpoint | None = Field(
        None, description="API endpoint configuration"
    )


class NemoEvaluationConfig(BaseModel):
    """Evaluation configuration for NeMo Evaluator."""

    model_config = ConfigDict(extra="allow")

    output_dir: str | None = Field(None, description="Output directory for results")
    params: NemoConfigParams | None = Field(None, description="Evaluation parameters")
    supported_endpoint_types: list[str] | None = Field(
        None, description="Supported endpoint types"
    )
    type: str | None = Field(None, description="Task type")


class NemoEvaluationRequest(BaseModel):
    """Complete evaluation request for NeMo Evaluator."""

    model_config = ConfigDict(extra="allow")

    command: str = Field(..., description="Jinja template of command to execute")
    framework_name: str = Field(..., description="Framework name")
    pkg_name: str = Field(..., description="Package name")
    config: NemoEvaluationConfig = Field(..., description="Evaluation configuration")
    target: NemoEvaluationTarget = Field(..., description="Target configuration")


# Response models for NeMo Evaluator results
class NemoScoreStats(BaseModel):
    """Statistics for a score in NeMo Evaluator results."""

    count: int | None = Field(None, description="Number of values")
    sum: float | None = Field(None, description="Sum of values")
    sum_squared: float | None = Field(None, description="Sum of squared values")
    min: float | None = Field(None, description="Minimum value")
    max: float | None = Field(None, description="Maximum value")
    mean: float | None = Field(None, description="Mean value")
    variance: float | None = Field(None, description="Population variance")
    stddev: float | None = Field(None, description="Population standard deviation")
    stderr: float | None = Field(None, description="Standard error")


class NemoScore(BaseModel):
    """Score information from NeMo Evaluator."""

    value: float = Field(..., description="Score value")
    stats: NemoScoreStats = Field(..., description="Score statistics")


class NemoMetricResult(BaseModel):
    """Metric result from NeMo Evaluator."""

    scores: dict[str, NemoScore] = Field(
        default_factory=dict, description="Metric scores"
    )


class NemoTaskResult(BaseModel):
    """Task result from NeMo Evaluator."""

    metrics: dict[str, NemoMetricResult] = Field(
        default_factory=dict, description="Task metrics"
    )


class NemoGroupResult(BaseModel):
    """Group result from NeMo Evaluator."""

    groups: dict[str, "NemoGroupResult"] | None = Field(
        None, description="Subgroup results"
    )
    metrics: dict[str, NemoMetricResult] = Field(
        default_factory=dict, description="Group metrics"
    )


class NemoEvaluationResult(BaseModel):
    """Complete evaluation result from NeMo Evaluator."""

    tasks: dict[str, NemoTaskResult] | None = Field(
        default_factory=dict, description="Task results"
    )
    groups: dict[str, NemoGroupResult] | None = Field(
        default_factory=dict, description="Group results"
    )


class NemoContainerConfig(BaseModel):
    """Configuration for NeMo Evaluator container endpoints."""

    model_config = ConfigDict(extra="allow")

    endpoint: str = Field(..., description="NeMo Evaluator container endpoint URL")
    port: int = Field(default=3825, description="NeMo Evaluator adapter port")
    timeout_seconds: int = Field(default=3600, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    health_check_endpoint: str | None = Field(None, description="Health check endpoint")
    auth_token: str | None = Field(None, description="Authentication token")
    verify_ssl: bool = Field(default=True, description="Verify SSL certificates")

    def get_full_endpoint(self) -> str:
        """Get the complete endpoint URL."""
        if "://" in self.endpoint:
            return (
                f"{self.endpoint}:{self.port}"
                if not self.endpoint.endswith(str(self.port))
                else self.endpoint
            )
        else:
            return f"http://{self.endpoint}:{self.port}"

    def get_health_check_url(self) -> str | None:
        """Get the health check URL if configured."""
        if self.health_check_endpoint:
            base = self.get_full_endpoint()
            return f"{base}{self.health_check_endpoint}"
        return None


# Update GroupResult to allow self-referencing
NemoGroupResult.model_rebuild()
