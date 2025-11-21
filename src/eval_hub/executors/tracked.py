"""Base class for executors with experiment tracking integration."""

from abc import ABC
from typing import Any

from ..core.logging import get_logger
from ..models.evaluation import (
    EvaluationRequest,
    EvaluationResult,
    EvaluationSpec,
    EvaluationStatus,
    ExperimentConfig,
    Model,
)
from ..utils.datetime_utils import utcnow
from .base import ExecutionContext, Executor


class TrackedExecutor(Executor, ABC):
    """Base class for executors with experiment tracking integration."""

    def __init__(self, backend_config: dict[str, Any]):
        super().__init__(backend_config)
        self.logger = get_logger(self.__class__.__name__)

    async def _track_start(
        self, context: ExecutionContext, provider_name: str
    ) -> str | None:
        """Start experiment tracking for this evaluation.

        Returns:
            Tracking run ID if successful, None if tracking unavailable
        """
        if not context.mlflow_client:
            self.logger.debug(
                "No tracking client available, skipping experiment tracking"
            )
            return None

        try:
            # Create evaluation spec from context
            evaluation_spec = EvaluationSpec(
                id=context.evaluation_id,
                name=f"{provider_name}_{context.benchmark_spec.name}_{context.model_name}",
                description=None,
                model=Model(url=context.model_url, name=context.model_name),
                collection_id=None,
                risk_category=None,
                priority=1,
            )

            # Create evaluation request for experiment creation
            experiment_name = (
                context.experiment_name
                or f"{provider_name}_{context.benchmark_spec.name}"
            )
            evaluation_request = EvaluationRequest(
                evaluations=[evaluation_spec],
                experiment=ExperimentConfig(name=experiment_name),
                callback_url=None,
                created_at=context.started_at,
            )

            # Create experiment and run
            experiment_id = await context.mlflow_client.create_experiment(
                evaluation_request
            )
            run_id: str = await context.mlflow_client.start_evaluation_run(
                experiment_id=experiment_id,
                evaluation=evaluation_spec,
                backend_name=provider_name,
                benchmark_name=context.benchmark_spec.name,
            )

            self.logger.info(
                "Started experiment tracking",
                evaluation_id=str(context.evaluation_id),
                run_id=run_id,
                experiment_id=experiment_id,
            )

            return run_id

        except Exception as e:
            self.logger.error(
                "Failed to start experiment tracking",
                evaluation_id=str(context.evaluation_id),
                error=str(e),
            )
            return None

    async def _track_complete(
        self, result: EvaluationResult, context: ExecutionContext
    ) -> None:
        """Log completed evaluation result to experiment tracking."""
        if not context.mlflow_client or not result.mlflow_run_id:
            return

        try:
            await context.mlflow_client.log_evaluation_result(result)
            self.logger.info(
                "Logged result to experiment tracking",
                evaluation_id=str(result.evaluation_id),
                run_id=result.mlflow_run_id,
            )
        except Exception as e:
            self.logger.error(
                "Failed to log result to experiment tracking",
                evaluation_id=str(result.evaluation_id),
                error=str(e),
            )

    async def _track_failure(
        self,
        context: ExecutionContext,
        provider_name: str,
        error_message: str,
        run_id: str | None = None,
    ) -> str | None:
        """Log a failed evaluation to experiment tracking.

        Returns:
            Tracking run ID if logged successfully
        """
        if not context.mlflow_client:
            return run_id

        try:
            # Create run if not provided
            if not run_id:
                run_id = await self._track_start(context, provider_name)

            if run_id:
                failed_result = EvaluationResult(
                    evaluation_id=context.evaluation_id,
                    provider_id=provider_name,
                    benchmark_id=context.benchmark_spec.name,
                    benchmark_name=context.benchmark_spec.name,
                    status=EvaluationStatus.FAILED,
                    error_message=error_message,
                    started_at=context.started_at,
                    completed_at=utcnow(),
                    duration_seconds=0.0,
                    mlflow_run_id=run_id,
                    metrics={},
                    artifacts={},
                )

                await context.mlflow_client.log_evaluation_result(failed_result)

        except Exception as e:
            self.logger.error(
                "Failed to log failed evaluation to experiment tracking",
                evaluation_id=str(context.evaluation_id),
                error=str(e),
            )

        return run_id

    def _with_tracking(
        self, result: EvaluationResult, run_id: str | None
    ) -> EvaluationResult:
        """Return evaluation result with tracking ID attached."""
        # Create new result with tracking run_id
        return EvaluationResult(
            evaluation_id=result.evaluation_id,
            provider_id=result.provider_id,
            benchmark_id=result.benchmark_id,
            benchmark_name=result.benchmark_name,
            status=result.status,
            metrics=result.metrics,
            artifacts=result.artifacts,
            started_at=result.started_at,
            completed_at=result.completed_at,
            duration_seconds=result.duration_seconds,
            error_message=result.error_message,
            mlflow_run_id=run_id,  # Tracking ID attached
        )
