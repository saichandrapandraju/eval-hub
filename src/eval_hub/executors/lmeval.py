"""LM Evaluation Harness executor for running evaluations."""

import asyncio
import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from ..core.exceptions import BackendError
from ..core.logging import get_logger
from ..models.evaluation import EvaluationResult, EvaluationStatus
from .base import ExecutionContext, Executor


class LMEvalExecutor(Executor):
    """Executor for running evaluations using lm-evaluation-harness."""

    def __init__(self, backend_config: dict[str, Any]):
        self.logger = get_logger(__name__)
        # Set model before super().__init__() to avoid validation error
        # Model will be taken from context during execution if not in config
        self.model = backend_config.get("model", None)
        super().__init__(backend_config)

        # Default configuration
        self.model_args = backend_config.get("model_args", "")
        self.batch_size = backend_config.get("batch_size", "auto")
        self.device = backend_config.get("device", "cuda:0")
        self.output_path = backend_config.get("output_path", "/tmp/lmeval_results")
        self.limit = backend_config.get("limit", None)
        self.num_fewshot = backend_config.get("num_fewshot", 0)
        self.lm_eval_path = backend_config.get("lm_eval_path", "lm_eval")
        self.timeout_seconds = backend_config.get("timeout_seconds", 3600)

        # Ensure output directory exists
        Path(self.output_path).mkdir(parents=True, exist_ok=True)

    def _validate_config(self) -> None:
        """Validate LM Evaluation Harness configuration."""
        # Model can be provided in config or will be taken from context during execution
        pass

    @classmethod
    def get_backend_type(cls) -> str:
        """Get the backend type identifier."""
        return "lm-evaluation-harness"

    async def health_check(self) -> bool:
        """Check if lm-evaluation-harness is available."""
        return True

    async def execute_benchmark(
        self,
        context: ExecutionContext,
        progress_callback: Callable[[str, float, str], None] | None = None,
    ) -> EvaluationResult:
        """Execute a benchmark evaluation using lm-evaluation-harness."""

        # Get model from context if not in config
        model = self.model or context.model_name
        if not model:
            raise BackendError("Model name is required (either in backend config or context)")

        self.logger.info(
            "Starting LM Evaluation Harness execution",
            evaluation_id=str(context.evaluation_id),
            benchmark=context.benchmark_spec.name,
            model=model,
        )

        try:
            # Report progress start
            if progress_callback:
                progress_callback(
                    str(context.evaluation_id),
                    0.0,
                    f"Preparing {context.benchmark_spec.name} for LM Evaluation Harness",
                )

            # Build command arguments
            task_name = context.benchmark_spec.name
            tasks = context.benchmark_spec.tasks or [task_name]

            # Construct and log LMEval job YAML CR
            job_cr = self._build_lmeval_job_cr(context, tasks, model)
            job_cr_yaml = yaml.dump(job_cr, default_flow_style=False, sort_keys=False)
            self.logger.info(
                "LMEval Job YAML CR",
                evaluation_id=str(context.evaluation_id),
                benchmark=context.benchmark_spec.name,
            )
            # Log YAML as a separate message for better readability
            self.logger.info(f"LMEval Job YAML CR:\n{job_cr_yaml}")

            # Report progress: preparing command
            if progress_callback:
                progress_callback(
                    str(context.evaluation_id),
                    10.0,
                    f"Building command for {task_name}",
                )

            # Execute the evaluation
            result = await self._run_lm_eval(
                tasks=tasks,
                context=context,
                model=model,
                progress_callback=progress_callback,
            )

            # Report completion
            if progress_callback:
                progress_callback(
                    str(context.evaluation_id),
                    100.0,
                    f"Completed {context.benchmark_spec.name} on LM Evaluation Harness",
                )

            # Convert result to eval-hub format
            eval_result = await self._convert_lmeval_result_to_eval_hub(result, context)

            self.logger.info(
                "LM Evaluation Harness execution completed",
                evaluation_id=str(context.evaluation_id),
                benchmark=context.benchmark_spec.name,
                status=eval_result.status,
            )

            return eval_result

        except Exception as e:
            self.logger.error(
                "LM Evaluation Harness execution failed",
                evaluation_id=str(context.evaluation_id),
                benchmark=context.benchmark_spec.name,
                error=str(e),
            )

            return EvaluationResult(
                evaluation_id=context.evaluation_id,
                backend_name="lm-evaluation-harness",
                benchmark_name=context.benchmark_spec.name,
                status=EvaluationStatus.FAILED,
                error_message=str(e),
                started_at=context.started_at,
                completed_at=datetime.utcnow(),
                duration_seconds=(
                    datetime.utcnow() - context.started_at
                ).total_seconds(),
            )

    async def cleanup(self) -> None:
        """Perform post-evaluation cleanup."""
        # Cleanup is handled by the subprocess
        pass

    def _build_lmeval_job_cr(
        self, context: ExecutionContext, tasks: list[str], model: str
    ) -> dict[str, Any]:
        """Build an LMEval job Kubernetes Custom Resource YAML."""
        benchmark_config = context.benchmark_spec.config or {}

        # Determine limit
        limit = benchmark_config.get("limit") or self.limit
        num_fewshot = benchmark_config.get("num_fewshot") or self.num_fewshot

        # Build model args
        model_args_dict: dict[str, Any] = {}
        if self.model_args:
            try:
                model_args_dict = json.loads(self.model_args)
            except json.JSONDecodeError:
                model_args_dict = {"raw": self.model_args}

        # Construct the CR
        job_cr = {
            "apiVersion": "evaluation.nvidia.com/v1alpha1",
            "kind": "LMEvalJob",
            "metadata": {
                "name": f"lmeval-{context.evaluation_id}-{context.benchmark_spec.name}",
                "namespace": self.backend_config.get("namespace", "default"),
                "labels": {
                    "evaluation-id": str(context.evaluation_id),
                    "benchmark": context.benchmark_spec.name,
                    "model": model,
                    "backend": "lm-evaluation-harness",
                },
            },
            "spec": {
                "model": model,
                "modelArgs": model_args_dict if model_args_dict else None,
                "tasks": tasks,
                "batchSize": self.batch_size,
                "device": self.device,
                "numFewshot": num_fewshot,
                "outputPath": self.output_path,
                "limit": limit,
                "timeoutSeconds": self.timeout_seconds,
                "config": {
                    "evaluationId": str(context.evaluation_id),
                    "modelName": context.model_name,
                    "modelServerId": context.model_server_id,
                    "benchmarkName": context.benchmark_spec.name,
                    "startedAt": context.started_at.isoformat() if context.started_at else None,
                },
            },
        }

        # Remove None values from spec
        spec = job_cr["spec"]
        if spec.get("modelArgs") is None:
            spec.pop("modelArgs", None)
        if spec.get("limit") is None:
            spec.pop("limit", None)

        return job_cr

    async def _run_lm_eval(
        self,
        tasks: list[str],
        context: ExecutionContext,
        model: str,
        progress_callback: Callable[[str, float, str], None] | None,
    ) -> dict[str, Any]:
        """Run lm-evaluation-harness command and return results."""

        # Build command
        cmd = [
            self.lm_eval_path,
            "--model",
            model,
            "--tasks",
            ",".join(tasks),
            "--batch_size",
            str(self.batch_size),
            "--device",
            self.device,
            "--num_fewshot",
            str(self.num_fewshot),
            "--output_path",
            self.output_path,
        ]

        # Add model args if provided
        if self.model_args:
            cmd.extend(["--model_args", self.model_args])

        # Add limit if provided
        if self.limit:
            cmd.extend(["--limit", str(self.limit)])

        # Add benchmark-specific config
        benchmark_config = context.benchmark_spec.config or {}
        if "limit" in benchmark_config:
            cmd.extend(["--limit", str(benchmark_config["limit"])])
        if "num_fewshot" in benchmark_config:
            cmd.extend(["--num_fewshot", str(benchmark_config["num_fewshot"])])

        # Add output file path
        output_file = f"{self.output_path}/results_{context.evaluation_id}_{context.benchmark_spec.name}.json"
        cmd.extend(["--output_path", output_file])

        self.logger.debug(
            "Running LM Evaluation Harness command",
            evaluation_id=str(context.evaluation_id),
            command=" ".join(cmd),
        )

        # Report progress: executing
        if progress_callback:
            progress_callback(
                str(context.evaluation_id),
                30.0,
                "Running LM Evaluation Harness evaluation",
            )

        # Run the command with timeout
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Monitor progress (simplified - in real implementation, parse stdout)
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout_seconds,
            )

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise BackendError(
                    f"LM Evaluation Harness failed with return code {process.returncode}: {error_msg}"
                )

            # Report progress: processing results
            if progress_callback:
                progress_callback(
                    str(context.evaluation_id),
                    90.0,
                    "Processing LM Evaluation Harness results",
                )

            # Load results from output file
            if Path(output_file).exists():
                with open(output_file, "r") as f:
                    result_data = json.load(f)
                return result_data
            else:
                # Try to parse stdout as JSON
                try:
                    return json.loads(stdout.decode())
                except json.JSONDecodeError:
                    raise BackendError(
                        "Failed to parse LM Evaluation Harness output as JSON"
                    )

        except asyncio.TimeoutError:
            raise BackendError(
                f"LM Evaluation Harness execution timed out after {self.timeout_seconds} seconds"
            )

    async def _convert_lmeval_result_to_eval_hub(
        self, lmeval_result: dict[str, Any], context: ExecutionContext
    ) -> EvaluationResult:
        """Convert LM Evaluation Harness result to eval-hub EvaluationResult format."""

        metrics = {}
        artifacts = {}

        # Extract metrics from results
        # LM Evaluation Harness results structure: {task_name: {metric_name: value}}
        for task_name, task_results in lmeval_result.items():
            if isinstance(task_results, dict):
                for metric_name, metric_value in task_results.items():
                    # Flatten metric names
                    full_metric_name = (
                        f"{task_name}_{metric_name}"
                        if len(lmeval_result) > 1
                        else metric_name
                    )
                    metrics[full_metric_name] = metric_value

        # Add artifacts
        output_file = f"{self.output_path}/results_{context.evaluation_id}_{context.benchmark_spec.name}.json"
        artifacts["lmeval_results"] = output_file

        # Save full result for debugging
        full_result_file = f"{self.output_path}/full_results_{context.evaluation_id}_{context.benchmark_spec.name}.json"
        with open(full_result_file, "w") as f:
            json.dump(lmeval_result, f, indent=2, default=str)
        artifacts["lmeval_full_results"] = full_result_file

        return EvaluationResult(
            evaluation_id=context.evaluation_id,
            backend_name="lm-evaluation-harness",
            benchmark_name=context.benchmark_spec.name,
            status=EvaluationStatus.COMPLETED,
            metrics=metrics,
            artifacts=artifacts,
            started_at=context.started_at,
            completed_at=datetime.utcnow(),
            duration_seconds=(datetime.utcnow() - context.started_at).total_seconds(),
        )

    def get_recommended_timeout_minutes(self) -> int:
        """Get the recommended timeout for LM Evaluation Harness."""
        return self.timeout_seconds // 60

    def get_max_retry_attempts(self) -> int:
        """Get the maximum retry attempts for LM Evaluation Harness."""
        return self.backend_config.get("max_retries", 2)

