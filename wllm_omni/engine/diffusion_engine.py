from wllm_omni.config import EngineConfig
from wllm_omni.engine.model_runner import ModelRunner
from wllm_omni.outputs import OmniOutput
from wllm_omni.request import OmniRequest
from wllm_omni.sched.step_scheduler import StepScheduler


class DiffusionEngine:

    def __init__(self, config: EngineConfig, runner: ModelRunner | None = None):
        self.config = config
        # Concurrency comes from the config now that the executor batches. How
        # many of these actually run together is still decided by the scheduler:
        # only requests sharing a StepBatchSamplingParamsKey join one batch.
        self.scheduler = StepScheduler(max_num_running_reqs=config.max_num_seqs)
        self.runner = runner if runner is not None else ModelRunner(config)

    def submit(self, request: OmniRequest) -> str:
        """Queue a request without running anything.

        Requests may be submitted between step() calls: the scheduler admits a
        newcomer into the running set as soon as it is compatible with the
        requests already in flight, which is what lets an upstream stage feed
        this engine one request at a time and still get batching.
        """
        return self.scheduler.add_request(request)

    def step(self) -> list[OmniOutput]:
        """Advance every running request by one denoise step.

        Returns the outputs of requests that finished on this step, if any.
        """
        outputs, _ = self._step_once()
        return outputs

    def has_work(self) -> bool:
        return self.scheduler.has_requests()

    def generate(self, requests: OmniRequest | list[OmniRequest]) -> list[OmniOutput]:
        if isinstance(requests, OmniRequest):
            requests = [requests]

        for request in requests:
            self.submit(request)

        outputs: list[OmniOutput] = []
        while self.has_work():
            step_outputs, schedule_was_empty = self._step_once()
            if schedule_was_empty:
                break
            outputs.extend(step_outputs)

        return outputs

    def _step_once(self) -> tuple[list[OmniOutput], bool]:
        sched_output = self.scheduler.schedule()
        if sched_output.is_empty:
            return [], True

        runner_output = self.runner.execute(sched_output)
        finished_req_ids = self.scheduler.update_from_output(sched_output, runner_output)
        for finished_req_id in finished_req_ids:
            self.scheduler.pop_request_state(finished_req_id)

        return [item.result for item in runner_output.outputs if item.result is not None], False
