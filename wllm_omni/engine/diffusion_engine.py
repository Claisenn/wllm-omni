from wllm_omni.config import EngineConfig
from wllm_omni.engine.model_runner import ModelRunner
from wllm_omni.outputs import OmniOutput
from wllm_omni.request import OmniRequest
from wllm_omni.sched.step_scheduler import StepScheduler


class DiffusionEngine:

    def __init__(self, config: EngineConfig):
        self.config = config
        # Concurrency comes from the config now that the executor batches. How
        # many of these actually run together is still decided by the scheduler:
        # only requests sharing a StepBatchSamplingParamsKey join one batch.
        self.scheduler = StepScheduler(max_num_running_reqs=config.max_num_seqs)
        self.runner = ModelRunner(config)

    def generate(self, requests: OmniRequest | list[OmniRequest]) -> list[OmniOutput]:
        if isinstance(requests, OmniRequest):
            requests = [requests]

        for request in requests:
            self.scheduler.add_request(request)

        outputs: list[OmniOutput] = []
        while self.scheduler.has_requests():
            sched_output = self.scheduler.schedule()
            if sched_output.is_empty:
                break

            runner_output = self.runner.execute(sched_output)
            finished_req_ids = self.scheduler.update_from_output(sched_output, runner_output)
            for finished_req_id in finished_req_ids:
                self.scheduler.pop_request_state(finished_req_id)

            for item in runner_output.outputs:
                if item.result is not None:
                    outputs.append(item.result)

        return outputs
