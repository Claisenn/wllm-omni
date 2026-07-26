from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from wllm_omni.engine.connectors import ConnectorContext
from wllm_omni.engine.stage import Stage, StageOutput
from wllm_omni.engine.stage_graph import StageGraph, StageNode, StageResultStore
from wllm_omni.request import OmniRequest


@dataclass(slots=True)
class StageExecutionRecord:
    node_id: str
    stage_name: str
    request_id: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class StageSchedulerResult:
    root_request_id: str
    outputs: StageResultStore
    records: list[StageExecutionRecord]
    final_outputs: list[StageOutput]


@dataclass(slots=True)
class StageBatchSchedulerResult:
    root_request_ids: list[str]
    records: list[StageExecutionRecord]
    # One list of per-request outputs per leaf node, in graph leaf order.
    final_outputs: list[list[StageOutput]]


class StageScheduler:
    """Executes a StageGraph in dependency order.

    This scheduler is the top-level omni scheduler. It decides which stage node
    is ready, runs that stage, then uses edge connectors to create downstream
    requests. It deliberately does not understand AR tokens, KV cache, or
    diffusion denoise steps.
    """

    def __init__(self, graph: StageGraph):
        self.graph = graph
        self.graph.validate()

    def run(self, root_request: OmniRequest) -> StageSchedulerResult:
        outputs = StageResultStore()
        records: list[StageExecutionRecord] = []
        completed: set[str] = set()
        scheduled: set[str] = set()
        requests: dict[str, OmniRequest] = {}

        for root in self.graph.roots():
            requests[root.node_id] = root_request

        while len(completed) < len(self.graph.nodes):
            ready_nodes = self.graph.ready_nodes(completed=completed, scheduled=scheduled)
            if not ready_nodes:
                remaining = sorted(set(self.graph.nodes) - completed)
                raise RuntimeError(f"StageGraph stalled; remaining nodes: {remaining}")

            for node in ready_nodes:
                scheduled.add(node.node_id)
                request = self._request_for_node(node, root_request, requests, outputs)
                output, elapsed_s = self._run_stage(node.stage, request)
                outputs.put(node.node_id, output)
                records.append(self._make_record(node, output, elapsed_s, root_request))
                completed.add(node.node_id)
                self._materialize_downstream_requests(node, root_request, requests, outputs)

        final_outputs = [outputs.get(node.node_id) for node in self.graph.leaves()]
        return StageSchedulerResult(
            root_request_id=root_request.request_id,
            outputs=outputs,
            records=records,
            final_outputs=final_outputs,
        )

    def run_batch(self, root_requests: list[OmniRequest]) -> StageBatchSchedulerResult:
        """Execute the stage graph for several requests together.

        The DAG walk is identical to run(); what changes is the unit handed to
        each stage. Every ready node receives the full request list via
        run_batch, so a stage that can batch (diffusion) sees all requests in
        one call, while stages that cannot (AR) fall back to a loop. Connectors
        stay per-request: the i-th downstream request is built from the i-th
        upstream output.
        """
        if not root_requests:
            raise ValueError("run_batch requires at least one root request.")

        records: list[StageExecutionRecord] = []
        completed: set[str] = set()
        scheduled: set[str] = set()
        requests: dict[str, list[OmniRequest]] = {}
        outputs: dict[str, list[StageOutput]] = {}

        for root in self.graph.roots():
            requests[root.node_id] = list(root_requests)

        while len(completed) < len(self.graph.nodes):
            ready_nodes = self.graph.ready_nodes(completed=completed, scheduled=scheduled)
            if not ready_nodes:
                remaining = sorted(set(self.graph.nodes) - completed)
                raise RuntimeError(f"StageGraph stalled; remaining nodes: {remaining}")

            for node in ready_nodes:
                scheduled.add(node.node_id)
                node_requests = self._requests_for_node_batch(node, root_requests, requests, outputs)
                prepare_metadata = node.stage.prepare()
                start = perf_counter()
                node_outputs = node.stage.run_batch(node_requests)
                elapsed_s = perf_counter() - start
                if len(node_outputs) != len(node_requests):
                    raise RuntimeError(
                        f"Stage {node.node_id!r} returned {len(node_outputs)} outputs "
                        f"for {len(node_requests)} requests."
                    )
                for request, output in zip(node_requests, node_outputs, strict=True):
                    if prepare_metadata:
                        output.metadata.update(prepare_metadata)
                    records.append(self._make_batch_record(node, output, elapsed_s, len(node_requests)))
                outputs[node.node_id] = node_outputs
                completed.add(node.node_id)

        return StageBatchSchedulerResult(
            root_request_ids=[request.request_id for request in root_requests],
            records=records,
            final_outputs=[outputs[node.node_id] for node in self.graph.leaves()],
        )

    def _requests_for_node_batch(
        self,
        node: StageNode,
        root_requests: list[OmniRequest],
        requests: dict[str, list[OmniRequest]],
        outputs: dict[str, list[StageOutput]],
    ) -> list[OmniRequest]:
        if node.node_id in requests:
            return requests[node.node_id]

        in_edges = self.graph.in_edges(node.node_id)
        if len(in_edges) != 1:
            raise RuntimeError(
                f"Stage node {node.node_id!r} requires exactly one input edge in V1, got {len(in_edges)}."
            )
        edge = in_edges[0]
        source_outputs = outputs[edge.source]
        node_requests = [
            edge.connector.connect(
                ConnectorContext(
                    root_request=root_request,
                    source_node=edge.source,
                    target_node=edge.target,
                    source_output=source_output,
                )
            )
            for root_request, source_output in zip(root_requests, source_outputs, strict=True)
        ]
        requests[node.node_id] = node_requests
        return node_requests

    def _make_batch_record(
        self,
        node: StageNode,
        output: StageOutput,
        elapsed_s: float,
        batch_size: int,
    ) -> StageExecutionRecord:
        metadata = dict(output.metadata)
        # The stage ran once for the whole batch, so per-request wall time is
        # not observable; record the batch-level figure on every request.
        metadata["elapsed_s"] = elapsed_s
        metadata["batch_size"] = batch_size
        metadata["paradigm"] = node.stage.paradigm.value
        in_edges = self.graph.in_edges(node.node_id)
        if in_edges:
            metadata.setdefault("source_node", in_edges[0].source)
            metadata.setdefault("source_request_id", output.request_id)
        return StageExecutionRecord(
            node_id=node.node_id,
            stage_name=node.stage.name,
            request_id=output.request_id,
            metadata=metadata,
        )

    def _request_for_node(
        self,
        node: StageNode,
        root_request: OmniRequest,
        requests: dict[str, OmniRequest],
        outputs: StageResultStore,
    ) -> OmniRequest:
        if node.node_id in requests:
            return requests[node.node_id]

        in_edges = self.graph.in_edges(node.node_id)
        if len(in_edges) != 1:
            raise RuntimeError(
                f"Stage node {node.node_id!r} requires exactly one input request in V1, got {len(in_edges)}."
            )
        edge = in_edges[0]
        context = ConnectorContext(
            root_request=root_request,
            source_node=edge.source,
            target_node=edge.target,
            source_output=outputs.get(edge.source),
        )
        request = edge.connector.connect(context)
        requests[node.node_id] = request
        return request

    def _materialize_downstream_requests(
        self,
        node: StageNode,
        root_request: OmniRequest,
        requests: dict[str, OmniRequest],
        outputs: StageResultStore,
    ) -> None:
        for edge in self.graph.out_edges(node.node_id):
            if edge.target in requests:
                continue
            context = ConnectorContext(
                root_request=root_request,
                source_node=edge.source,
                target_node=edge.target,
                source_output=outputs.get(edge.source),
            )
            requests[edge.target] = edge.connector.connect(context)

    @staticmethod
    def _run_stage(stage: Stage, request: OmniRequest) -> tuple[StageOutput, float]:
        prepare_metadata = stage.prepare()
        start = perf_counter()
        output = stage.run(request)
        elapsed_s = perf_counter() - start
        if prepare_metadata:
            output.metadata.update(prepare_metadata)
        return output, elapsed_s

    def _make_record(
        self,
        node: StageNode,
        output: StageOutput,
        elapsed_s: float,
        root_request: OmniRequest,
    ) -> StageExecutionRecord:
        metadata = dict(output.metadata)
        metadata["elapsed_s"] = elapsed_s
        metadata["paradigm"] = node.stage.paradigm.value
        in_edges = self.graph.in_edges(node.node_id)
        if in_edges:
            metadata.setdefault("source_node", in_edges[0].source)
            metadata.setdefault("source_request_id", root_request.request_id)
        return StageExecutionRecord(
            node_id=node.node_id,
            stage_name=node.stage.name,
            request_id=output.request_id,
            metadata=metadata,
        )
