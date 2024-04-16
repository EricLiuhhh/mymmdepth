from typing import Callable, Optional, Union
from mmengine.hooks import ProfilerHook
from mmdepth.registry import HOOKS

@HOOKS.register_module()
class DepthProfilerHook(ProfilerHook):
    def __init__(self,
                 *,
                 by_epoch: bool = True,
                 profile_times: int = 1,
                 activity_with_cpu: bool = True,
                 activity_with_cuda: bool = False,
                 schedule: Optional[dict] = None,
                 on_trace_ready: Union[Callable, dict, None] = None,
                 record_shapes: bool = False,
                 profile_memory: bool = False,
                 with_stack: bool = False,
                 with_flops: bool = False,
                 json_trace_path: Optional[str] = None) -> None:
        super().__init__(
            by_epoch=by_epoch, 
            profile_times=profile_times, 
            activity_with_cpu=activity_with_cpu, activity_with_cuda=activity_with_cuda, 
            schedule=schedule, 
            on_trace_ready=on_trace_ready, 
            record_shapes=record_shapes, 
            profile_memory=profile_memory, 
            with_stack=with_stack, 
            with_flops=with_flops, 
            json_trace_path=json_trace_path)
        
    def after_val_epoch(self, runner, metrics):
        """Determine if the content is exported."""
        # `after_train_epoch` will also be called in IterBasedTrainLoop.
        # Here we check `self._closed` to avoid exiting twice.
        if not self._closed:
            self._export_chrome_trace(runner)

    def after_val_iter(self, runner, batch_idx, data_batch, outputs):
        """profiler will call `step` method if it is not closed."""
        if not self._closed:
            self.profiler.step()
        if batch_idx == self.profile_times - 1 and not self.by_epoch:
            self._export_chrome_trace(runner)