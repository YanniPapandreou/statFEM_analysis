import functools
from typing import Callable, List

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
)


def inner_loop(func: Callable, progress: Progress, task_id: TaskID) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        value = func(*args, **kwargs)
        progress.update(task_id, advance=1)
        return value

    return wrapper


def outer_loop(args: List[dict], func: Callable) -> dict:
    res = {}
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
    ) as progress:
        total_steps = len(args)
        outer_task = progress.add_task("Outer loop", total=total_steps)
        last_inner_task = None
        assert len(args) != 0
        for i, arg in enumerate(args):
            sub_steps = arg["steps"]
            assert sub_steps is not None
            inner_task = progress.add_task(f"Subtask {arg = }", total=sub_steps)
            last_inner_task = inner_task
            res[arg] = inner_loop(func, progress, inner_task)(*arg)
            if i < total_steps - 1:
                progress.remove_task(inner_task)
                progress.refresh()
            progress.update(outer_task, advance=1)
        # Ensure both outer and last inner task are fully completed
        progress.update(outer_task, completed=total_steps)
        progress.update(last_inner_task, completed=sub_steps)
        # Replace spinner with a static checkmark by creating a new column list
        progress.columns = [
            TextColumn("✔ "),  # Checkmark instead of spinner
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
        ]
        progress.refresh()
    return res


if __name__ == "__main__":
    import time
    def refine(steps: int, h: float) -> List:
        out = []
        for i in range(steps):
            time.sleep(0.1)
            out.append((i,h))
        return out
    args = [
        {"steps": 10, "h": 0.1},
        {"steps": 20, "h": 0.5},
    ]
    res = outer_loop(args, refine)
    print(res)
