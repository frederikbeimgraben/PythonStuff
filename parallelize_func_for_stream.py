"""
Decorator to automatically parallelize a function.
It can be fed with a continuous stream of data and will also yield one.

Usage:
    @parallelize(
        cache = C,                  # Cache the results of the function (with a memory of <C: int>)
        max_workers = W,            # Use <W: int> workers
        break_on_except = (..., ),  # Break on exceptions contained in the tuple
                                    # -> by default all break the stream
        timed = True | False,       # Return the time it took to run a worker in the output stream
        ordered = True | False,     # Return the results in the order they were called
    )
    def your_function(arg1: A, arg2: B, arg3: C, kwarg1: D=..., kwarg2: E=...) -> Any:
        ...

    iter: Iterable[Tuple[A, B, D]]
    arg: C
    kwarg: E

    for result in your_function(iter, arg3, kwarg2=kwarg):
        ...
"""

import inspect
from typing import Any, Callable, List, Tuple, Iterable, Optional
import multiprocessing
import functools
import time
import dill


# A function that unpacks the pickled function and runs it
def _run(
        function_pickled: bytes,
        queue: multiprocessing.Queue,
        worker_id: int,
        *args,
        **kwargs) -> Tuple[float, Any]:
    """
    Unpacks a pickled function and runs it.
    """
    # Unpickle the function
    function = dill.loads(function_pickled)

    # Run the function
    start = time.time()
    try:
        result = function(*args, **kwargs)
        end = time.time()
        queue.put((worker_id, end - start, result,))
    # Make the Linter ignore this line
    # pylint: disable=broad-except
    except Exception as exc:
        end = time.time()
        queue.put((worker_id, end - start, exc,))


def parallelize(
        cache: int=0,
        max_workers: int=None,
        break_on_except: Iterable[Exception]=(Exception,),
        timed: bool=False,
        ordered: bool=False) -> Callable:
    """
    Decorator to automatically parallelize a function like:
        foo(A) -> B: ...

    =>  foo(Iterable[A]) -> Iterable[Tuple[{time}, B]]: ...

    This way it can handle continuous streams of data at a higher performance.
    """

    def eval_result(
            workers: List[multiprocessing.Process],
            queue: multiprocessing.Queue) -> Any:
        """
        Evaluate the result of a process.
        """

        elem: Any
        if ordered: # Pick the next worker
            worker = workers.pop(0)
            worker[1].join()
            elem = queue.get()

            # Looks bad, but it´s at most O(max_workers)
            while elem[0] != worker[0]:
                queue.put(elem)
                elem = queue.get()
        else: # Pick any worker, that is finished
            elem = queue.get()
            broken = None
            # Wait for a worker to finish
            while not broken:
                for worker_id, worker in workers:
                    if not worker.is_alive():
                        worker.join()
                        workers.remove((worker_id, worker,))
                        broken = worker
            del broken

        # Match the result
        match elem:
            # Make the linter ignore used before assignment
            # pylint: disable=used-before-assignment
            case (_, _, result,) if any(
                isinstance(result, exception)
                for exception in break_on_except):
                # If the result is an exception, raise it
                for _, worker in workers:
                    worker.terminate()
                raise result
            case (_, took, result,):
                # If the result is not an exception, return it
                return (took, result,) if timed else result

    def _wrapper(function: Callable) -> Callable:
        # Get the signature of the function
        signature = inspect.signature(function)

        # Get the parameter and return annotations
        param_annotation = Tuple[tuple(
            Optional[parameter.annotation]
            for parameter in signature.parameters.values()
        )]
        return_annotation = signature.return_annotation

        # This is the `replacement` function that will be returned
        def _new(
            input_stream: Iterable[param_annotation],
            *global_args,
            **global_kwargs) -> Iterable[return_annotation]:

            # If the input stream is not a generator of tuples, make it one
            input_stream = (
                (arg,) if not isinstance(arg, tuple) else arg
                for arg in input_stream
            )

            # Pickle the function
            _function = dill.dumps(
                functools.lru_cache(maxsize=cache)(function)
                if cache > 0 else function
            )

            # Create a list of workers
            workers: List[Tuple[int, multiprocessing.Process]] = []

            # Create an id counter to identify the workers for ordered mode
            worker_id = 0

            # Create a queue to communicate with the workers
            queue = multiprocessing.Queue()

            # manually create a pool using multiprocessing.Process
            # because the multiprocessing.Pool class does not support
            # returning continuously from a generator
            for args in input_stream:
                # Create a worker and start it
                worker = multiprocessing.Process(
                    target=_run,
                    args=(_function, queue, worker_id, *args, *global_args),
                    kwargs=global_kwargs
                )
                worker.start()
                workers.append((id, worker,))

                # Increment the id
                worker_id += 1

                # If our pool is full, yield a result
                while (
                    len(workers) >= (max_workers or multiprocessing.cpu_count())
                ):
                    yield eval_result(
                        workers,
                        queue
                    )

            # If there are still workers, yield the results
            while workers:
                yield eval_result(
                    workers,
                    queue
                )

        # Overwrite the docstring and name of the function
        _new.__doc__ = function.__doc__
        _new.__name__ = function.__name__

        # Return the new function
        return _new

    # Return the wrapper
    return _wrapper
