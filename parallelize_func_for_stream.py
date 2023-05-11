"""
Decorator to automatically parallelize a function.
It can be fed with a continuous stream of data and will also yield one.

Usage:
    @mappable(
        max_workers = W,            # Use <W: int> workers
        break_on_except = (..., ),  # Break on exceptions of types ... (tuple)
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


def __run(
        function_pickled: bytes,
        queue: multiprocessing.Queue,
        worker_id: int,
        *args,
        **kwargs) -> Tuple[float, Any]:
    """
    Unpacks a pickled function, runs it and puts the result into the queue
    This is necessary because multiprocessing can't pickle all functions and thus we have to pickle
    them manually using dill.
    The function is on module level, so it can be passed to the workers.
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


def async_map(
        max_workers: int=None,
        break_on_except: Iterable[Exception]=(Exception,),
        timed: bool=False,
        ordered: bool=False,
        timeout: int=5,
        wait: bool=True) -> Callable:
    """
    Decorator to automatically parallelize a function like:
        foo(A) -> B: ...

    =>  foo(Iterable[A]) -> Generator[Tuple[{time}, B]]: ...

    This way it can handle continuous streams of data at a higher performance.
    """

    

    def wrapper(function: Callable) -> Callable:
        """
        The Function Wrapper
        """
        
        def eval_result(
            workers: List[Tuple[int, multiprocessing.Process]],
            queue: multiprocessing.Queue) -> Any:
            """
            Evaluate the result of a process.
            """

            # Element to be processed
            elem: Any = None

            # Has it been found?
            found: bool = False

            try:
                if ordered: # Pick the next worker in order
                    worker_id, worker = workers.pop(0)
                    worker.join(timeout=timeout)
                    
                    elem = queue.get()
                    while not found:
                        if elem[0] == worker_id:
                            raise StopIteration()
                        else:
                            # Cycle through the results
                            queue.put(elem)
                            elem = queue.get()
                
                else: # Pick any worker, that is finished
                    start = time.time()
                    while not found and time.time() - start < timeout:
                        for worker_id, worker in workers:
                            if not worker.is_alive():
                                worker.join(timeout=timeout - (time.time() - start))
                                workers.remove((worker_id, worker,))
                                elem = queue.get()
                                raise StopIteration()
                    
                    # If it timed out, stop a worker.
                    worker_id, worker = workers.pop(0)
                    worker.join(timeout=0)
                    elem = queue.get()
            except StopIteration:
                pass

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
                case None:
                    return (-1, None,)

        # Get the signature of the function
        signature = inspect.signature(function)

        # Get the parameter and return annotations
        params = Tuple[tuple(
            Optional[parameter.annotation]
            for parameter in signature.parameters.values()
        )]
        returns = signature.return_annotation

        # This is the `replacement` function that will be returned
        def new(
            input_stream: Iterable[params],
            *global_args,
            **global_kwargs) -> Iterable[returns]:

            # Check if the input stream is iterable
            if not '__next__' in dir(input_stream):
                if '__iter__' in dir(input_stream):
                    input_stream = iter(input_stream)
                else:
                    raise TypeError(
                        'The input stream must implement the `__next__` or `__iter__` method'
                    )
                
            # Pickle the function
            _function = dill.dumps(function)

            # Create a list of workers
            workers: List[Tuple[int, multiprocessing.Process]] = []

            # Create an id counter to identify the workers for ordered mode
            worker_id = 0

            # Create a queue to communicate with the workers
            queue = multiprocessing.Queue()

            # Equivalent to a Pool
            # FIXME: Will be reworked to reuse workers in the future
            while True:
                try:
                    while True:
                        # Get the next arguments
                        # Will raise StopIteration if the stream is empty
                        # This will be caught by the outer try-except
                        # If we would use a for loop, it would catch the exception 
                        # which we don't want
                        args = next(input_stream)

                        # If the arguments are not a tuple, make them a tuple
                        args = (args, ) if not isinstance(args, tuple) else args

                        # Create a worker and start it
                        worker = multiprocessing.Process(
                            target=__run,
                            args=(_function, queue, worker_id, *args, *global_args),
                            kwargs=global_kwargs
                        )
                        worker.start()

                        # Add the worker to the list
                        workers.append((worker_id, worker,))

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
                
                # If the stream is empty, handle like:
                except StopIteration:
                    # If `wait` is not set, break
                    if queue.empty() and not wait:
                        break
                    # Otherwise yield the results and wait for new ones
                    if not queue.empty():
                        yield eval_result(
                            workers,
                            queue
                        )
                        continue

        # Overwrite the docstring and name of the function
        new.__doc__ = function.__doc__
        new.__name__ = function.__name__

        # Return the new function
        return new

    # Return the wrapper
    return wrapper
