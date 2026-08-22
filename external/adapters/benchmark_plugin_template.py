"""Template only: copy outside the frozen core and bind to an official evaluator."""
class BenchmarkPlugin:
    suite='UNSET'
    def problems(self): raise NotImplementedError
    def bounds(self, problem, dimension): raise NotImplementedError
    def evaluate(self, problem, dimension, x):
        """Return (objective, constraint_vector) from the official evaluator."""
        raise NotImplementedError
