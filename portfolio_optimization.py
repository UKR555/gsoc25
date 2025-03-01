import cvxpy as cp

class BaseOptimizationProgram:
    def __init__(self):
        self.variables = []
        self.constraints = []
        self.objective = None

    def add_variable(self, name, shape=1, lb=None, ub=None):
        var = cp.Variable(shape, name=name)
        if lb is not None:
            self.constraints.append(var >= lb)
        if ub is not None:
            self.constraints.append(var <= ub)
        self.variables.append(var)
        return var

    def add_constraint(self, constraint):
        self.constraints.append(constraint)

    def set_objective(self, sense, expr):
        if sense == "min":
            self.objective = cp.Minimize(expr)
        elif sense == "max":
            self.objective = cp.Maximize(expr)
        else:
            raise ValueError("Objective sense must be 'min' or 'max'")

    def solve(self, solver=cp.GUROBI):
        problem = cp.Problem(self.objective, self.constraints)
        result = problem.solve(solver=solver)
        return result, {var.name(): var.value for var in self.variables}


class LinearProgram(BaseOptimizationProgram):
    def set_objective(self, sense, coefficients):
        if len(coefficients) != len(self.variables):
            raise ValueError("Coefficient length must match variable count")
        objective_expr = sum(c * v for c, v in zip(coefficients, self.variables))
        super().set_objective(sense, objective_expr)


class ConvexProgram(BaseOptimizationProgram):
    def set_objective(self, sense, expr):
        if not isinstance(expr, cp.Expression):
            raise TypeError("Objective must be a cvxpy expression")
        super().set_objective(sense, expr)
