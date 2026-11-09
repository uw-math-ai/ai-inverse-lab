"""Exact rational checks for the coordinate examples in PROOF_AUDIT.md.

These calculations support the audit; they are not Lean proofs of the paper.
"""
from fractions import Fraction as F


def matvec(matrix, vector):
    return [sum(a * b for a, b in zip(row, vector, strict=True)) for row in matrix]


def dot(left, right):
    return sum(a * b for a, b in zip(left, right, strict=True))


def weighted_dot(mass, left, right):
    return sum(m * a * b for m, a, b in zip(mass, left, right, strict=True))


# Path b--i, boundary {b}, unit edge conductance, vertex masses (2,1), T=2.
# W sends controls (f(0,b),f(1,b)) to U(2) in value coordinates.
W = [[F(1, 2), F(1)], [F(1), F(0)]]
W_star = [[F(1, 2), F(1, 2)], [F(1), F(0)]]
vertex_mass = [F(2), F(1)]
control_mass = [F(2), F(2)]
for control in ([F(1), F(0)], [F(0), F(1)]):
    for state in ([F(1), F(0)], [F(0), F(1)]):
        assert weighted_dot(vertex_mass, matvec(W, control), state) == weighted_dot(
            control_mass, control, matvec(W_star, state)
        )

harmonic = [F(1), F(1)]
exact_control = [F(1), F(1, 2)]
assert matvec(W, exact_control) == harmonic
observed_adjoint = matvec(W_star, harmonic)
assert observed_adjoint == [F(1), F(1)]
correct_interior_moment = weighted_dot(control_mass, exact_control, observed_adjoint) - F(2)
printed_interior_moment = dot(exact_control, observed_adjoint) - F(2)
assert correct_interior_moment == F(1)
assert printed_interior_moment == F(-1, 2)
print("Equation (4.3): weighted formula =", correct_interior_moment,
      "; printed value-coordinate formula =", printed_interior_moment)

# Path b--i--j, boundary {b}, masses (1,2,3), unit conductances.
# Interior harmonic products span the constant vector, so project onto that line.
interior_mass = [F(2), F(3)]
ones = [F(1), F(1)]
euclidean_coefficient = dot(interior_mass, ones) / dot(ones, ones)
weighted_coefficient = weighted_dot(interior_mass, interior_mass, ones) / weighted_dot(
    interior_mass, ones, ones
)
assert euclidean_coefficient == F(5, 2)
assert weighted_coefficient == F(13, 5)
assert euclidean_coefficient != weighted_coefficient
print("Projection: Euclidean coefficient =", euclidean_coefficient,
      "; mass-weighted coefficient =", weighted_coefficient)
