"""The README quick start: a three-product instance given by its coefficients."""
import robust_mckp as rm

# Option j of group i: revenue v, nominal margin surplus s, endpoint loss t >= 0.
# A decision is feasible if  sum_i s_i - (sum of the Gamma largest t_i) >= 0.
inst = rm.instance_from_arrays(
    values=[[10, 12, 13], [8, 9, 11], [5, 7, 8]],
    margins=[[4, 2, 1], [3, 2, 0], [2, 1, -1]],
    losses=[[1, 2, 3], [1, 1, 2], [0.5, 1, 2]],
    gamma=1,
)

res = rm.solve(inst)                       # optimal decision (threshold DnC)
print(res.status, res.selection, res.revenue, res.exact.feasible)

res = rm.solve(inst, method="certified")   # HullRound's certified interval, fast
print(res.revenue, "<= OPT <=", res.upper_bound, "gap", res.gap)

print(rm.verify(inst, res.selection))      # exact rational re-check
cert = rm.certify(inst, res.selection)     # exact global certificate (Theorem 12)
assert rm.verify_global_certificate(inst, cert).valid
print("exact certificate: H =", cert.h, " U =", cert.u)
