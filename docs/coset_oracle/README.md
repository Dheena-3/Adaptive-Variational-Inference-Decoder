# Logical-coset oracle (d = 3)

Exact syndrome / logical-class probabilities P(s, l), the Bayes-optimal logical decision, and the
same-syndrome ambiguity catalogue for the rotated d=3 surface code. Offline oracle and scorer only:
it takes the 21 fault probabilities (the model), never sampled errors or logical labels.

## One command

From the repo root:

    python3 -m simulation.reference.noise_model.coset_oracle --scenario S2 --check

Expected (excerpt):

    Bayes-optimal logical error rate (LER_oracle) = 5.708 %
    exact ties: []
    logically ambiguous syndromes: ['0x1', '0x2', '0x4', '0x8']
    [check] DP vs brute force over 2^21 subsets: max |diff| = 9.23e-14  OK
    [check] stabilizer group size 16, 32 cosets of sizes [16], bijection to (s, l) classes: OK

Scenarios: `S1` uniform p=0.01, `S2` Z4Z5 burst 0.08, `S3` Z3 burst 0.08 (ambiguity study, placeholder rates);
`R0` stationary, `R1` hot pair, `R3` hot single (independent validation report, as reported by its author).
Use `--basis Z` for X-type faults, `--json PATH` for one scenario's full report.

## Fixtures (`simulation/fixtures/coset/`)

Regenerate: `python3 -m simulation.reference.noise_model.coset_oracle --export-fixtures simulation/fixtures/coset`

| File | Content |
|---|---|
| `mechanisms.json` | 21 mechanisms per basis: support, syndrome signature, logical bit, class id (= 2*s + l) |
| `alias_catalogue.json` | per syndrome, which mechanisms have l = 0 vs l = 1; `ambiguous` if both |
| `scenarios.json` | per scenario: theta, P(s,l) as 16x2, P(s), P(l=1 given s), MAP bit, irreducible error, ties, LER_oracle |
| `opposite_truth_fixture.json` | S2 vs S3: identical P(s), opposite optimal decision at 0x4, cost of the wrong decision table |

Key fixture values: P(l=0 | 0x4) = 0.824 (S2) vs 0.176 (S3); LER_oracle = 5.708 % in both; wrong tuning = 12.04 %;
93.2 % of the extra error is at syndrome 0x4.

## Tests

    pytest tests/unit/test_coset_oracle.py -v        # 26 tests, ~2 s, no Verilator needed

Exact DP vs independent 2^21 enumeration (both bases), 512 patterns -> 32 classes of 16 (stabilizer cosets),
alias catalogue (0x4 = {Z3, Z6} with l=1 and {Z4Z5, Z7Z8} with l=0), opposite-truth fixture, committed fixtures
match a fresh export, and the oracle's R0 / R1 / R3 optimal LERs match the independent report (2.53 / 4.05 / 3.20 %).

## What this evidence establishes, and what it does not

Establishes: the exact joint distribution and optimal-decision bound for the d=3 model, checked two independent
ways; the logical ambiguity is real and confined to syndromes 0x1, 0x2, 0x4, 0x8.

Does not establish: anything about d=5 (the oracle here is d=3 only); finite-shot behaviour; the calibration
estimator or any label path into the system (separate deliverable); closed-loop or RTL timing. The scenarios use
placeholder or report-supplied rates, not hardware-measured noise.
