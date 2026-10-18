# Tier 2 observations: the large-reactor cases that solved

Commit 19ce0e1. docs/model.md A5. **Observations, not counted.** V1-V4 stay "not evaluable"; no pass rule is applied here.

## V3: core height

| Reference | Core height | Status | dp (kPa) | Published dp (kPa) | Difference | beta | Peak fuel (K) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| outlet | 80 cm | stable | 550.0 | 551.8 | -0.3% | 2.244 | 4296 |
| outlet | 100 cm | stable | 684.1 | 693.0 | -1.3% | 3.347 | 5101 |
| outlet | 120 cm | no solution below 5000 K | n/a | 865.7 | n/a | n/a | n/a |
| inlet | 80 cm | stable | 591.7 | 551.8 | +7.2% | 2.251 | 4298 |
| inlet | 100 cm | stable | 751.6 | 693.0 | +8.5% | 3.373 | 5110 |
| inlet | 120 cm | no solution below 5000 K | n/a | 865.7 | n/a | n/a | n/a |

## V4: frit-dominance variants (1.2 m)

| Reference | Variant | Status | Peak fuel (K) | dp (kPa) | Flow CoV |
| --- | --- | --- | --- | --- | --- |
| outlet | baseline | no solution below 5000 K | n/a | n/a | n/a |
| outlet | cold dominated | stable | 3613 | 2063.4 | 0.1155 |
| outlet | hot dominated | stable | 3671 | 2377.5 | 0.1290 |
| inlet | baseline | no solution below 5000 K | n/a | n/a | n/a |
| inlet | cold dominated | stable | 3627 | 2890.4 | 0.1216 |
| inlet | hot dominated | stable | 3697 | 3606.6 | 0.1397 |

## Limits
- Our gas temperatures run hotter than the Tsinghua papers' (NIST enthalpy), so cases near the 5000 K property limit fail for us that may solve for them.
- A missing case means the check cannot be judged. It is not evidence for or against the paper.
