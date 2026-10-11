# robust_laplacian point-cloud bias — tables

Uniform clouds of the main experiment, mean of 3 trials. Reference eigenvalues: analytic (square, disk), P2-FEM h=0.01 (square_hole).

## 1. sum(diag M) vs true area (n_neighbors=30)

| shape | N | true_area | mass_sum | mass_ratio |
|---|---|---|---|---|
| disk | 1000 | 3.1416 | 3.1008 | 0.9870 |
| disk | 4000 | 3.1416 | 3.1222 | 0.9938 |
| disk | 8000 | 3.1416 | 3.1287 | 0.9959 |
| square | 1000 | 1.0000 | 0.9903 | 0.9903 |
| square | 4000 | 1.0000 | 0.9935 | 0.9935 |
| square | 8000 | 1.0000 | 0.9954 | 0.9954 |
| square_hole | 1000 | 0.8743 | 0.8919 | 1.0201 |
| square_hole | 4000 | 0.8743 | 0.8774 | 1.0035 |
| square_hole | 8000 | 0.8743 | 0.8738 | 0.9994 |

## 2. low-mode bias mean_{i=1..10}(λ_i/λ_i^ref − 1): original M vs M' = M·area/sum(M)

| shape | N | bias | bias_mass_rescaled |
|---|---|---|---|
| disk | 1000 | -0.0536 | -0.0659 |
| disk | 4000 | -0.0472 | -0.0531 |
| disk | 8000 | -0.0471 | -0.0510 |
| square | 1000 | -0.0459 | -0.0551 |
| square | 4000 | -0.0484 | -0.0546 |
| square | 8000 | -0.0475 | -0.0519 |
| square_hole | 1000 | -0.0303 | -0.0109 |
| square_hole | 4000 | -0.0437 | -0.0403 |
| square_hole | 8000 | -0.0465 | -0.0471 |

## 2b. energy ratios for smooth test functions: fᵀLf/∫|∇f|², fᵀMf/∫f², xᵀLx/area

| shape | N | stiffness_ratio | mass_fn_ratio | linear_energy_ratio |
|---|---|---|---|---|
| disk | 1000 | 0.9346 | 0.9870 | 0.9412 |
| disk | 4000 | 0.9509 | 0.9931 | 0.9515 |
| disk | 8000 | 0.9543 | 0.9954 | 0.9542 |
| square | 1000 | 0.9469 | 0.9881 | 0.9456 |
| square | 4000 | 0.9509 | 0.9924 | 0.9496 |
| square | 8000 | 0.9543 | 0.9946 | 0.9531 |
| square_hole | 1000 | 0.9624 | 1.0139 | 0.9590 |
| square_hole | 4000 | 0.9563 | 1.0015 | 0.9547 |
| square_hole | 8000 | 0.9551 | 0.9981 | 0.9537 |

## 3. low-mode bias vs n_neighbors

| shape | N | nn=10 | nn=20 | nn=30 | nn=50 |
|---|---|---|---|---|---|
| disk | 1000 | -0.2054 | -0.0658 | -0.0536 | -0.0522 |
| disk | 4000 | -0.1680 | -0.0537 | -0.0472 | -0.0471 |
| disk | 8000 | -0.1698 | -0.0528 | -0.0471 | -0.0472 |
| square | 1000 | -0.1597 | -0.0540 | -0.0459 | -0.0464 |
| square | 4000 | -0.1746 | -0.0573 | -0.0484 | -0.0483 |
| square | 8000 | -0.1681 | -0.0543 | -0.0475 | -0.0471 |
| square_hole | 1000 | -0.1536 | -0.0434 | -0.0303 | -0.0220 |
| square_hole | 4000 | -0.1686 | -0.0515 | -0.0437 | -0.0395 |
| square_hole | 8000 | -0.1739 | -0.0553 | -0.0465 | -0.0434 |

## 4. self-calibrated stiffness L' = c·L, c = sum(M) / mean(xᵀLx, yᵀLy)

| shape | N | calib_factor | bias | bias_calibrated |
|---|---|---|---|---|
| disk | 1000 | +1.0487 | -0.0536 | -0.0076 |
| disk | 4000 | +1.0445 | -0.0472 | -0.0048 |
| disk | 8000 | +1.0437 | -0.0471 | -0.0055 |
| square | 1000 | +1.0473 | -0.0459 | -0.0007 |
| square | 4000 | +1.0462 | -0.0484 | -0.0044 |
| square | 8000 | +1.0444 | -0.0475 | -0.0053 |
| square_hole | 1000 | +1.0637 | -0.0303 | +0.0314 |
| square_hole | 4000 | +1.0511 | -0.0437 | +0.0052 |
| square_hole | 8000 | +1.0479 | -0.0465 | -0.0009 |
