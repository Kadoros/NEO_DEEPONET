## span_loss (mean)

split                        test_in                                      test_out                                   
set                       quasi_2048 quasi_8192 random_2048 random_8192 quasi_2048 quasi_8192 random_2048 random_8192
model                                                                                                                
ft_lr1e-4                    0.00297   0.000745     0.00463     0.00136     0.0047    0.00147      0.0062     0.00213
ft_lr3e-4                    0.00281   0.000771     0.00426     0.00133    0.00493    0.00179     0.00638     0.00242
scratch_lr5e-4               0.00455    0.00111     0.00824     0.00203     0.0104    0.00526      0.0164     0.00629
surface Large (zero-shot)    0.00864    0.00337      0.0132     0.00487     0.0112    0.00426       0.015     0.00593

## eig_relerr (mean)

split                        test_in                                      test_out                                   
set                       quasi_2048 quasi_8192 random_2048 random_8192 quasi_2048 quasi_8192 random_2048 random_8192
model                                                                                                                
ft_lr1e-4                     0.0279     0.0155      0.0458      0.0329      0.045     0.0282      0.0635      0.0467
ft_lr3e-4                     0.0265     0.0155      0.0425      0.0322     0.0455     0.0305      0.0632      0.0489
scratch_lr5e-4                0.0381     0.0177      0.0694       0.039     0.0663     0.0367       0.108      0.0634
surface Large (zero-shot)     0.0625     0.0295        0.11      0.0596     0.0903     0.0447       0.137      0.0795

## eig_relerr_1to10 (mean)

split                        test_in                                      test_out                                   
set                       quasi_2048 quasi_8192 random_2048 random_8192 quasi_2048 quasi_8192 random_2048 random_8192
model                                                                                                                
ft_lr1e-4                    0.00661    0.00381      0.0126     0.00759     0.0243     0.0242      0.0362       0.031
ft_lr3e-4                    0.00618    0.00373      0.0113     0.00708     0.0262     0.0269      0.0381      0.0331
scratch_lr5e-4                0.0087    0.00529      0.0176     0.00941     0.0339     0.0332      0.0475      0.0409
surface Large (zero-shot)     0.0324     0.0123      0.0601       0.025     0.0639     0.0423      0.0946      0.0595

## span_loss by family (test_in, all sets)

family                        cup  sh_ball   shell  superquadric   torus
model                                                                   
ft_lr1e-4                 0.00258   0.0024 0.00189       0.00225 0.00301
ft_lr3e-4                 0.00241  0.00226 0.00179       0.00214 0.00287
scratch_lr5e-4             0.0046  0.00362 0.00287       0.00355 0.00528
surface Large (zero-shot) 0.00746  0.00732 0.00658       0.00726 0.00904
