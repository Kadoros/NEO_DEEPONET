## span_loss (mean)

split                            test_in                                        test_out                                    
set                          quasi_16384 quasi_8192 random_16384 random_8192 quasi_16384 quasi_8192 random_16384 random_8192
model                                                                                                                       
main t2048 (2048-pt targets)    0.000513   0.000759     0.000758     0.00128      0.0018    0.00222      0.00208      0.0029
main t4096 (4096-pt targets)    0.000271   0.000518     0.000489    0.000993     0.00119    0.00167       0.0015     0.00237
pilot ft_lr3e-4 (1 h)           0.000951    0.00125       0.0012     0.00185     0.00139    0.00174      0.00165      0.0024
surface Large (zero-shot)          0.003    0.00333      0.00349       0.005     0.00362    0.00418      0.00418     0.00592

## eig_relerr (mean)

split                            test_in                                        test_out                                    
set                          quasi_16384 quasi_8192 random_16384 random_8192 quasi_16384 quasi_8192 random_16384 random_8192
model                                                                                                                       
main t2048 (2048-pt targets)      0.0123     0.0158       0.0246      0.0315       0.027     0.0327       0.0418      0.0513
main t4096 (4096-pt targets)     0.00908     0.0129       0.0195      0.0261      0.0204     0.0272       0.0339      0.0445
pilot ft_lr3e-4 (1 h)             0.0149      0.019       0.0282      0.0371      0.0238       0.03        0.039      0.0492
surface Large (zero-shot)          0.024     0.0306       0.0413       0.063      0.0339      0.045       0.0542      0.0808

## eig_relerr_1to10 (mean)

split                            test_in                                        test_out                                    
set                          quasi_16384 quasi_8192 random_16384 random_8192 quasi_16384 quasi_8192 random_16384 random_8192
model                                                                                                                       
main t2048 (2048-pt targets)     0.00324    0.00377      0.00576     0.00722      0.0145     0.0155       0.0196      0.0224
main t4096 (4096-pt targets)     0.00211    0.00279      0.00421      0.0057     0.00923      0.011       0.0137      0.0175
pilot ft_lr3e-4 (1 h)            0.00449    0.00512      0.00717     0.00899      0.0126     0.0141       0.0182      0.0215
surface Large (zero-shot)         0.0104     0.0127       0.0161      0.0271      0.0287     0.0344       0.0383      0.0536

## span_loss by family (test_in, all sets)

family                            cup  elongated  multihole  sh_ball    shell  superquadric   torus
model                                                                                              
main t2048 (2048-pt targets) 0.000865    0.00106   0.000795 0.000733 0.000614      0.000687 0.00103
main t4096 (4096-pt targets)  0.00057   0.000725   0.000508 0.000538 0.000446      0.000498 0.00069
pilot ft_lr3e-4 (1 h)        0.000912      0.002    0.00302 0.000752 0.000654      0.000719 0.00115
surface Large (zero-shot)     0.00382    0.00338     0.0038   0.0036  0.00342       0.00344 0.00446
