# Results

## Experiments on Nvidia T600 laptop GPU

**Timing in seconds** 
|  | jinns | DeepXDE - JAX | DeepXDE - Pytorch | PINA | Nvidia Modulus |
|---|:---:|:---:|:---:|:---:|:---:|
| Burgers1D | **445** | 723 | 671 | 1977 | 646 |
| NS2d-C | **265** | 278 | 441 | 1600 | 275 |
| PInv | 149 | 218 | *CC* | 1509 | **135** |
| Diffusion-Reaction-Inv | **284** | *NI* | 3424 | 4061 | 2541 |
| Navier-Stokes-Inv | **175** | *NI* | 1511 | 1403 | 498 |

**L1 relative error rate**
|  | jinns | DeepXDE - JAX | DeepXDE - Pytorch | PINA | Nvidia Modulus |
|---|:---:|:---:|:---:|:---:|:---:|
| Burgers1D | 0.013 | **0.011** | 0.012 | 0.040 | 0.019 |
| NS2d-C | **0.031** | 0.064 | 0.070 | 0.385 | 0.076 |
| PInv | 0.070 | 0.170 | *CC* | 0.067 | **0.034** |
| Diffusion-Reaction-Inv | (0.016,0.013) | *NI* | **(0.005,0.017)** | (0.024,0.028) | (0.018,0.084) |
| Navier-Stokes-Inv | (0.007,0.015) | *NI* | **(0.00,0.010)** | (0.001,0.054) | (0.024,0.027) |


**L2 relative error rate**
|  | jinns | DeepXDE - JAX | DeepXDE - Pytorch | PINA | Nvidia Modulus |
|---|:---:|:---:|:---:|:---:|:---:|
| Burgers1D | 0.050 | 0.035 | **0.020** | 0.147 | 0.101 |
| NS2d-C | **0.064** | 0.105 | 0.105 | 0.623 | 0.125 |
| PInv | 0.088 | 0.262 | *CC* | 0.083 | **0.043** |
| Diffusion-Reaction-Inv | (0.016,0.013) | *NI* | **(0.005,0.017)** | (0.024,0.028) | (0.018,0.084) |
| Navier-Stokes-Inv | (0.007,0.015) | *NI* | **(0.00,0.010)** | (0.001,0.054) | (0.024,0.027) |


*CC*: code crash / *NI*: Not Implemented in the specified backend
