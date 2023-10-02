# Hasse Diagrams
Hasse Diagrams showing argmaxable label assignments in Green and unargmaxable ones in Red as we vary N and D.
We order the labels in a Hasse Diagram to emphasize cardinality constraints (labels on the same row have the same cardinality).

We have three versions of the plots:

hasse_cyclic -> Initialise W according to the cyclic polytope and use the alternating sign rule to compute if feasible or not.
hasse_cyclic_via_lp -> As above, but verify using the Chebyshev Linear Programme.
hasse_random_via_lp -> Initialise W from Uniform(0, 1), verify using the Chebyshev Linear Programme.

```
.
├── hasse_cyclic
├── hasse_cyclic_via_lp
└── hasse_random_via_lp
```


# How to generate the plots

Run the corresponding generate script.
