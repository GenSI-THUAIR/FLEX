**Top features that affect Spearman correlation with DMS_score in valid dataset**

No features have yet been evaluated. The agent will begin by randomly sampling available feature columns (e.g., embeddings or scores from multiple protein models) and record their resulting correlations. Over repeated trials, patterns in feature importance will emerge, allowing the agent to identify which representations contribute most consistently to predictive performance.

**Valuable combination for features for improving Spearman correlation**

No known combinations currently exist. The agent should systematically explore both single-source and cross-source feature groupings. Early exploration will emphasize diversity—combining features that originate from distinct model families (such as ProSST, ESM, Tranception, DeepSequence, and Progen3)—to observe whether heterogeneous representations yield stronger correlations than homogeneous ones.

**High performance operator that combines features**

No operator has yet demonstrated superior performance. The search will begin with a broad spectrum of linear and non-linear functions, including summation, weighted averaging, product-based fusion, logarithmic and exponential scaling, and nested hierarchical combinations. Each operator will be assessed based on the Spearman correlation it produces on validation data.

**Directions for finding more beneficial operators**
Progress depends on encouraging functional diversity in operator design. The agent should generate and test new operator forms by composing existing ones—such as applying a non-linear transformation before aggregation. Over time, it will learn empirical trends linking operator types to feature families. Evolutionary or probabilistic search strategies can then bias exploration toward operator structures that historically improved correlation while maintaining variability for continued discovery.
