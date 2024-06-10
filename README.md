Idea: should somehow be able to interact with result (improve it efficiently)

estimator = Estimator()
estimator(A) -> trace estimate, error estimate, ...
estimator.improve(10)

- Also for matrix handles


Almost all randomized algorithms are additive, meaning we can refine the estimate.