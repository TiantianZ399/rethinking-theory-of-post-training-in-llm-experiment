"""TODO: Real coverage-performance curve.

Steps:
1. Load prompts and SFT policy.
2. Generate large pool P_M per prompt.
3. Score pool with RM and policy scorer.
4. Approximate q_lambda^* on P_M.
5. Select smaller supports C_K by random/diverse/top/submodular.
6. Compute rho(C_K).
7. Run target matching on selected supports.
8. Plot rho vs performance.
"""

if __name__ == "__main__":
    raise SystemExit("TODO: implement real LLM coverage experiment")
