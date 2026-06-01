1. Relative risk aversion is good but still gives some un-intuitive results. Consider this portfolio with `a=2.75` and 50% drawdown:

```
Name      Cost P(win) Edge   Full_Kelly Sized 
ContractA 0.85 0.93   +8.0¢  9.9%       17.3% 
ContractB 0.91 0.97   +6.0¢  2.0%       21.7% 
ContractC 0.83 0.97   +14.0¢ 62.1%      40.4% 
ContractD 0.63 0.78   +15.0¢ 26.0%      13.9% 
```

Contracts B and C have the same P(win), but C is much cheaper. It makes sense to still stay diversified, but I would expect that the edge would make us lean more heavily towards C than it is (but maybe not, we do have almost twice the sized stake).

2. The sizing policy dashboard needs to be simplified. Let's only have manual mode, user selects Ruin=drawdown definition and relative risk aversion. No Hard exposure cap, always whole contracts only, no "Max acceptable risk," no "Scale to cash" mode.

3. The graph under the 2^n enumeration Risk of Ruin percentage is uninterpretable. I want to see an actual probability density function graph with portfolio size on the X-axis.

4. Come up with some way to link correlated contracts. In the above example, ContractA cannot be True and ContractD false (although ContractD can be True and ContractA false). 
