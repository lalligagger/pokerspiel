## OpenSpiel CFR Architecture & Reduced-Depth CFR Feasibility

### **Current Architecture Overview**

OpenSpiel has a well-structured CFR implementation with **three main components**:

#### 1. **Full CFR** (`open_spiel/algorithms/cfr.h` and `cfr.cc`)
- Base class: `CFRSolverBase` with recursive tree traversal
- **Core recursion**: `ComputeCounterFactualRegret()` → `ComputeCounterFactualRegretForActionProbs()`
- Key methods:
  - Lines 331-408: `ComputeCounterFactualRegret()` - full depth traversal
  - Lines 443-469: `ComputeCounterFactualRegretForActionProbs()` - action evaluation
- **No depth tracking** - recurses until terminal states
- Information states stored in `CFRInfoStateValuesTable` with cumulative regrets/policies

#### 2. **MCCFR Variants** (Python implementations)
- **External Sampling** (`external_sampling_mccfr.py`): Samples actions at opponent nodes, walks all actions at player's nodes
- **Outcome Sampling** (`outcome_sampling_mccfr.py`): Samples single trajectory to terminal
- Both use `_episode()` recursion without depth limiting

#### 3. **No Existing Reduced-Depth Implementation**
- **FSICFR** (Fixed Strategy Iteration CFR) exists but isn't reduced-depth CFR
- **JAX CFR** has `max_depth` parameter but only for state space organization, not for early termination
- No value approximation at leaf nodes anywhere in the codebase

---

### **Difficulty Assessment: Moderate to High** (~2-4 weeks)

**Why harder than poker_ai:**

1. **C++ full CFR is deeply recursive** (game tree traversal is core logic)
2. **MCCFR implementations also need updating** (Python needs separate changes)
3. **Value function design is non-trivial** - no existing infrastructure
4. **Testing is critical** - exploitability will degrade if approximation is poor

---

## **What You Need to Add**

### **Option A: Modify Full CFR (C++)**

```c++
// Key changes to open_spiel/algorithms/cfr.h

class CFRSolverBase {
  // ... existing code ...
  
  // NEW: Constructor parameter for max depth
  CFRSolverBase(const Game& game, bool alternating_updates,
                bool linear_averaging, bool regret_matching_plus,
                int max_depth = -1);  // -1 = unlimited (default)

protected:
  int max_depth_;  // -1 means full tree traversal
  
private:
  // NEW: Overloaded version with depth tracking
  std::vector<double> ComputeCounterFactualRegret(
      const State& state, const absl::optional<int>& alternating_player,
      const std::vector<double>& reach_probabilities,
      const std::vector<const Policy*>* policy_overrides,
      int current_depth = 0);  // NEW parameter
  
  // NEW: Leaf value approximator (interface)
  virtual std::vector<double> ApproximateLeafValue(
      const State& state, int current_player);
};
```

**Required changes in `cfr.cc`:**

1. **Line 331-408** - Add depth check before recursion:
   ```c++
   if (current_depth >= max_depth_ && max_depth_ > 0) {
     return ApproximateLeafValue(state, current_player);
   }
   ```

2. **Implement default approximator** (simple rollout or random):
   ```c++
   std::vector<double> CFRSolverBase::ApproximateLeafValue(
       const State& state, int current_player) {
     // Option 1: Random rollout
     // Option 2: Return uniform value (0 for zero-sum)
     // Option 3: Heuristic eval function
   }
   ```

### **Option B: Modify MCCFR (Python)**

For `outcome_sampling_mccfr.py` and `external_sampling_mccfr.py`:

```python
class OutcomeSamplingSolver(mccfr.MCCFRSolverBase):
  def __init__(self, game, max_depth=-1, value_approximator=None):
    super().__init__(game)
    self.max_depth = max_depth
    self.value_approximator = value_approximator or self._default_approximator
  
  def _episode(self, state, update_player, my_reach, opp_reach, 
               sample_reach, current_depth=0):
    # Line 75: Add early exit
    if self.max_depth > 0 and current_depth >= self.max_depth:
      return self.value_approximator(state, update_player)
    
    # ... rest of existing code ...
    # Increment depth in recursive calls
```

---

### **Implementation Work Breakdown**

| Task | Effort | Complexity | Notes |
|------|--------|-----------|-------|
| **Add depth parameter** | 2-4 hours | Easy | Add constructor arg, member variable |
| **Add depth check in recursion** | 1-2 hours | Easy | One if-statement at recursion entry |
| **Implement leaf value approximator** | 4-8 hours | Medium | Multiple options (rollout, heuristic, NN) |
| **Regret propagation from approximated nodes** | 2-4 hours | Medium | Ensure values flow back correctly |
| **Update MCCFR Python variants** | 4-6 hours | Medium | Same logic in 2 implementations |
| **Testing & tuning** | 8-20 hours | High | Convergence tests, exploitability metrics |
| **Documentation** | 2-4 hours | Easy | Comments + usage examples |

**Total: 2.5 - 3.5 weeks for a solid implementation**

---

## **Critical Design Decisions**

### **1. Value Approximation Strategy**

**Best for poker (in order):**
- **Uniform (0 for zero-sum)** - Simplest, surprisingly competitive
- **Random rollout** - 10-50 moves to terminal, biased but informative  
- **Neural network eval** - Requires pre-training, high variance initially
- **Hand strength heuristic** - Fast, but poker-specific

### **2. Depth Tuning**

For poker (typical game tree depth ~30 decisions per player):
- **Shallow (5-7)**: Training fast but exploitable (~200+ BB/100)
- **Medium (10-15)**: Good balance (~50-100 BB/100 gap)
- **Full**: Optimal but slow convergence

### **3. Importance Sampling Correction**

When you truncate at depth `d`:
- Regrets from shallow nodes have **lower weight** than full tree
- You'll need to apply **inverse depth weighting** to avoid bias
- Or use **adaptive depth** that increases with iteration

---

## **Why This Isn't Trivial**

1. **OpenSpiel's recursion is fundamental** - not an optional component like poker_ai's pruning
2. **Value approximation affects convergence** - bad approximator makes training diverge
3. **MCCFR has sampling variance** - truncation + sampling = high variance if not careful
4. **Poker needs care** - exploitability metric is your ground truth (run best response evaluations)

---

## **Recommended Path Forward**

1. **Start with Python MCCFR** (outcome sampling)
   - Easier to iterate
   - Poker-specific tests already exist
   - Can prototype value approximators quickly

2. **Simple approximator first**
   - Uniform value (0 for zero-sum games)
   - Or random rollout (10 moves)
   - Measure convergence/exploitability vs. full depth

3. **Then optimize**
   - Add depth scheduling (increase max_depth over iterations)
   - Try better value functions
   - Backport to C++ if needed for speed

Does this help clarify the work? Want me to outline actual code changes for the Python MCCFR path?
