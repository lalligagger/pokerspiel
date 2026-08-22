# pokerspiel/app_solver.py
from open_spiel.python.algorithms import mccfr
from pokerspiel.data.flop_subsets import FlopSubsetSampler

class DecoupledExternalSolver(mccfr.ExternalSamplingSolver):
    def __init__(self, game):
        super().__init__(game)
        self.sampler = FlopSubsetSampler()

    def custom_cfr_loop(self, state, player_id):
        """
        Custom tree traversal walker that intercepts chance nodes 
        to apply the PioSolver 25-flop constraints.
        """
        # 1. Base terminal node cleanup
        if state.is_terminal():
            return state.returns()[player_id]

        # 2. Intercept Flop Dealing Node
        # Check if it is a chance node AND no public board cards exist yet
        if state.is_chance_node() and len(state.public_cards()) == 0:
            # Sample a landmark board from our 25 list
            flop_tuple, importance_weight = self.sampler.sample_board()
            
            # Create a localized deep copy of the tree state so we don't pollute other paths
            subgame_state = state.clone()
            
            # Force apply the 3 cards using your PokerKit/OpenSpiel string bindings
            for card_str in flop_tuple:
                # Map 'As' string directly to OpenSpiel action token
                action_id = self.pokerkit_card_to_action_id(card_str)
                subgame_state.apply_action(action_id)
                
            # Continue the traversal down the newly forced postflop subgame path
            raw_postflop_ev = self.custom_cfr_loop(subgame_state, player_id)
            
            # 🌟 ELIMINATE BIAS: Scale the postflop EV by the importance sampling weight
            # before passing it back up to your preflop strategy nodes!
            return raw_postflop_ev * importance_weight

        # 3. Standard OpenSpiel Player/Chance Logic Fallback
        # If it's preflop or a standard node, fall back to native OpenSpiel behavior
        if state.is_chance_node():
            # Standard random sampling for preflop hole cards
            action = np.random.choice(state.legal_actions(), p=state.chance_outcomes())
            new_state = state.clone()
            new_state.apply_action(action)
            return self.custom_cfr_loop(new_state, player_id)
            
        # Add your standard player action regret matching loop below...
        # (Copy the rest of OpenSpiel's default external_sampling walk logic here)
