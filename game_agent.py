"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import logging


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # TODO: finish this function!
    # raise NotImplementedError
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    return float(len(game.get_legal_moves(player)))


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=False, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left
        
        # returning immediately if there are no legal moves
        if not legal_moves:
            return (-1, -1)
        
        #Opening book with a centre move
        # if len(legal_moves) == game.width*game.height:
        #     return (int(game.width*0.5), int())

        #move placeholder
        current_best_move = (-1, -1)
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.iterative:
                tdepth = 1
                while True:
                    if self.method == 'minimax':
                        core, current_best_move = self.minimax(game=game, depth=tdepth,
                                                               maximizing_player=True)
                    elif self.method == 'alphabeta':
                        core, current_best_move = self.alphabeta(game=game, depth=tdepth,
                                                                 maximizing_player=True)
                    tdepth += 1
            else:
                if self.method == 'minimax':
                    core, current_best_move = self.minimax(game=game, depth=self.search_depth, maximizing_player=True)
                elif self.method == 'alphabeta':
                    core, current_best_move = self.alphabeta(game=game, depth=self.search_depth, maximizing_player=True)
        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        return current_best_move


        # raise NotImplementedError

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # check for the terminal nodes
        if game.is_winner(self):
            return self.score(game,self), game.get_player_location(self)
        elif game.is_loser(self):
            return self.score(game,self), game.get_player_location(self)

        #check if we reached the depth, return current score and move
        elif depth == 0:
            return self.score(game, self), game.get_player_location(self)

        legal_moves = game.get_legal_moves()

        if not legal_moves:
            return self.score(game, self), game.get_player_location(self)

        else:
            #return max/min of all move scores by decreasing the depth
            best_score = None
            best_action = None
            for action in legal_moves:
                if maximizing_player:
                    temp_score, _ = self.minimax(game=game.forecast_move(action),depth=depth-1, maximizing_player=False)
                    if (best_score==None or temp_score > best_score):
                        best_score = temp_score
                        best_action = action
                if not maximizing_player:
                    temp_score, _ = self.minimax(game=game.forecast_move(action),depth=depth-1, maximizing_player=True)
                    if (best_score==None or temp_score < best_score):
                        best_score = temp_score
                        best_action = action
            return best_score, best_action


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # check for the terminal nodes
        if game.is_winner(self):
            return self.score(game,self), game.get_player_location(self)
        elif game.is_loser(self):
            return self.score(game,self), game.get_player_location(self)

        #check if we reached the depth, return current score and move
        elif depth == 0:
            return self.score(game, self), game.get_player_location(self)

        legal_moves = game.get_legal_moves()

        if not legal_moves:
            return self.score(game, self), game.get_player_location(self)

        else:
            #return max/min of all move scores by decreasing the depth
            best_score = None
            best_action = None
            for action in legal_moves:
                if maximizing_player:
                    temp_score, _ = self.alphabeta(game=game.forecast_move(action),depth=depth-1,
                                                   alpha=alpha, beta=beta,maximizing_player=False)

                    if (best_score==None or temp_score > best_score):
                        best_score = temp_score
                        best_action = action

                    if temp_score >= beta:
                        return temp_score, action

                    alpha = max([alpha, temp_score])


                if not maximizing_player:
                    temp_score, _ = self.alphabeta(game=game.forecast_move(action),depth=depth-1,
                                                   alpha=alpha, beta=beta,maximizing_player=True)
                    if (best_score==None or temp_score < best_score):
                        best_score = temp_score
                        best_action = action

                    if temp_score <= alpha:
                        return temp_score, action

                    beta = min([beta, temp_score])

            return best_score, best_action
