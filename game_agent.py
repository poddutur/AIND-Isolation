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


def game_occupancy(game):
    """ This function returns the % of the game board occupied, in other words it returns
    how close we might be to the end game.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    Returns
    -------
    float
        Ratio of the number of  occupied / blocked spots on the board to total number of spots on the board.
    """

    blank_spaces = len(game.get_blank_spaces())
    total = game.width*game.height

    return (total-blank_spaces)/total


def potential_moves(game, spot):
    """ This function returns number of L shaped moves possible from a given spot

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    spot : (int, int)
        The coordinate pair (row, column) of the spot from which we are computing how many
        potential moves are possible.

    Returns
    -------
    int
        Returns the number of possible moves from a give spot given the arrangement of the board.
    """

    # list of all possible L shaped moves from a spot (ignore boundaries as we check in the black spaces anyways)
    all_moves = [(spot[0]+1, spot[1]+2),
                 (spot[0]+1, spot[1]-2),
                 (spot[0]-1, spot[1]+2),
                 (spot[0]-1, spot[1]-2),
                 (spot[0]+2, spot[1]+1),
                 (spot[0]+2, spot[1]-1),
                 (spot[0]-2, spot[1]+1),
                 (spot[0]-2, spot[1]-1),
                 ]

    num_potential_moves = 0
    for move in all_moves:
        #increment the number of pontential moves if the move is in current black spot
        if move in game.get_blank_spaces():
            num_potential_moves += 1

    return int(num_potential_moves)


def ratio_heuristic(game, player):
    """ This is a heuristic similar to the base heuristic (IM Improved) but more aggressive in the sense that
    it weights the opponents legal moves in a ratio to player's legals moves before taking the difference.
    The function outputs the number of player legal moves minus ratio number of opponent legal moves.

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
        The heuristic value as number of player legal moves minus ratio number of opponents legal moves.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    #I tested it for few ratios and selected 1.5
    ratio_number = 1.5

    return float(own_moves - ratio_number*opp_moves)


def corners_walls_heuristic(game, player):
    """ This is a heuristic weights a legal move to wall or corner of the board lower than other places.
      This comes into play after we are greater than initial_phase %
      into the game as defined by game_occupancy() method.

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
        The heuristic value as number of player legal moves minus ratio number of opponents legal moves. Each legal move
        is weighted equally before initial phase % game occupancy after which
        legal moves to walls and corners are weighted lower.
     """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))

    ratio_number = 1.5

    # this variable indicates after what % of board occupancy the weighting comes into play
    initial_phase = 60

    current_phase = game_occupancy(game)*100

    # defining walls list and corners list
    walls = [[(0, i) for i in range(game.width)],
             [(i, 0) for i in range(game.height)],
             [(game.width - 1, i) for i in range(game.width)],
             [(i, game.height - 1) for i in range(game.height)],
             ]

    corners = [(0, 0), (0, game.width-1), (game.height-1, 0), (game.height-1, game.width-1)]

    if current_phase > initial_phase:
        own_moves_modified = 0
        for move in own_moves:
            if move in corners:
                #corners are incremented by 7 instead of 10
                own_moves_modified += 10*0.7
            elif move in walls:
                # walls incremented by 9 instead of 10
                own_moves_modified += 10*0.9
            else:
                own_moves_modified += 10

        opp_moves_modified = 0
        for move in opp_moves:
            if move in corners:
                opp_moves_modified += 10*0.7
            elif move in walls:
                opp_moves_modified += 10*0.9
            else:
                opp_moves_modified += 10

        return float(own_moves_modified-opp_moves_modified)

    else:
        return float(len(own_moves)*10 - ratio_number*len(opp_moves)*10)


def one_step_ahead_heuristic(game, player):
    """ This is a heuristic more generic version of corners and walls heuristic, it
    weights a legal move by number of possible hops it can take from that spot, in other word we are
    looking at one step ahead of the game. Again this comes into play after the initial phase.

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
        The heuristic value as number of player legal moves minus ratio number of opponents legal moves.
        Each legal move is weighted equally until initial phase % game occupancy, after that it is weighted
        based number of possible hops it can take from that spot.
     """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))

    ratio_number = 1.5

    initial_phase = 60

    current_phase = game_occupancy(game)*100

    if current_phase > initial_phase:
        own_moves_modified = 0
        for move in own_moves:
            #dividing the number of potential hops by 8 (as 8 is the maximum from a spot)
            move_factor = potential_moves(game, move)/8.0
            own_moves_modified += 3+7*move_factor

        opp_moves_modified = 0
        for move in opp_moves:
            move_factor = potential_moves(game, move)/8.0
            own_moves_modified += 3+7*move_factor

        return float(own_moves_modified-opp_moves_modified)

    else:
        return float(len(own_moves)*10 - ratio_number*len(opp_moves)*10)


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
    #after heuristic analysis I have choosen corners and walls heuristic combine with ratio number in the
    # initial phase
    return corners_walls_heuristic(game, player)


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
                 iterative=False, method='minimax', timeout=15.):
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
                    # getting the max value
                    if (best_score==None or temp_score > best_score):
                        best_score = temp_score
                        best_action = action
                if not maximizing_player:
                    temp_score, _ = self.minimax(game=game.forecast_move(action),depth=depth-1, maximizing_player=True)
                    # getting the min value
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
                    #beta pruning while maximazing
                    if temp_score >= beta:
                        return temp_score, action

                    # update the alpha  while maximizing
                    alpha = max([alpha, temp_score])


                if not maximizing_player:
                    temp_score, _ = self.alphabeta(game=game.forecast_move(action),depth=depth-1,
                                                   alpha=alpha, beta=beta,maximizing_player=True)
                    if (best_score==None or temp_score < best_score):
                        best_score = temp_score
                        best_action = action
                    #alpha pruning while minimizing
                    if temp_score <= alpha:
                        return temp_score, action
                    #upate beta while minimizing
                    beta = min([beta, temp_score])

            return best_score, best_action
