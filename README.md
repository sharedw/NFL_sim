# NFL Game Simulator :football:

This Project simulates NFL games at the play by play level. For a given matchup you can simulate the game, and it will track each player and team's stats. Every step of the game will be simulated. For an example play the steps are:
- Predicts which type of play will be run (pass, run, timeout, field goal, punt, etc.)
- Pass play is predicted, predicts who the passer is based on game state (generally the starter, except in end of blowouts)
- Predicts which receiver will be passed to
- Predicts how many yards in the air the ball will travel
- Predicts if the pass is caught
- If caught, predicts how many yards the receiver will run before being tackled
- Simulates how much time has run off the clock during this play

## Data

Each of these steps is trained on years of play by play data. The inputs for each step are a combination of game context (time on the clock, score, field position, etc.), team data(offense and defense stats, play call frequency, etc) as well as player specific stats( how good is this quarterback/receiver/runningback).

## Models
The models are a mixture of gradient boosted decision trees(XGBoost), and custom Neural Networks. I have also experimented with some quantile regression, flow based models, and other NN architectures. Because this is a simulator, I can't just predict medians, I need the full distribution of possible outcomes which is why the NN's have been helpful. They more accurately tease out the relationship between field position, yards needed for a first down, player skill etc, and give me the full distribution from -99 to 99 for how many yards can potentially be gained on each play.

## Repository structure

- `data`: All the data for the project is loaded with the nfl-data-py package, and stored in the `data` folder
- `exploration` The exploratory notebooks for new models is stored here
- `models` The model weights and a dictionary of features is stored here
- `train_models` Model training is in the process of being moved from notebooks to `.py` files here
- `sim.ipynb` Here is the notebook for experimenting with the actual simulator
	- There is a class for each of the positions, `QB`, `RB`, etc. to store player info as well as the `Team` itself
	- The `GameState` class holds the logic for the game itself, tracking game context, calling the models, etc.
	- todo: Adding a more robust testing framework, to see the accuracy of game outcomes, team level stats, and player level stats.
- `clean_feature_eng` and `team_stats` Feature engineering is a bit scattered currently, across these files
