# NBA-Game-Predictor
Short Description: This repository contains an R program which uses an Extreme Gradient Boosting model trained on 2023-24 NBA game data to predict the outcome of NBA games each night.

Long Description: I have always been a big fan of these two things: the NBA and predictive analytics. Since I was in middle school, I would always spend chunks of my free time trying to find ways to accurately make NBA predictions, whether it be about the Finals winner, the MVP, or just game-by-game outcomes. Recently, I set myself the goal to build a Machine Learning based model which predicts NBA games each night. With access to a dataset I had obtained from Kaggle (https://www.kaggle.com/datasets/albi9702/nba-boxscore-season-2023-2024), which had all of the box scores from the 2023-24, I finally felt like this was a possibility.

This dataset was originally set-up where each row was a player's statline from a game. The team, player, and game were noted by TEAM_ID, PLAYER_ID, and GAME_ID, respectively. I cleaned the dataset so that it took on a new format, where each row was a game, including the stats for the top-10 (or less) players from each team in that game. I knew that when predicting, the data I would have access to were the stats of NBA players heading into a game, so I also edited my training data so that the statistics for each player were there averages heading into that game. This was possible since I had all of the games from the 23-24 season and thus could create a cumulative average since I could order the games based on their actual chronological order.

Once I had a cleaned dataset, I entered a testing phase where I tried out multiple Machine Learning models, mainly focusing on Random Forests and Extreme Gradient Boosting. While I would still like to do more work to improve the model in the future, I ended up landing on Extreme Gradient Boosting with a certain set of parameter settings, which can be seen in the code. These settings were decided through a grid search.

The model uses the following statistics as features: MIN, FGA, FGM, 3PA, 3PM, FTA, FTM, STL, BLK, REB, AST, PTS, PF, TOV (for each of the top-10 players on each team in a stat that combines Games and Minutes played). The model accounts for teams having smaller lineups by giving the missing spaces in the lineup (i.e. Player 9, Player 10) zeroes for their statistics. Thus, the model is able to pick up on differences in performance based on lineup depth. The model predicts the value for WIN, which is 1 if the team labeled Team1 wins and 0 if the team labeled Team2 wins. Team1 and Team2 win equally in the training data.

Notably, the dataset I have does not include home/away data. I have accounted for this by adding an outside-the-model adjustment for home/away record on the season. This is not used in the original model but is used on current NBA season predictions (which I do daily).

I split the 2023-24 data into 80% training, 10% validation, and 10% test, and tweaked the model using validation performance. The final performance on the test set was 68% accuracy, a value that nearly competes with some of the best performing models, which have accuracies in the low 70s.

Now, I have been assessing the performance of the model on 2024-25 NBA season games, as they happen. I have been running the model on these games each night, obtaining the average prediction across 500 iterations of the XGBoost model, which use different randomized subsets of the 2023-24 dataset. As of 1/29, the model has been 71.25% accurate in 80 games.


