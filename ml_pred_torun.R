##################################
# TRAIN MODEL
##################################

# Load required libraries
library(caret)       # For splitting data
library(randomForest) # For training the model
library(xgboost)
# Load the dplyr package
library(dplyr)
library(tidyverse)

# Get the data, which has now been cleaned
# Each row of this dataset is the statistics for active players for each game 
# in the 2023-24 NBA season (i.e. a Lakers game includes LeBron's averages heading
# into that game)
final_data <- read.csv("nba_cleaned2.csv")
final_data <- final_data %>% select(-GAME)

# List to include models created below
model_list <- list()
# List to include accuracies (helps me keep track of model performance)
accs <- list()

# Iterate through seeds 1-100 and create models
for (seed in 1:500) {
  set.seed(seed)
  # Split the data into 80% train, 10% validation, and 10% test sets
  train_index <- createDataPartition(final_data$WIN, p = 0.8, list = FALSE)
  train_data <- final_data[train_index, ]
  valid_data <- final_data[-train_index, ]
  
  # Prepare training and validation sets as matrices
  train_matrix <- data.matrix(train_data[, -which(names(train_data) == "WIN")])  # Exclude target variable
  train_label <- train_data$WIN
  valid_matrix <- data.matrix(valid_data[, -which(names(valid_data) == "WIN")])  # Exclude target variable
  valid_label <- valid_data$WIN
  
  # Convert training, validation, and test data into xgb.DMatrix format
  dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
  dvalid <- xgb.DMatrix(data = valid_matrix, label = valid_label)
  
  # Calculate the class weights
  scale_pos_weight <- sum(train_label == 0) / sum(train_label == 1)
  
  # Parameters for the model (tuning already done)
  params <- list(
    objective = 'binary:logistic',
    eval_metric = 'logloss',
    scale_pos_weight = scale_pos_weight,
    max_depth = 4,
    min_child_weight = 5,
    subsample = 0.1,
    colsample_bytree = 0.8,
    gamma = 1,
    eta = 0.05  # Lower learning rate
  )
  
  # Train the XGBoost model using xgb.train()
  watchlist <- list(train = dtrain, eval = dvalid)  # Specify the training and validation datasets for monitoring

  # Run model
  model <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = 500,  # Number of boosting rounds
    watchlist = watchlist,  # Monitor training and validation performance
    early_stopping_rounds = 10,  # Stop if no improvement for 10 rounds
    verbose = 1  # Display training progress
  )  # replace this with your actual model
  
  # Predict on the validation set
  valid_predictions <- predict(model, dvalid)
  
  # Convert probabilities to binary predictions (threshold at 0.5)
  valid_preds_binary <- ifelse(valid_predictions > 0.5, 1, 0)
  
  # Calculate accuracy and adds to accs list
  accuracy <- sum(valid_preds_binary == valid_label) / length(valid_label)
  accs <- append(accs, accuracy)
  # Save the model in the list
  model_list[[seed]] <- model
}


##################################
# FINISHED TRAINING MODEL
##################################

##################################
# GATHER NBA STATS FOR TONIGHT
##################################

library(rvest)
library(stringr)

# Create list of nba teams
nba_teams <- list(
  "Atlanta" = list("ATL", "atlanta-hawks"),
  "Boston" = list("BOS", "boston-celtics"),
  "Brooklyn" = list("BKN", "brooklyn-nets"),
  "Charlotte" = list("CHA", "charlotte-hornets"),
  "Chicago" = list("CHI", "chicago-bulls"),
  "Cleveland" = list("CLE", "cleveland-cavaliers"),
  "Dallas" = list("DAL", "dallas-mavericks"),
  "Denver" = list("DEN", "denver-nuggets"),
  "Detroit" = list("DET", "detroit-pistons"),
  "Golden State" = list("GS", "golden-state-warriors"),
  "Houston" = list("HOU", "houston-rockets"),
  "Indiana" = list("IND", "indiana-pacers"),
  "LA" = list("LAC", "los-angeles-clippers"),
  "Los Angeles" = list("LAL", "los-angeles-lakers"),
  "Memphis" = list("MEM", "memphis-grizzlies"),
  "Miami" = list("MIA", "miami-heat"),
  "Milwaukee" = list("MIL", "milwaukee-bucks"),
  "Minnesota" = list("MIN", "minnesota-timberwolves"),
  "New Orleans" = list("NO", "new-orleans-pelicans"),
  "New York" = list("NY", "new-york-knicks"),
  "Oklahoma City" = list("OKC", "oklahoma-city-thunder"),
  "Orlando" = list("ORL", "orlando-magic"),
  "Philadelphia" = list("PHI", "philadelphia-76ers"),
  "Phoenix" = list("PHO", "phoenix-suns"),
  "Portland" = list("POR", "portland-trail-blazers"),
  "Sacramento" = list("SAC", "sacramento-kings"),
  "San Antonio" = list("SA", "san-antonio-spurs"),
  "Toronto" = list("TOR", "toronto-raptors"),
  "Utah" = list("UTA", "utah-jazz"),
  "Washington" = list("WAS", "washington-wizards")
)

# Pull NBA games for tonight
# Pull html data from online
games_tn <- read_html("https://www.espn.com/nba/schedule")
# Pull tables from html
games_tn_tables <- games_tn %>% html_nodes("table")
# Table 1 is tonights games
tonight <- games_tn_tables[[1]] %>% html_table(fill=TRUE)
# Fix column names and data
colnames(tonight)[1] <- "AWAY"  # Rename first column to AWAY
colnames(tonight)[2] <- "HOME"  # Rename second column to HOME
# Remove "@ " from the beginning of each cell in the HOME column
tonight$HOME <- gsub("^@ ", "", tonight$HOME)
tonight$HOME <- substring(tonight$HOME, 2)

# Pull current standings from ESPN
standings <- read_html("https://www.espn.com/nba/standings")
standings_tables <- standings %>% html_nodes("table")
east_teams <- standings_tables[[1]] %>% html_table(fill=TRUE)
east_stds <- standings_tables[[2]] %>% html_table(fill=TRUE)
west_teams <- standings_tables[[3]] %>% html_table(fill=TRUE)
west_stds <- standings_tables[[4]] %>% html_table(fill=TRUE)

# Iterate through the nba_teams list
for (team in nba_teams) {
  # Extract the abbreviation and city
  abbreviation <- team[1]
  city <- team[2]
  
  # Update the first column of east_teams if the abbreviation is found
  east_teams[[1]] <- ifelse(grepl(abbreviation, east_teams[[1]]),
                            abbreviation,
                            east_teams[[1]])
}

# Washington Wizards doesn't exactly match up
east_teams[[1]] <- ifelse(east_teams[[1]] == "WSHWashington Wizards", "WAS", east_teams[[1]])

# Edit for Sacramento Kings (need to automate this since changes daily)
west_teams[[1]] <- ifelse(west_teams[[1]] == "10SACSacramento Kings", "SC", west_teams[[1]])
# Iterate through the nba_teams list
for (team in nba_teams) {
  # Extract the abbreviation and city
  abbreviation <- team[1]
  city <- team[2]
  
  # Update the first column of east_teams if the abbreviation is found
  west_teams[[1]] <- ifelse(grepl(abbreviation, west_teams[[1]]),
                            abbreviation,
                            west_teams[[1]])
}
# Edit for SAC
west_teams[[1]] <- ifelse(west_teams[[1]] == "SC", "SAC", west_teams[[1]])
# PHX same as Washington
west_teams[[1]] <- ifelse(west_teams[[1]] == "8PHXPhoenix Suns", "PHO", west_teams[[1]])

west_standings <- cbind(west_teams, west_stds)
colnames(west_standings)[1] <- "Team"
east_standings <- cbind(east_teams, east_stds)
colnames(east_standings)[1] <- "Team"

# Combine to overall standings
ovr_standings <- rbind(east_standings, west_standings)

# For each game tonight we will make predictions using all 500 models
for (i in 1:nrow(tonight)) {
  # Find the away team
  away <- tonight$AWAY[i]
  # Find the home team
  home <- tonight$HOME[i]
  
  # Get html player per game stats home
  # If else statements account for inconsistencies in team abbreviation
  if (home == "Charlotte") {
    home_player_per <- read_html(paste0("https://www.basketball-reference.com/teams/CHO/2025.html"))
  } else if (home == "New Orleans") {
    home_player_per <- read_html(paste0("https://www.basketball-reference.com/teams/NOP/2025.html"))
  } else if (home == "New York") {
    home_player_per <- read_html(paste0("https://www.basketball-reference.com/teams/NYK/2025.html"))
  } else if (home == "San Antonio") {
    home_player_per <- read_html(paste0("https://www.basketball-reference.com/teams/SAS/2025.html"))
  } else if (home == "Brooklyn") {
    home_player_per <- read_html(paste0("https://www.basketball-reference.com/teams/BRK/2025.html"))
  } else if (home == "Golden State") {
    home_player_per <- read_html(paste0("https://www.basketball-reference.com/teams/GSW/2025.html"))
  } else {
    home_player_per <- read_html(paste0("https://www.basketball-reference.com/teams/", as.character(nba_teams[[home]][1]), "/2025.html"))
  }
  # Get html player per game stats away
  if (away == "Charlotte") {
    away_player_per <- read_html(paste0("https://www.basketball-reference.com/teams/CHO/2025.html"))
  } else if (away == "New Orleans") {
    away_player_per <- read_html(paste0("https://www.basketball-reference.com/teams/NOP/2025.html"))
  } else if (away == "New York") {
    away_player_per <- read_html(paste0("https://www.basketball-reference.com/teams/NYK/2025.html"))
  } else if (away == "San Antonio") {
    away_player_per <- read_html(paste0("https://www.basketball-reference.com/teams/SAS/2025.html"))
  } else if (away == "Brooklyn") {
    away_player_per <- read_html(paste0("https://www.basketball-reference.com/teams/BRK/2025.html"))
  } else if (away == "Golden State") {
    away_player_per <- read_html(paste0("https://www.basketball-reference.com/teams/GSW/2025.html"))
  } else {
    away_player_per <- read_html(paste0("https://www.basketball-reference.com/teams/", as.character(nba_teams[[away]][1]), "/2025.html"))
  }
  # Sleep to prevent too many requests
  Sys.sleep(10)
  
  # Get tables from html for player stats
  home_player_per_tables <- home_player_per %>% html_nodes("table")
  away_player_per_tables <- away_player_per %>% html_nodes("table")
  
  # Get player per game stats
  home_player_per_table <- home_player_per_tables[[2]] %>% html_table(fill = TRUE)
  away_player_per_table <- away_player_per_tables[[2]] %>% html_table(fill = TRUE)
  
  # Get html for injuries home and away
  home_injury_report <- read_html(paste0("https://www.cbssports.com/nba/teams/", as.character(nba_teams[[home]][1]), "/", as.character(nba_teams[[home]][2]), "/injuries/"))
  away_injury_report <- read_html(paste0("https://www.cbssports.com/nba/teams/", as.character(nba_teams[[away]][1]), "/", as.character(nba_teams[[away]][2]), "/injuries/"))
  
  # Get tables from html injuries
  home_inj_tables <- home_injury_report %>% html_nodes("table")
  away_inj_tables <- away_injury_report %>% html_nodes("table")
  
  # Function for finding the index of the first capital
  find_first_capital <- function(string) {
    # Define the regular expression
    pattern <- "(?<!^|\\s|-|Mc|Mac|Di)[A-Z]"
    
    # Use str_locate to find the location of the first match
    match <- str_locate(string, pattern)
    
    # Return the starting position of the match or NA if no match
    return(match[1])
  }
  
  # Function for removing before that index
  remove_before_index <- function(string, index) {
    # Use substr to extract the part of the string from the given index onwards
    return(substr(string, index, nchar(string)))
  }
  
  # Make data frame out of injury table
  # Use try catch incase no injuries
  tryCatch({
    # Get injury tabvle
    home_table <- home_inj_tables[[1]] %>% html_table(fill = TRUE)
    # Pull multiple injury tables if team has multiple (from different days)
    # Injury tables are separated by day of report on CBS sports
    if (length(home_inj_tables) > 1) {
      for (i in 2:length(home_inj_tables)) {
        home_table <- rbind(home_table, (home_inj_tables[[i]] %>% html_table(fill = TRUE)))
      }
    }
    # Get rid of first initial
    home_table$Player <- gsub("^[A-Za-z]+\\. ", "", home_table$Player)
    # Apply the functions to the Player column of injury tables
    home_table <- home_table %>%
      mutate(Player = sapply(Player, function(x) {
        # Find the first capital letter's position
        index <- find_first_capital(x)
        
        # If a capital letter is found, remove everything before it
        if (!is.na(index)) {
          return(remove_before_index(x, index))
        } else {
          return(x)  # If no capital letter found, return the original string
        }
      }))
    # Filter injury tables where Injury Status is not "Game Time Decision"
    home_players_to_remove <- home_table$Player[home_table$`Injury Status` != "Game Time Decision"]
    home_players_to_remove <- c(home_players_to_remove, "Team Totals")
    home_players_to_remove <- c(home_players_to_remove, "Jimmy Butler") # Always suspended!
    # Remove the corresponding rows from player_per_table
    home_player_per_table <- home_player_per_table[!(home_player_per_table$Player %in% home_players_to_remove), ]
  }, error = function(e) {
    # If error, just remove Team Totals
    home_players_to_remove <<- c("Team Totals")
    home_player_per_table <<- home_player_per_table[!(home_player_per_table$Player %in% home_players_to_remove), ]
    # Handle the error by skipping the remaining lines
    message("An error occurred with the initial line, skipping subsequent processing.")
  })
  
  # Same try catch logic except for away teams
  tryCatch({
    away_table <- away_inj_tables[[1]] %>% html_table(fill = TRUE)
    if (length(away_inj_tables) > 1) {
      for (i in 2:length(away_inj_tables)) {
        away_table <- rbind(away_table, (away_inj_tables[[i]] %>% html_table(fill = TRUE)))
      }
    }
    away_table$Player <- gsub("^[A-Za-z]+\\. ", "", away_table$Player)
    away_table <- away_table %>%
      mutate(Player = sapply(Player, function(x) {
        # Find the first capital letter's position
        index <- find_first_capital(x)
        
        # If a capital letter is found, remove everything before it
        if (!is.na(index)) {
          return(remove_before_index(x, index))
        } else {
          return(x)  # If no capital letter found, return the original string
        }
      }))
    away_players_to_remove <- away_table$Player[away_table$`Injury Status` != "Game Time Decision"]
    away_players_to_remove <- c(away_players_to_remove, "Team Totals")
    away_players_to_remove <- c(away_players_to_remove, "Jimmy Butler") # Alwayssss suspended!
    # Remove the corresponding rows from player_per_table
    away_player_per_table <- away_player_per_table[!(away_player_per_table$Player %in% away_players_to_remove), ]
  }, error = function(e) {
    away_players_to_remove <<- list()
    away_players_to_remove <<- c(away_players_to_remove, "Team Totals")
    away_player_per_table <<- away_player_per_table[!(away_player_per_table$Player %in% away_players_to_remove), ]
    # Handle the error by skipping the remaining lines
    message("An error occurred with the initial line, skipping subsequent processing.")
  })
  
  # Add a new column to player_per_table by multiplying 0.5 * G and MP
  # This will be what we sort on for player performance
  # Shows players who play in most games and minutes
  # Tries to account for star players who have been injured and are back
  # Need to improve this!
  home_player_per_table$G_times_MP <- (0.5*home_player_per_table$G) * home_player_per_table$MP
  away_player_per_table$G_times_MP <- (0.5*away_player_per_table$G) * away_player_per_table$MP
  
  # Sort player_per_table by G_times_MP in descending order
  home_player_per_table_sorted <- home_player_per_table[order(-home_player_per_table$G_times_MP), ]
  away_player_per_table_sorted <- away_player_per_table[order(-away_player_per_table$G_times_MP), ]
  
  # Take the top 10 players (lines up with data used in models)
  home_player_per_table_sorted <- head(home_player_per_table_sorted, 10)
  away_player_per_table_sorted <- head(away_player_per_table_sorted, 10)
  
  # Following code sets up the data to look the same as data used in models
  home_box_score <- home_player_per_table_sorted %>% arrange(desc(MP))
  away_box_score <- away_player_per_table_sorted %>% arrange(desc(MP))
  
  home_box_score <- home_box_score %>% select(-Rk, -Player, -Age, -Pos, -G, -GS, -'FG%', -'3P%', -'2P', -'2PA', -'2P%', -'eFG%', -'FT%', -Awards, -G_times_MP)
  away_box_score <- away_box_score %>% select(-Rk, -Player, -Age, -Pos, -G, -GS, -'FG%', -'3P%', -'2P', -'2PA', -'2P%', -'eFG%', -'FT%', -Awards, -G_times_MP)
  
  home_box_score <- home_box_score %>% rename(MIN = MP, FGM = FG, FG3M = '3P', FG3A = '3PA', FTM = FT, OREB = ORB, DREB = DRB, REB = TRB, TO = TOV)
  away_box_score <- away_box_score %>% rename(MIN = MP, FGM = FG, FG3M = '3P', FG3A = '3PA', FTM = FT, OREB = ORB, DREB = DRB, REB = TRB, TO = TOV)
  
  colnames(home_box_score) <- paste0(colnames(home_box_score), "_avg")
  colnames(away_box_score) <- paste0(colnames(away_box_score), "_avg")
  
  # Add row numbers for player index
  home_box_score$Player <- seq_len(nrow(home_box_score)) - 1
  
  # Reshape the dataframe using pivot_wider
  new_home_box_score <- home_box_score %>%
    pivot_wider(
      names_from = Player, 
      values_from = c(MIN_avg, FGM_avg, FGA_avg, FG3M_avg, FG3A_avg, FTM_avg, FTA_avg, 
                      OREB_avg, DREB_avg, REB_avg, AST_avg, STL_avg, BLK_avg, TO_avg, PF_avg, PTS_avg),
    )
  
  # Add row numbers for player index
  away_box_score$Player <- seq_len(nrow(away_box_score)) - 1
  
  # Reshape the dataframe using pivot_wider
  new_away_box_score <- away_box_score %>%
    pivot_wider(
      names_from = Player, 
      values_from = c(MIN_avg, FGM_avg, FGA_avg, FG3M_avg, FG3A_avg, FTM_avg, FTA_avg, 
                      OREB_avg, DREB_avg, REB_avg, AST_avg, STL_avg, BLK_avg, TO_avg, PF_avg, PTS_avg),
    )
  
  full_box_score <- bind_cols(
    new_home_box_score %>% rename_with(~ paste0(. , "_team1")),  # Add '_team1' suffix to columns from reshaped_df
    new_away_box_score %>% rename_with(~ paste0(. , "_team2"))  # Add '_team2' suffix to columns from reshaped_df2
  )
  
  fbs <- data.matrix(full_box_score)
  full_box_score_matrix <- xgb.DMatrix(data = fbs)
  
  # The following code is adding an adjustment for home/away record of the teams playing
  # Pull home team home record
  # Get the HOME record for the specified team
  home_record <- ovr_standings$HOME[ovr_standings$Team == as.character(nba_teams[[home]][1])]
  

  # Split the HOME record into wins (W) and losses (L)
  home_record_split <- strsplit(home_record, "-")[[1]]
  W <- as.numeric(home_record_split[1]) # Wins
  L <- as.numeric(home_record_split[2]) # Losses
  
  # Calculate the home winning percentage
  home2_pct <- W / (W + L)
  # Scale it
  h2p_sc <- (home2_pct - 0.5) / 10
  
  # Pull away team away record
  # Get the HOME record for the specified team
  away_record <- ovr_standings$AWAY[ovr_standings$Team == as.character(nba_teams[[away]][1])]
  
  # Split the HOME record into wins (W) and losses (L)
  away_record_split <- strsplit(away_record, "-")[[1]]
  W <- as.numeric(away_record_split[1]) # Wins
  L <- as.numeric(away_record_split[2]) # Losses
  
  # Calculate the home winning percentage
  away2_pct <- W / (W + L)
  a2p_sc <- (0.5 - away2_pct) / 10
  
  # Make predictions using the 500 models
  prediction_list <- list()
  pred_adj_list <- list()
  for (j in 1:length(model_list)) {
    prediction <- predict(model_list[[j]], full_box_score_matrix)
    prediction_list <- append(prediction_list, prediction)
    # Adjusted prediction w/ home away adjustment
    pred_adj <- prediction + h2p_sc + a2p_sc
    pred_adj_list <- append(pred_adj_list, pred_adj)
  }
  
  # Convert the list to a numeric vector
  final_pred_list <- unlist(prediction_list)
  final_pred_adj_list <- unlist(pred_adj_list)
  
  # Calculate the average prediction
  avg_pred <- mean(final_pred_list)
  avg_pred_adj <- mean(final_pred_adj_list)
  
  # Calculate the percentage of home team wins (predictions > 0.5)
  home_team_wins_perc <- sum(final_pred_list > 0.5) / 5
  home_team_wins_adj_perc <- sum(final_pred_adj_list > 0.5) / 5
  
  # Logic for printing out the outcome, added this to make it clearer which team is favored
  if (avg_pred > 0.5 && home_team_wins_perc > 50) {
    print(paste0("IsaacScore2.0: ", home, " vs. ", away, ": ", round(avg_pred,3), " (", home, ")"))
    print(paste0("Home Win %: ", home, " vs. ", away, ": ", home_team_wins_perc, "%"))
  } else if (avg_pred < 0.5 && home_team_wins_perc < 50) {
    print(paste0("IsaacScore2.0: ", home, " vs. ", away, ": ", round(avg_pred,3), " (", away, ")"))
    print(paste0("Home Win %: ", home, " vs. ", away, ": ", home_team_wins_perc, "%"))
  } else {
    print(paste0("IsaacScore2.0: ", home, " vs. ", away, ": ", round(avg_pred,3), " (uncertain)"))
    print(paste0("Home Win %: ", home, " vs. ", away, ": ", home_team_wins_perc, "%"))
  }

  if (avg_pred_adj > 0.5 && home_team_wins_adj_perc > 50) {
    print(paste0("H/A Adjusted: ", home, " vs. ", away, ": ", round(avg_pred_adj,3), " (", home, ")"))
    print(paste0("Home Win % Adjusted: ", home, " vs. ", away, ": ", home_team_wins_adj_perc, "%"))
  } else if (avg_pred_adj < 0.5 && home_team_wins_adj_perc < 50) {
    print(paste0("H/A Adjusted: ", home, " vs. ", away, ": ", round(avg_pred_adj,3), " (", away, ")"))
    print(paste0("Home Win % Adjusted: ", home, " vs. ", away, ": ", home_team_wins_adj_perc, "%"))
  } else {
    print(paste0("H/A Adjusted: ", home, " vs. ", away, ": ", round(avg_pred_adj,3), " (uncertain)"))
    print(paste0("Home Win % Adjusted: ", home, " vs. ", away, ": ", home_team_wins_adj_perc, "%"))
  }
  cat("\n")
}



