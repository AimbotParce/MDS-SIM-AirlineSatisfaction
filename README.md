<!-- README.md is generated from AirlineSatisfaction.Rmd. Please edit that file. -->

This project aims to predict airline passenger satisfaction levels by
developing and analyzing binary regression models. Using a dataset
containing customer demographics, flight details, and satisfaction
ratings for various services, the project explores key factors
influencing satisfaction. The workflow encompasses comprehensive data
preparation, including cleaning, transformation, and profiling, followed
by exploratory data analysis and model building.

# Data preparation

Data preparation will be broken down into two main steps. The first step
involves transforming the data into a usable format, converting factors
into such and removing useless columns, and will be applied to both
training and testing datasets. The second one will focus on cleaning the
data, removing missing values, and outliers, and will be applied only to
the training dataset.

## Data transformation

We’ll start by defining a new function that will perform the first step
of data preparation. This function will remove the “X” and “id” columns,
and convert most of the columns to factors.

``` r
prepareDataset <- function(data) {
  "Prepare dataset for analysis"
  # Remove useless columns "X" and "id"
  data$X <- NULL
  data$id <- NULL

  # Convert columns to factors
  data$Gender <- as.factor(data$Gender)
  data$Customer.Type <- as.factor(data$Customer.Type)
  data$Type.of.Travel <- as.factor(data$Type.of.Travel)
  data$Class <- factor(data$Class, levels = c("Eco", "Eco Plus", "Business"), ordered = TRUE)

  stais_cols <- c(
    "Inflight.wifi.service", "Departure.Arrival.time.convenient",
    "Ease.of.Online.booking", "Gate.location", "Food.and.drink",
    "Online.boarding", "Seat.comfort", "Inflight.entertainment",
    "On.board.service", "Leg.room.service", "Baggage.handling",
    "Checkin.service", "Inflight.service", "Cleanliness"
  )
  for (col in stais_cols) {
    data[[col]] <- factor(data[[col]], levels = 0:5, ordered = TRUE)
  }

  data$satisfaction <- as.factor(data$satisfaction)
  return(data)
}
```

Now we’ll load the training and testing datasets and apply the
`prepareDataset` function to both.

``` r
# Load datasets
train <- read.csv("data/train_data.csv")
test <- read.csv("data/test_data.csv")

# Prepare datasets
train <- prepareDataset(train)
test <- prepareDataset(test)
```

Let’s take a look at the first few rows of the training dataset to
ensure that the transformation was successful.

``` r
str(train)
```

    ## 'data.frame':    5000 obs. of  23 variables:
    ##  $ Gender                           : Factor w/ 2 levels "Female","Male": 1 1 2 1 2 2 1 1 1 1 ...
    ##  $ Customer.Type                    : Factor w/ 2 levels "disloyal Customer",..: 2 2 2 2 1 2 2 2 2 2 ...
    ##  $ Age                              : int  40 39 58 19 42 30 47 56 46 77 ...
    ##  $ Type.of.Travel                   : Factor w/ 2 levels "Business travel",..: 1 1 1 1 1 2 1 1 2 1 ...
    ##  $ Class                            : Ord.factor w/ 3 levels "Eco"<"Eco Plus"<..: 3 3 3 3 3 3 2 3 1 1 ...
    ##  $ Flight.Distance                  : int  1471 3447 3084 2756 793 516 1119 980 565 2288 ...
    ##  $ Inflight.wifi.service            : Ord.factor w/ 6 levels "0"<"1"<"2"<"3"<..: 2 5 2 3 2 4 3 2 3 4 ...
    ##  $ Departure.Arrival.time.convenient: Ord.factor w/ 6 levels "0"<"1"<"2"<"3"<..: 2 2 2 6 2 6 2 2 5 2 ...
    ##  $ Ease.of.Online.booking           : Ord.factor w/ 6 levels "0"<"1"<"2"<"3"<..: 2 5 2 6 2 4 2 2 3 2 ...
    ##  $ Gate.location                    : Ord.factor w/ 6 levels "0"<"1"<"2"<"3"<..: 2 5 2 6 4 3 2 2 2 2 ...
    ##  $ Food.and.drink                   : Ord.factor w/ 6 levels "0"<"1"<"2"<"3"<..: 6 4 4 3 4 6 4 3 5 5 ...
    ##  $ Online.boarding                  : Ord.factor w/ 6 levels "0"<"1"<"2"<"3"<..: 5 6 5 3 2 4 5 6 5 4 ...
    ##  $ Seat.comfort                     : Ord.factor w/ 6 levels "0"<"1"<"2"<"3"<..: 6 5 6 3 4 5 5 6 6 3 ...
    ##  $ Inflight.entertainment           : Ord.factor w/ 6 levels "0"<"1"<"2"<"3"<..: 6 6 4 3 4 6 3 6 6 4 ...
    ##  $ On.board.service                 : Ord.factor w/ 6 levels "0"<"1"<"2"<"3"<..: 6 6 4 4 6 4 3 6 6 4 ...
    ##  $ Leg.room.service                 : Ord.factor w/ 6 levels "0"<"1"<"2"<"3"<..: 6 6 5 4 4 4 3 6 3 4 ...
    ##  $ Baggage.handling                 : Ord.factor w/ 6 levels "0"<"1"<"2"<"3"<..: 6 6 4 5 6 6 3 6 6 4 ...
    ##  $ Checkin.service                  : Ord.factor w/ 6 levels "0"<"1"<"2"<"3"<..: 6 5 5 4 6 4 3 5 6 4 ...
    ##  $ Inflight.service                 : Ord.factor w/ 6 levels "0"<"1"<"2"<"3"<..: 6 6 4 5 5 5 3 6 6 4 ...
    ##  $ Cleanliness                      : Ord.factor w/ 6 levels "0"<"1"<"2"<"3"<..: 5 6 6 3 4 6 3 6 5 4 ...
    ##  $ Departure.Delay.in.Minutes       : int  0 0 16 12 0 0 0 15 25 47 ...
    ##  $ Arrival.Delay.in.Minutes         : int  0 0 12 19 0 0 0 4 52 28 ...
    ##  $ satisfaction                     : Factor w/ 2 levels "neutral or dissatisfied",..: 2 2 2 1 1 1 1 2 1 1 ...

## Data cleaning

This next step will focus on cleaning the training dataset. We’ll start
by removing duplicate rows, counting and imputing missing values.

``` r
# Find duplicate rows
duplicates <- train[duplicated(train), ]
cat("Number of duplicate rows:", nrow(duplicates), "\n")
```

    ## Number of duplicate rows: 0

We found no duplicates. Let’s now count the missing values in each
column.

``` r
# Count missing values
missing_values <- colSums(is.na(train))
for (col in names(missing_values)) {
  if (missing_values[col] > 0) {
    cat("Column", col, "has", missing_values[col], "missing values\n")
  }
}
```

    ## Column Arrival.Delay.in.Minutes has 12 missing values

There’s 12 missing values in the “Arrival.Delay.in.Minutes” column. As
most of our columns are factors, we would impute them using Multiple
Correspondence Analysis (MCA). However, the dataset at hand is too large
for this method to be feasible. Instead, seeing as the number of missing
values is small, we’ll simply remove the rows with missing values.

``` r
# Remove rows with missing values
before <- nrow(train)
train <- train[complete.cases(train), ]
after <- nrow(train)
cat("Removed", before - after, "rows with missing values\n")
```

    ## Removed 12 rows with missing values

Now that we’ve removed the rows with missing values, let’s check for
outliers in the numerical columns.

``` r
# Check for outliers
numeric_df <- Filter(is.numeric, train)
# There's 4 numerical columns
par(mfrow = c(2, 2))
for (col in names(numeric_df)) {
  boxplot(numeric_df[[col]], main = col, horizontal = TRUE)

  # Calculate IQR
  q1 <- quantile(numeric_df[[col]], 0.25)
  q3 <- quantile(numeric_df[[col]], 0.75)
  iqr <- q3 - q1

  # Add the lines to the plot
  abline(v = c(q1 - 1.5 * iqr, q3 + 1.5 * iqr), col = "orange")
  abline(v = c(q1 - 3 * iqr, q3 + 3 * iqr), col = "red")

  # Find outliers
  mild_outliers <- which(
    numeric_df[[col]] < (q1 - 1.5 * iqr) & numeric_df[[col]] > (q1 - 3 * iqr) |
      numeric_df[[col]] > (q3 + 1.5 * iqr) & numeric_df[[col]] < (q3 + 3 * iqr)
  )
  severe_outliers <- which(numeric_df[[col]] < (q1 - 3 * iqr) | numeric_df[[col]] > (q3 + 3 * iqr))


  cat("Column", col, "has", length(mild_outliers), "mild outliers and", length(severe_outliers), "severe outliers\n")
}
```

    ## Column Age has 0 mild outliers and 0 severe outliers

    ## Column Flight.Distance has 95 mild outliers and 0 severe outliers

    ## Column Departure.Delay.in.Minutes has 248 mild outliers and 407 severe outliers

![](images/unnamed-chunk-11-1.png)

    ## Column Arrival.Delay.in.Minutes has 252 mild outliers and 410 severe outliers

Looking at the boxplots, we can see that there are no severe outliers in
neither “Age” nor “Flight.Distance”, but there are some in
“Departure.Delay.in.Minutes” and “Arrival.Delay.in.Minutes”. However,
none of them look too extreme to be considered errors. Moreover, knowing
as we aim to predict satisfaction, arguably the more extreme the delays
are, the more likely the passenger is to be dissatisfied, providing
valuable modeling information. Therefore, we’ll keep them as they are.

Instead, knowing that the arrival and departure delays must be strongly
correlated, we’ll plot them to check if any anomalies are present,
indicating errors in the data.

``` r
# Plot departure vs arrival delays
par(mfrow = c(1, 1))
plot(train$Departure.Delay.in.Minutes, train$Arrival.Delay.in.Minutes, xlab = "Departure Delay (min)", ylab = "Arrival Delay (min)")
abline(a = 0, b = 1, col = "red")
```

![](images/unnamed-chunk-13-1.png)

No points seem to be too far from the red line, so we’ll consider them
as valid data points.

Before proceeding, we must note that there is a slight imbalance in the
target variable “Satisfaction”, which could affect the model’s
performance. We’ll address this issue in later sections.

``` r
table(train$satisfaction)
```

    ## 
    ## neutral or dissatisfied               satisfied 
    ##                    2851                    2137
