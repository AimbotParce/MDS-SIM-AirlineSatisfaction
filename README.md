<!-- README.md is generated from AirlineSatisfaction.Rmd. Please edit that file. -->

This project aims to predict airline passenger satisfaction levels by
developing and analyzing binary regression models. Using a dataset
containing customer demographics, flight details, and satisfaction
ratings for various services, the project explores key factors
influencing satisfaction.

This work is organized as follows: [Section 1](#data-preparation) covers
data preparation, including data transformation and cleaning. [Section
2](#eda) explores each variable in the dataset in detail, whilst
[Section 3](#profiling) searches for relationships between variables and
satisfaction levels. [Section 4](#model-building), having studied the
data, iteratively builds a logistic regression model to predict
satisfaction. [Section 5](#model-validation) introduces the validation
metrics used, and uses them to compare two of the best models selected
on the previous section. Finally, [Section 6](#conclusion) gives a brief
interpretation of the results.

# 1. Data preparation

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

# 2. Exploratory Data Analysis

This section will give a small explanation of each variable in the
dataset, including its type, levels, and distribution. We’ll also
provide visualizations to help understand the data better.

## Variable 1: Gender

This is a Nominal variable with 2 levels (binary). A pie chart shows
that the two levels are well represented.

``` r
summary(train$Gender)
```

    ## Female   Male 
    ##   2544   2444

``` r
piechart(train$Gender) # See code for the function definition.
```

    ##     data Freq
    ## 1 Female 2544
    ## 2   Male 2444

![](images/unnamed-chunk-16-1.png)

## Variable 2: Customer.Type

This is a Nominal variable with 2 levels (binary). Another pie chart
shows an imbalance between the two groups, with the loyal customers
group being 4x larger than the disloyal customer group. Since the
imbalance is not large enough to affect the model, no action is taken.

``` r
summary(train$Customer.Type)
```

    ## disloyal Customer    Loyal Customer 
    ##               932              4056

``` r
piechart(train$Customer.Type)
```

    ##                data Freq
    ## 1 disloyal Customer  932
    ## 2    Loyal Customer 4056

![](images/unnamed-chunk-17-1.png)

## Variable 3: Age

This is a Numeric interval variable. It is visualized by a histogram and
a boxplot to examine its distribution, which appears to be well
represented for people of all ages, with no outliers. Visually, a slight
deviation from a normal distribution (centered at age 40) is seen at the
ages 20-30, being slightly higher represented.

``` r
summary(train$Age)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##    7.00   27.00   40.00   39.41   51.00   85.00

``` r
hist(train$Age, breaks = 30, xlab = "Age", main = NULL)
```

![](images/unnamed-chunk-18-1.png)

``` r
boxplot(train$Age, horizontal = TRUE, xlab = "Age")
```

![](images/unnamed-chunk-18-2.png)

## Variable 4: Type.of.Travel

This is a Nominal variable with 2 levels (binary). Looking at a pie
chart, it becomes clear that all levels are well represented, with
Business travel having 2x the individuals than Personal travel.

``` r
summary(train$Type.of.Travel)
```

    ## Business travel Personal Travel 
    ##            3452            1536

``` r
piechart(train$Type.of.Travel)
```

    ##              data Freq
    ## 1 Business travel 3452
    ## 2 Personal Travel 1536

![](images/unnamed-chunk-19-1.png)

## Variable 5: Class

This is a Nominal variable with 3 levels. When visualizing it on a pie
chart, one can easily see that one of the levels is underrepresented
with only 7.2% of the observations. Since this is still a significant
portion of the observations, no action was taken.

``` r
summary(train$Class)
```

    ##      Eco Eco Plus Business 
    ##     2271      361     2356

``` r
piechart(train$Class)
```

    ##       data Freq
    ## 1      Eco 2271
    ## 2 Eco Plus  361
    ## 3 Business 2356

![](images/unnamed-chunk-20-1.png)

## Variable 6: Flight.Distance

This is a continuous ratio variable. It is visualized by a histogram and
a boxplot to examine its distribution. A few univariate outliers are
detected, but they are not so far from the interquartile range to
consider changing or deleting them. It is clearly not normally
distributed, confirmed by the near-null p-value of the shapiro
normallity test.

``` r
summary(train$Flight.Distance)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##      67     414     842    1194    1754    4983

``` r
hist(train$Flight.Distance, breaks = 30, main = NULL, xlab = "Flight Distance")
```

![](images/unnamed-chunk-21-1.png)

``` r
boxplot(train$Flight.Distance, horizontal = TRUE, xlab = "Flight Distance")
```

![](images/unnamed-chunk-21-2.png)

``` r
shapiro.test(train$Flight.Distance)
```

    ## 
    ##  Shapiro-Wilk normality test
    ## 
    ## data:  train$Flight.Distance
    ## W = 0.86751, p-value < 2.2e-16

## Variable 7: Inflight.wifi.service

This is the first of many Nominal variables with 6 levels (0 to 5,
customer satisfaction level). It is visualized by a bar plot, in which
it is clear that all levels are well represented.

``` r
summary(train$Inflight.wifi.service)
```

    ##    0    1    2    3    4    5 
    ##  149  875 1206 1240  971  547

``` r
plot(train$Inflight.wifi.service, xlab = "Inflight wifi service satisfaction level")
```

![](images/unnamed-chunk-22-1.png)

##Variable 8: Departure.Arrival.time.convenient This is another Nominal
variable with 6 levels. It is visualized by a bar plot, in which it is
clear that all levels are well represented.

``` r
summary(train$Departure.Arrival.time.convenient)
```

    ##    0    1    2    3    4    5 
    ##  247  751  800  876 1246 1068

``` r
plot(train$Departure.Arrival.time.convenient, xlab = "Convinience of Departure and Arrival time satisfaction level")
```

![](images/unnamed-chunk-23-1.png)

## Variable 9: Ease.of.Online.booking

This is another Nominal variable with 6 levels. It is visualized by a
bar plot, in which it is clear that all levels are well represented.

``` r
summary(train$Ease.of.Online.booking)
```

    ##    0    1    2    3    4    5 
    ##  217  849 1091 1200  966  665

``` r
plot(train$Ease.of.Online.booking, xlab = "Ease of Online booking satisfaction level")
```

![](images/unnamed-chunk-24-1.png)

## Variable 10: Gate.location

This is another Nominal variable with 6 levels. It is visualized by a
bar plot. One of the levels is not represented at all, but is not
deleted in case the test dataset contains observations at this level.

``` r
summary(train$Gate.location)
```

    ##    0    1    2    3    4    5 
    ##    0  835  907 1376 1185  685

``` r
plot(train$Gate.location, xlab = "Gate location satisfaction level")
```

![](images/unnamed-chunk-25-1.png)

## Variable 11: Food.and.drink

This is another Nominal variable with 6 levels. It is visualized by a
bar plot. The 0 level is very underrepresented, but as before is not
deleted in case it is not underrepresented in the test dataset.

``` r
summary(train$Food.and.drink)
```

    ##    0    1    2    3    4    5 
    ##    5  668  982 1105 1181 1047

``` r
plot(train$Food.and.drink, xlab = "Food and drink satisfaction level")
```

![](images/unnamed-chunk-26-1.png)

## Variable 12: Online.boarding

This is another Nominal variable with 6 levels. It is visualized by a
bar plot, in which it is clear that all levels are well represented.

``` r
summary(train$Online.boarding)
```

    ##    0    1    2    3    4    5 
    ##  119  535  812 1028 1535  959

``` r
plot(train$Online.boarding, xlab = "Online boarding satisfaction level")
```

![](images/unnamed-chunk-27-1.png)

## Variable 13: Seat.comfort

This is another Nominal variable with 6 levels. It is visualized by a
bar plot. The 0 level is not represented, but as before is not deleted
in case it is not underrepresented in the test dataset.

``` r
summary(train$Seat.comfort)
```

    ##    0    1    2    3    4    5 
    ##    0  626  702  877 1543 1240

``` r
plot(train$Seat.comfort, xlab = "Seat comfort satisfaction level")
```

![](images/unnamed-chunk-28-1.png)

## Variable 14: Inflight.entertainment

This is another Nominal variable with 6 levels. It is visualized by a
bar plot. The 0 level is very underrepresented, but as before is not
deleted in case it is not underrepresented in the test dataset.

``` r
summary(train$Inflight.entertainment)
```

    ##    0    1    2    3    4    5 
    ##    1  625  812  947 1433 1170

``` r
plot(train$Inflight.entertainment, xlab = "Inflight entertainment satisfaction level")
```

![](images/unnamed-chunk-29-1.png)

## Variable 15: On.board.service

This is another Nominal variable with 6 levels. It is visualized by a
bar plot. The 0 level is very underrepresented, but as before is not
deleted in case it is not underrepresented in the test dataset.

``` r
summary(train$On.board.service)
```

    ##    0    1    2    3    4    5 
    ##    1  555  712 1125 1531 1064

``` r
plot(train$On.board.service, xlab = "On-board service satisfaction level")
```

![](images/unnamed-chunk-30-1.png)

## Variable 16: Leg.room.service

This is another Nominal variable with 6 levels. It is visualized by a
bar plot, in which it is clear that all levels are well represented.

``` r
summary(train$Leg.room.service)
```

    ##    0    1    2    3    4    5 
    ##   22  502  958  969 1347 1190

``` r
plot(train$Leg.room.service, xlab = "Leg room service satisfaction level")
```

![](images/unnamed-chunk-31-1.png)

## Variable 17: Baggage.handling

This is another Nominal variable with 6 levels. It is visualized by a
bar plot. The 0 level is not represented, but as before is not deleted
in case it is represented in the test dataset.

``` r
summary(train$Baggage.handling)
```

    ##    0    1    2    3    4    5 
    ##    0  359  542 1007 1807 1273

``` r
plot(train$Baggage.handling, xlab = "Baggage handling satisfaction level")
```

![](images/unnamed-chunk-32-1.png)

## Variable 18: Checkin.service

This is another Nominal variable with 6 levels. It is visualized by a
bar plot. The 0 level is not represented, but as before is not deleted
in case it is represented in the test dataset.

``` r
summary(train$Checkin.service)
```

    ##    0    1    2    3    4    5 
    ##    0  649  604 1394 1416  925

``` r
plot(train$Checkin.service, xlab = "Check-in service satisfaction level")
```

![](images/unnamed-chunk-33-1.png)

## Variable 19: Inflight.service

This is another Nominal variable with 6 levels. It is visualized by a
bar plot. The 0 level is very underrepresented, but as before is not
deleted in case it is not underrepresented in the test dataset.

``` r
summary(train$Inflight.service)
```

    ##    0    1    2    3    4    5 
    ##    1  354  532 1003 1870 1228

``` r
plot(train$Inflight.service, xlab = "Inflight service satisfaction level")
```

![](images/unnamed-chunk-34-1.png)

## Variable 20: Cleanliness

This is another Nominal variable with 6 levels. It is visualized by a
bar plot. The 0 level is not represented, but as before is not deleted
in case it is represented in the test dataset.

``` r
summary(train$Cleanliness)
```

    ##    0    1    2    3    4    5 
    ##    0  658  754 1204 1332 1040

``` r
plot(train$Cleanliness, xlab = "Cleanliness satisfaction level")
```

![](images/unnamed-chunk-35-1.png)

## Variable 21: Departure.Delay.in.Minutes

This is a continuous ratio variable. It is visualized by a histogram and
a boxplot to examine its distribution. 56.70% of the observations have a
delay of 0 minutes, and the variable clearly doesn’t follow a normal
distribution, confirmed by a shapiro test. Due to its nature, it
contains a lot of outliers (almost any observation that is not 0) so no
action is taken to change/delete them, not only because they represent a
large percentage of the observations, but also because, having some
domain knowledge, high amounts of delay will probably highly influence
the satisfaction level - or lack thereof.

``` r
summary(train$Departure.Delay.in.Minutes)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##    0.00    0.00    0.00   14.98   13.00  692.00

``` r
mean(train$Departure.Delay.in.Minutes == 0) * 100
```

    ## [1] 56.69607

``` r
hist(train$Departure.Delay.in.Minutes, breaks = 30, main = NULL, xlab = "Departure Delay (min)")
```

![](images/unnamed-chunk-36-1.png)

``` r
boxplot(train$Departure.Delay.in.Minutes, horizontal = TRUE, xlab = "Departure Delay (min)")
```

![](images/unnamed-chunk-36-2.png)

``` r
shapiro.test(train$Departure.Delay.in.Minutes)
```

    ## 
    ##  Shapiro-Wilk normality test
    ## 
    ## data:  train$Departure.Delay.in.Minutes
    ## W = 0.42753, p-value < 2.2e-16

## Variable 22: Arrival.Delay.in.Minutes

This is a continuous ratio variable. It is visualized by a histogram and
a boxplot to examine its distribution. Similarly to the last variable,
55.77% of the observations have a delay of 0 minutes, and the variable
clearly doesn’t follow a normal distribution, confirmed by a shapiro
test. Again, due to its nature, it contains a lot of outliers, but no
related action is taken.

As explained before (in [section 1](#data-preparation)), as expected,
this variable is highly correlated with the Departure.Delay.in.Minutes
variable.

``` r
summary(train$Arrival.Delay.in.Minutes)
```

    ##    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
    ##    0.00    0.00    0.00   15.28   13.00  702.00

``` r
mean(train$Arrival.Delay.in.Minutes == 0) * 100
```

    ## [1] 55.77386

``` r
hist(train$Arrival.Delay.in.Minutes, breaks = 30, main = NULL, xlab = "Arrival Delay (min)")
```

![](images/unnamed-chunk-37-1.png)

``` r
boxplot(train$Arrival.Delay.in.Minutes, horizontal = TRUE, xlab = "Arrival Delay (min)")
```

![](images/unnamed-chunk-37-2.png)

``` r
shapiro.test(train$Arrival.Delay.in.Minutes)
```

    ## 
    ##  Shapiro-Wilk normality test
    ## 
    ## data:  train$Arrival.Delay.in.Minutes
    ## W = 0.42739, p-value < 2.2e-16

## Variable 23: Satisfaction

This is our response variable. It is a Nominal variable with 2 levels
(binary). It is visualized by a bar plot, in which it is clear that all
levels are well represented, although a slight imbalance is present, not
high enough to significantly affect the resulting model.

``` r
summary(train$satisfaction)
```

    ## neutral or dissatisfied               satisfied 
    ##                    2851                    2137

``` r
piechart(train$satisfaction)
```

    ##                      data Freq
    ## 1 neutral or dissatisfied 2851
    ## 2               satisfied 2137

![](images/unnamed-chunk-38-1.png)

A higher imbalance could affect the model’s performance, which could be
counteracted by using techniques such as oversampling (duplicating some
observation from the minority class), undersampling (removing some
observations from the majority class), or, in the fitting process, using
higher weights for the “losses” of the minority class.

# 3. Profiling

To analyze the categorical response variable `satisfaction`, the
`catdes` function from the `FactoMineR` package was used. This function
identifies the relationship between the target variable and the other
quantitative and qualitative variables.

``` r
res.cat <- catdes(train, 23)
res.cat$test.chi2
```

    ##                                         p.value df
    ## Online.boarding                    0.000000e+00  5
    ## Inflight.wifi.service             6.260486e-299  5
    ## Class                             1.464241e-289  2
    ## Type.of.Travel                    4.062244e-215  1
    ## Inflight.entertainment            4.299461e-188  5
    ## Seat.comfort                      2.758809e-160  4
    ## On.board.service                  7.622598e-122  5
    ## Leg.room.service                  8.208984e-119  5
    ## Ease.of.Online.booking            9.700564e-104  5
    ## Cleanliness                       1.533379e-102  4
    ## Baggage.handling                   1.660675e-87  4
    ## Inflight.service                   4.483314e-82  5
    ## Checkin.service                    1.949045e-59  4
    ## Food.and.drink                     8.346914e-48  5
    ## Customer.Type                      3.751231e-46  1
    ## Gate.location                      2.439415e-25  4
    ## Departure.Arrival.time.convenient  1.621261e-02  5

The chi-squared test results show that several qualitative variables are
significantly associated with the satisfaction levels, as indicated by
their extremely low p-values. The variables most strongly associated
with satisfaction (in order of decreasing significance) are
`Online.boarding`, `Inflight.wifi.service`, `Class`, `Type.of.Travel`
and `Inflight.entertainment`.

``` r
res.cat$quanti.var
```

    ##                                   Eta2       P-value
    ## Flight.Distance            0.089838774 4.543902e-104
    ## Age                        0.021820600  9.843326e-26
    ## Arrival.Delay.in.Minutes   0.001750924  3.118368e-03
    ## Departure.Delay.in.Minutes 0.001226100  1.339278e-02

``` r
res.cat$quanti
```

    ## $`neutral or dissatisfied`
    ##                                v.test Mean in category Overall mean
    ## Arrival.Delay.in.Minutes     2.954972         16.68467     15.27506
    ## Departure.Delay.in.Minutes   2.472764         16.14170     14.97895
    ## Age                        -10.431650         37.50614     39.41399
    ## Flight.Distance            -21.166624        934.93020   1194.21592
    ##                            sd in category Overall sd      p.value
    ## Arrival.Delay.in.Minutes         41.51452   38.91009 3.126973e-03
    ## Departure.Delay.in.Minutes       40.88750   38.35497 1.340727e-02
    ## Age                              16.34356   14.91793 1.777733e-25
    ## Flight.Distance                 809.53607  999.17839 1.939642e-99
    ## 
    ## $satisfied
    ##                               v.test Mean in category Overall mean
    ## Flight.Distance            21.166624       1540.13243   1194.21592
    ## Age                        10.431650         41.95929     39.41399
    ## Departure.Delay.in.Minutes -2.472764         13.42770     14.97895
    ## Arrival.Delay.in.Minutes   -2.954972         13.39448     15.27506
    ##                            sd in category Overall sd      p.value
    ## Flight.Distance                1116.52166  999.17839 1.939642e-99
    ## Age                              12.31876   14.91793 1.777733e-25
    ## Departure.Delay.in.Minutes       34.62888   38.35497 1.340727e-02
    ## Arrival.Delay.in.Minutes         35.04801   38.91009 3.126973e-03

For the quantitative variables, the Eta² values and p-values provide
insight into their relationship with `satisfaction`. The most
influential variables include `Flight.Distance` (passengers with longer
flight distances tend to report higher satisfaction, as indicated by the
positive mean in the satisfied category), `Age` (older passengers are
more likely to report satisfaction, with a higher mean age in the
satisfied category compared to neutral or dissatisfied),
`Arrival.Delay.in.Minutes` and `Departure.Delay.in.Minutes` (both delays
are negatively associated with satisfaction, as expected, though their
impact is relatively small)

Passengers categorized as satisfied have a significantly higher mean in
`Flight.Distance` and `Age` compared to the overall averages.

Passengers categorized as neutral or dissatisfied have lower mean
`Flight.Distance` and `Age`. Both `Departure.Delay.in.Minutes` and
`Arrival.Delay.in.Minutes` are slightly higher.

The analysis reveals that satisfaction is most strongly influenced by
qualitative variables such as `Online.boarding`,
`Inflight.wifi.service`, and `Class`. Among quantitative variables,
`Flight.Distance` and `Age` are the most significant predictors.

# 4. Feature Selection & Model Building

``` r
numeric_vars <- sapply(train, is.numeric)
summary(train[, numeric_vars])
```

    ##       Age        Flight.Distance Departure.Delay.in.Minutes
    ##  Min.   : 7.00   Min.   :  67    Min.   :  0.00            
    ##  1st Qu.:27.00   1st Qu.: 414    1st Qu.:  0.00            
    ##  Median :40.00   Median : 842    Median :  0.00            
    ##  Mean   :39.41   Mean   :1194    Mean   : 14.98            
    ##  3rd Qu.:51.00   3rd Qu.:1754    3rd Qu.: 13.00            
    ##  Max.   :85.00   Max.   :4983    Max.   :692.00            
    ##  Arrival.Delay.in.Minutes
    ##  Min.   :  0.00          
    ##  1st Qu.:  0.00          
    ##  Median :  0.00          
    ##  Mean   : 15.28          
    ##  3rd Qu.: 13.00          
    ##  Max.   :702.00

``` r
corr_matrix <- cor(train[, numeric_vars], use = "complete.obs")
corrplot(corr_matrix)
```

![](images/unnamed-chunk-41-1.png)

``` r
train1 <- train
# CHI-TESTS(Testing independence between two Factors)
variables_to_clean <- c(
  "Gate.location", "Food.and.drink",
  "Inflight.entertainment", "Seat.comfort",
  "On.board.service", "Baggage.handling",
  "Checkin.service", "Inflight.service", "Cleanliness"
)

for (var in variables_to_clean) {
  train1 <- train1[train1[[var]] != "0", ]
  train1[[var]] <- factor(train1[[var]])
}

categorical_vars <- c(
  "Gender",
  "Customer.Type", "Type.of.Travel", "Class",
  "Inflight.wifi.service", "Departure.Arrival.time.convenient",
  "Cleanliness", "Food.and.drink", "Online.boarding",
  "Seat.comfort", "Inflight.entertainment", "On.board.service",
  "Leg.room.service", "Baggage.handling", "Checkin.service"
)

for (var in categorical_vars) {
  contingency_table <- table(train1[[var]], train1$satisfaction)
  chi_test <- chisq.test(contingency_table)
  cat("\nChi-Square Test for", var, "vs Satisfaction:\n")
  print(chi_test)
}
```

    ## 
    ## Chi-Square Test for Gender vs Satisfaction:
    ## 
    ##  Pearson's Chi-squared test with Yates' continuity correction
    ## 
    ## data:  contingency_table
    ## X-squared = 0.16077, df = 1, p-value = 0.6884
    ## 
    ## 
    ## Chi-Square Test for Customer.Type vs Satisfaction:
    ## 
    ##  Pearson's Chi-squared test with Yates' continuity correction
    ## 
    ## data:  contingency_table
    ## X-squared = 204.64, df = 1, p-value < 2.2e-16
    ## 
    ## 
    ## Chi-Square Test for Type.of.Travel vs Satisfaction:
    ## 
    ##  Pearson's Chi-squared test with Yates' continuity correction
    ## 
    ## data:  contingency_table
    ## X-squared = 976.55, df = 1, p-value < 2.2e-16
    ## 
    ## 
    ## Chi-Square Test for Class vs Satisfaction:
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  contingency_table
    ## X-squared = 1331.7, df = 2, p-value < 2.2e-16
    ## 
    ## 
    ## Chi-Square Test for Inflight.wifi.service vs Satisfaction:
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  contingency_table
    ## X-squared = 1387.9, df = 5, p-value < 2.2e-16
    ## 
    ## 
    ## Chi-Square Test for Departure.Arrival.time.convenient vs Satisfaction:
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  contingency_table
    ## X-squared = 14.012, df = 5, p-value = 0.01553
    ## 
    ## 
    ## Chi-Square Test for Cleanliness vs Satisfaction:
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  contingency_table
    ## X-squared = 478.63, df = 4, p-value < 2.2e-16
    ## 
    ## 
    ## Chi-Square Test for Food.and.drink vs Satisfaction:
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  contingency_table
    ## X-squared = 230.29, df = 4, p-value < 2.2e-16
    ## 
    ## 
    ## Chi-Square Test for Online.boarding vs Satisfaction:
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  contingency_table
    ## X-squared = 1912.1, df = 5, p-value < 2.2e-16
    ## 
    ## 
    ## Chi-Square Test for Seat.comfort vs Satisfaction:
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  contingency_table
    ## X-squared = 745.74, df = 4, p-value < 2.2e-16
    ## 
    ## 
    ## Chi-Square Test for Inflight.entertainment vs Satisfaction:
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  contingency_table
    ## X-squared = 883.09, df = 4, p-value < 2.2e-16
    ## 
    ## 
    ## Chi-Square Test for On.board.service vs Satisfaction:
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  contingency_table
    ## X-squared = 571.12, df = 4, p-value < 2.2e-16
    ## 
    ## 
    ## Chi-Square Test for Leg.room.service vs Satisfaction:
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  contingency_table
    ## X-squared = 561.69, df = 5, p-value < 2.2e-16
    ## 
    ## 
    ## Chi-Square Test for Baggage.handling vs Satisfaction:
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  contingency_table
    ## X-squared = 410.58, df = 4, p-value < 2.2e-16
    ## 
    ## 
    ## Chi-Square Test for Checkin.service vs Satisfaction:
    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  contingency_table
    ## X-squared = 278.7, df = 4, p-value < 2.2e-16

Regarding numerical variables, the correlation matrix shows that Age and
Flight. Distance have a low positive correlation (0.1147), indicating
minimal association between the two variables. The
Arrival.Delay.in.Minutes and Departure.Delay.in.Minutes are highly
correlated (0.9676), suggesting a strong linear relationship between
them.

The Chi-Square tests show that categorical variables such as
Customer.Type, Type.of.Travel, Class, Inflight.wifi.service,
Cleanliness, Seat.comfort, Inflight.entertainment, On.board.service,
Leg.room.service, Baggage.handling, and Checkin.service all have highly
significant p-values (all \< 2.2e-16), indicating a strong relationship
with satisfaction. In contrast, Gender and
Departure.Arrival.time.convenient have less impact, with p-values of
0.6884 and 0.01553, respectively. These results suggest that the most
important factors for modeling satisfaction are related to customer
type, travel type, flight class, and the quality of services like
inflight entertainment, wifi, cleanliness, and comfort during the
flight. To perform the Chi-Square test, we had to eliminate the
categories of the categorical variables that were empty.

## Numeric Variables

``` r
m0 <- glm(satisfaction ~ 1, data = train, family = binomial)

summary(m0)
```

    ## 
    ## Call:
    ## glm(formula = satisfaction ~ 1, family = binomial, data = train)
    ## 
    ## Coefficients:
    ##             Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept) -0.28827    0.02861  -10.07   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 6812.3  on 4987  degrees of freedom
    ## Residual deviance: 6812.3  on 4987  degrees of freedom
    ## AIC: 6814.3
    ## 
    ## Number of Fisher Scoring iterations: 4

``` r
m1 <- glm(satisfaction ~ Flight.Distance, data = train, family = binomial)

summary(m1)
```

    ## 
    ## Call:
    ## glm(formula = satisfaction ~ Flight.Distance, family = binomial, 
    ##     data = train)
    ## 
    ## Coefficients:
    ##                   Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)     -1.0586096  0.0480933  -22.01   <2e-16 ***
    ## Flight.Distance  0.0006392  0.0000317   20.16   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 6812.3  on 4987  degrees of freedom
    ## Residual deviance: 6355.9  on 4986  degrees of freedom
    ## AIC: 6359.9
    ## 
    ## Number of Fisher Scoring iterations: 4

``` r
anova(m0, m1)
```

    ## Analysis of Deviance Table
    ## 
    ## Model 1: satisfaction ~ 1
    ## Model 2: satisfaction ~ Flight.Distance
    ##   Resid. Df Resid. Dev Df Deviance  Pr(>Chi)    
    ## 1      4987     6812.3                          
    ## 2      4986     6355.9  1   456.35 < 2.2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
m1 <- glm(satisfaction ~ Flight.Distance, data = train, family = binomial)

summary(m1)
```

    ## 
    ## Call:
    ## glm(formula = satisfaction ~ Flight.Distance, family = binomial, 
    ##     data = train)
    ## 
    ## Coefficients:
    ##                   Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)     -1.0586096  0.0480933  -22.01   <2e-16 ***
    ## Flight.Distance  0.0006392  0.0000317   20.16   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 6812.3  on 4987  degrees of freedom
    ## Residual deviance: 6355.9  on 4986  degrees of freedom
    ## AIC: 6359.9
    ## 
    ## Number of Fisher Scoring iterations: 4

``` r
anova(m0, m1)
```

    ## Analysis of Deviance Table
    ## 
    ## Model 1: satisfaction ~ 1
    ## Model 2: satisfaction ~ Flight.Distance
    ##   Resid. Df Resid. Dev Df Deviance  Pr(>Chi)    
    ## 1      4987     6812.3                          
    ## 2      4986     6355.9  1   456.35 < 2.2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
m2 <- glm(satisfaction ~ Flight.Distance + Age, data = train, family = binomial)

summary(m2)
```

    ## 
    ## Call:
    ## glm(formula = satisfaction ~ Flight.Distance + Age, family = binomial, 
    ##     data = train)
    ## 
    ## Coefficients:
    ##                   Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)     -1.712e+00  9.354e-02 -18.301   <2e-16 ***
    ## Flight.Distance  6.178e-04  3.191e-05  19.359   <2e-16 ***
    ## Age              1.709e-02  2.046e-03   8.354   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 6812.3  on 4987  degrees of freedom
    ## Residual deviance: 6285.1  on 4985  degrees of freedom
    ## AIC: 6291.1
    ## 
    ## Number of Fisher Scoring iterations: 4

``` r
anova(m1, m2)
```

    ## Analysis of Deviance Table
    ## 
    ## Model 1: satisfaction ~ Flight.Distance
    ## Model 2: satisfaction ~ Flight.Distance + Age
    ##   Resid. Df Resid. Dev Df Deviance  Pr(>Chi)    
    ## 1      4986     6355.9                          
    ## 2      4985     6285.1  1   70.797 < 2.2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
m3 <- glm(satisfaction ~ Flight.Distance + poly(Age, 2), data = train, family = binomial)

summary(m3)
```

    ## 
    ## Call:
    ## glm(formula = satisfaction ~ Flight.Distance + poly(Age, 2), 
    ##     family = binomial, data = train)
    ## 
    ## Coefficients:
    ##                   Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)     -1.039e+00  4.948e-02 -21.005   <2e-16 ***
    ## Flight.Distance  5.824e-04  3.245e-05  17.947   <2e-16 ***
    ## poly(Age, 2)1    2.250e+01  2.435e+00   9.239   <2e-16 ***
    ## poly(Age, 2)2   -3.648e+01  2.625e+00 -13.900   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 6812.3  on 4987  degrees of freedom
    ## Residual deviance: 6057.4  on 4984  degrees of freedom
    ## AIC: 6065.4
    ## 
    ## Number of Fisher Scoring iterations: 4

``` r
anova(m2, m3)
```

    ## Analysis of Deviance Table
    ## 
    ## Model 1: satisfaction ~ Flight.Distance + Age
    ## Model 2: satisfaction ~ Flight.Distance + poly(Age, 2)
    ##   Resid. Df Resid. Dev Df Deviance  Pr(>Chi)    
    ## 1      4985     6285.1                          
    ## 2      4984     6057.4  1   227.71 < 2.2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
m4 <- glm(satisfaction ~ Flight.Distance + poly(Age, 2), data = train, family = binomial)

summary(m4)
```

    ## 
    ## Call:
    ## glm(formula = satisfaction ~ Flight.Distance + poly(Age, 2), 
    ##     family = binomial, data = train)
    ## 
    ## Coefficients:
    ##                   Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)     -1.039e+00  4.948e-02 -21.005   <2e-16 ***
    ## Flight.Distance  5.824e-04  3.245e-05  17.947   <2e-16 ***
    ## poly(Age, 2)1    2.250e+01  2.435e+00   9.239   <2e-16 ***
    ## poly(Age, 2)2   -3.648e+01  2.625e+00 -13.900   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 6812.3  on 4987  degrees of freedom
    ## Residual deviance: 6057.4  on 4984  degrees of freedom
    ## AIC: 6065.4
    ## 
    ## Number of Fisher Scoring iterations: 4

``` r
anova(m3, m4)
```

    ## Analysis of Deviance Table
    ## 
    ## Model 1: satisfaction ~ Flight.Distance + poly(Age, 2)
    ## Model 2: satisfaction ~ Flight.Distance + poly(Age, 2)
    ##   Resid. Df Resid. Dev Df Deviance Pr(>Chi)
    ## 1      4984     6057.4                     
    ## 2      4984     6057.4  0        0

``` r
m5 <- glm(satisfaction ~ Flight.Distance + poly(Age, 2) + Arrival.Delay.in.Minutes, data = train, family = binomial)

summary(m5)
```

    ## 
    ## Call:
    ## glm(formula = satisfaction ~ Flight.Distance + poly(Age, 2) + 
    ##     Arrival.Delay.in.Minutes, family = binomial, data = train)
    ## 
    ## Coefficients:
    ##                            Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)              -9.942e-01  5.072e-02 -19.601  < 2e-16 ***
    ## Flight.Distance           5.850e-04  3.251e-05  17.997  < 2e-16 ***
    ## poly(Age, 2)1             2.261e+01  2.439e+00   9.270  < 2e-16 ***
    ## poly(Age, 2)2            -3.685e+01  2.631e+00 -14.005  < 2e-16 ***
    ## Arrival.Delay.in.Minutes -3.278e-03  8.733e-04  -3.754 0.000174 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 6812.3  on 4987  degrees of freedom
    ## Residual deviance: 6042.3  on 4983  degrees of freedom
    ## AIC: 6052.3
    ## 
    ## Number of Fisher Scoring iterations: 4

``` r
anova(m4, m5)
```

    ## Analysis of Deviance Table
    ## 
    ## Model 1: satisfaction ~ Flight.Distance + poly(Age, 2)
    ## Model 2: satisfaction ~ Flight.Distance + poly(Age, 2) + Arrival.Delay.in.Minutes
    ##   Resid. Df Resid. Dev Df Deviance Pr(>Chi)    
    ## 1      4984     6057.4                         
    ## 2      4983     6042.3  1   15.163 9.86e-05 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
Anova(m5)
```

    ## Analysis of Deviance Table (Type II tests)
    ## 
    ## Response: satisfaction
    ##                          LR Chisq Df Pr(>Chisq)    
    ## Flight.Distance            357.93  1  < 2.2e-16 ***
    ## poly(Age, 2)               302.60  2  < 2.2e-16 ***
    ## Arrival.Delay.in.Minutes    15.16  1   9.86e-05 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
marginalModelPlots(m5)
```

    ## Warning in mmps(...): Splines and/or polynomials replaced by a fitted linear
    ## combination

![](images/unnamed-chunk-42-1.png)

``` r
residualPlots(m5)
```

![](images/unnamed-chunk-42-2.png)

    ##                          Test stat Pr(>|Test stat|)
    ## Flight.Distance             0.5791           0.4467
    ## poly(Age, 2)                                       
    ## Arrival.Delay.in.Minutes    1.8416           0.1748

To assess the logistic regression model for the target variable
“satisfaction,” we began with the null model, which included only the
intercept, to evaluate the improvement achieved by adding new
predictors. The null deviance of the model was 6812.3, with an AIC of
6814.3. The first predictor added was `Flight.Distance`, which
significantly improved the model fit, reducing the deviance to 6355.9
and the AIC to 6359.9. The second numerical variable included was `Age`,
which also led to a notable improvement. To capture non-linear effects
of `Age`, we refined the model by introducing a second-order polynomial
term for `Age`. Finally, we incorporated `Arrival.Delay.in.Minutes`,
which further improved the model, reducing the deviance to 6042.3 and
the AIC to 6052.3.

To evaluate the improvement of the models, we used the ANOVA test. The
results indicated that each addition of a predictor significantly
improved the model, as evidenced by the reductions in deviance and
corresponding p-values. In summary, the most critical improvements
werethe addition of `Flight.Distance`, the refinement of `Age` using
polynomial terms, and the inclusion of `Arrival.Delay.in.Minutes`, all
of which were confirmed by the ANOVA tests. No significant interaction
terms were identified inj the anova test, thus they are not included in
the model of numeric variables.

# Factor & Factor Interaction Variables

In this section, we apply the stepwise selection method to find a
suitable logistic regression model. We start with an initial model that
includes the previous model with the numerical variables and all the
factors. Then, the step function iteratively evaluates models by adding
or removing predictors to minimize the Akaike Information Criterion
(AIC). Following this, we perform residual analysis to assess the
model’s goodness-of-fit. The result is a simple model we will use later
to comper with a model including interactions.

``` r
par(mfrow = c(1, 1))
# STEP
initial_model <- glm(
  satisfaction ~ Gender + Customer.Type + poly(Age, 2) + Arrival.Delay.in.Minutes + Type.of.Travel +
    Class + Flight.Distance + Inflight.wifi.service +
    Ease.of.Online.booking + Gate.location + Food.and.drink + Online.boarding +
    Seat.comfort + Inflight.entertainment + On.board.service + Leg.room.service +
    Baggage.handling + Checkin.service + Inflight.service + Cleanliness,
  data = train, family = binomial
)

simpler_model <- step(initial_model, direction = "both", trace = 1)
```

    ## Start:  AIC=1932.58
    ## satisfaction ~ Gender + Customer.Type + poly(Age, 2) + Arrival.Delay.in.Minutes + 
    ##     Type.of.Travel + Class + Flight.Distance + Inflight.wifi.service + 
    ##     Ease.of.Online.booking + Gate.location + Food.and.drink + 
    ##     Online.boarding + Seat.comfort + Inflight.entertainment + 
    ##     On.board.service + Leg.room.service + Baggage.handling + 
    ##     Checkin.service + Inflight.service + Cleanliness
    ## 
    ##                            Df Deviance    AIC
    ## - Food.and.drink            5   1800.5 1926.5
    ## - Flight.Distance           1   1796.6 1930.6
    ## - Gender                    1   1796.9 1930.9
    ## - Arrival.Delay.in.Minutes  1   1798.1 1932.1
    ## <none>                          1796.6 1932.6
    ## - Cleanliness               4   1809.1 1937.1
    ## - poly(Age, 2)              2   1808.7 1940.7
    ## - Inflight.service          4   1813.6 1941.6
    ## - On.board.service          4   1817.2 1945.2
    ## - Ease.of.Online.booking    5   1823.1 1949.1
    ## - Checkin.service           4   1825.7 1953.7
    ## - Leg.room.service          5   1829.8 1955.8
    ## - Gate.location             4   1829.6 1957.6
    ## - Class                     2   1828.5 1960.5
    ## - Inflight.entertainment    4   1834.7 1962.7
    ## - Baggage.handling          4   1844.0 1972.0
    ## - Seat.comfort              4   1846.3 1974.3
    ## - Online.boarding           5   1989.3 2115.3
    ## - Customer.Type             1   2071.3 2205.3
    ## - Type.of.Travel            1   2200.3 2334.3
    ## - Inflight.wifi.service     5   2407.6 2533.6
    ## 
    ## Step:  AIC=1926.45
    ## satisfaction ~ Gender + Customer.Type + poly(Age, 2) + Arrival.Delay.in.Minutes + 
    ##     Type.of.Travel + Class + Flight.Distance + Inflight.wifi.service + 
    ##     Ease.of.Online.booking + Gate.location + Online.boarding + 
    ##     Seat.comfort + Inflight.entertainment + On.board.service + 
    ##     Leg.room.service + Baggage.handling + Checkin.service + Inflight.service + 
    ##     Cleanliness
    ## 
    ##                            Df Deviance    AIC
    ## - Flight.Distance           1   1800.5 1924.5
    ## - Gender                    1   1800.9 1924.9
    ## - Arrival.Delay.in.Minutes  1   1801.8 1925.8
    ## <none>                          1800.5 1926.5
    ## - Cleanliness               4   1811.9 1929.9
    ## + Food.and.drink            5   1796.6 1932.6
    ## - poly(Age, 2)              2   1812.4 1934.4
    ## - Inflight.service          4   1818.8 1936.8
    ## - On.board.service          4   1822.0 1940.0
    ## - Ease.of.Online.booking    5   1827.4 1943.4
    ## - Checkin.service           4   1829.6 1947.6
    ## - Gate.location             4   1833.5 1951.5
    ## - Leg.room.service          5   1835.6 1951.6
    ## - Class                     2   1832.0 1954.0
    ## - Baggage.handling          4   1849.8 1967.8
    ## - Inflight.entertainment    4   1849.9 1967.9
    ## - Seat.comfort              4   1850.9 1968.9
    ## - Online.boarding           5   1995.2 2111.2
    ## - Customer.Type             1   2081.3 2205.3
    ## - Type.of.Travel            1   2206.9 2330.9
    ## - Inflight.wifi.service     5   2410.2 2526.2
    ## 
    ## Step:  AIC=1924.45
    ## satisfaction ~ Gender + Customer.Type + poly(Age, 2) + Arrival.Delay.in.Minutes + 
    ##     Type.of.Travel + Class + Inflight.wifi.service + Ease.of.Online.booking + 
    ##     Gate.location + Online.boarding + Seat.comfort + Inflight.entertainment + 
    ##     On.board.service + Leg.room.service + Baggage.handling + 
    ##     Checkin.service + Inflight.service + Cleanliness
    ## 
    ##                            Df Deviance    AIC
    ## - Gender                    1   1800.9 1922.9
    ## - Arrival.Delay.in.Minutes  1   1801.8 1923.8
    ## <none>                          1800.5 1924.5
    ## + Flight.Distance           1   1800.5 1926.5
    ## - Cleanliness               4   1811.9 1927.9
    ## + Food.and.drink            5   1796.6 1930.6
    ## - poly(Age, 2)              2   1812.4 1932.4
    ## - Inflight.service          4   1818.9 1934.9
    ## - On.board.service          4   1822.0 1938.0
    ## - Ease.of.Online.booking    5   1827.4 1941.4
    ## - Checkin.service           4   1829.6 1945.6
    ## - Gate.location             4   1833.5 1949.5
    ## - Leg.room.service          5   1835.6 1949.6
    ## - Class                     2   1835.7 1955.7
    ## - Baggage.handling          4   1849.8 1965.8
    ## - Inflight.entertainment    4   1849.9 1965.9
    ## - Seat.comfort              4   1851.0 1967.0
    ## - Online.boarding           5   1995.2 2109.2
    ## - Customer.Type             1   2100.0 2222.0
    ## - Type.of.Travel            1   2212.2 2334.2
    ## - Inflight.wifi.service     5   2411.3 2525.3
    ## 
    ## Step:  AIC=1922.92
    ## satisfaction ~ Customer.Type + poly(Age, 2) + Arrival.Delay.in.Minutes + 
    ##     Type.of.Travel + Class + Inflight.wifi.service + Ease.of.Online.booking + 
    ##     Gate.location + Online.boarding + Seat.comfort + Inflight.entertainment + 
    ##     On.board.service + Leg.room.service + Baggage.handling + 
    ##     Checkin.service + Inflight.service + Cleanliness
    ## 
    ##                            Df Deviance    AIC
    ## - Arrival.Delay.in.Minutes  1   1802.2 1922.2
    ## <none>                          1800.9 1922.9
    ## + Gender                    1   1800.5 1924.5
    ## + Flight.Distance           1   1800.9 1924.9
    ## - Cleanliness               4   1812.2 1926.2
    ## + Food.and.drink            5   1796.9 1928.9
    ## - poly(Age, 2)              2   1812.9 1930.9
    ## - Inflight.service          4   1819.4 1933.4
    ## - On.board.service          4   1822.7 1936.7
    ## - Ease.of.Online.booking    5   1827.9 1939.9
    ## - Checkin.service           4   1830.2 1944.2
    ## - Gate.location             4   1834.0 1948.0
    ## - Leg.room.service          5   1836.1 1948.1
    ## - Class                     2   1836.0 1954.0
    ## - Baggage.handling          4   1850.2 1964.2
    ## - Inflight.entertainment    4   1850.4 1964.4
    ## - Seat.comfort              4   1851.3 1965.3
    ## - Online.boarding           5   1996.6 2108.6
    ## - Customer.Type             1   2100.0 2220.0
    ## - Type.of.Travel            1   2213.1 2333.1
    ## - Inflight.wifi.service     5   2411.3 2523.3
    ## 
    ## Step:  AIC=1922.22
    ## satisfaction ~ Customer.Type + poly(Age, 2) + Type.of.Travel + 
    ##     Class + Inflight.wifi.service + Ease.of.Online.booking + 
    ##     Gate.location + Online.boarding + Seat.comfort + Inflight.entertainment + 
    ##     On.board.service + Leg.room.service + Baggage.handling + 
    ##     Checkin.service + Inflight.service + Cleanliness
    ## 
    ##                            Df Deviance    AIC
    ## <none>                          1802.2 1922.2
    ## + Arrival.Delay.in.Minutes  1   1800.9 1922.9
    ## + Gender                    1   1801.8 1923.8
    ## + Flight.Distance           1   1802.2 1924.2
    ## - Cleanliness               4   1813.6 1925.6
    ## + Food.and.drink            5   1798.5 1928.5
    ## - poly(Age, 2)              2   1814.5 1930.5
    ## - Inflight.service          4   1820.8 1932.8
    ## - On.board.service          4   1823.7 1935.7
    ## - Ease.of.Online.booking    5   1829.2 1939.2
    ## - Checkin.service           4   1831.5 1943.5
    ## - Leg.room.service          5   1836.9 1946.9
    ## - Gate.location             4   1836.2 1948.2
    ## - Class                     2   1837.9 1953.9
    ## - Baggage.handling          4   1851.0 1963.0
    ## - Inflight.entertainment    4   1851.5 1963.5
    ## - Seat.comfort              4   1852.7 1964.7
    ## - Online.boarding           5   1998.2 2108.2
    ## - Customer.Type             1   2100.4 2218.4
    ## - Type.of.Travel            1   2213.7 2331.7
    ## - Inflight.wifi.service     5   2418.4 2528.4

``` r
#Step:  AIC=1922.22
#satisfaction ~ Customer.Type + poly(Age, 2) + Type.of.Travel + 
#    Class + Inflight.wifi.service + Ease.of.Online.booking + 
#    Gate.location + Online.boarding + Seat.comfort + Inflight.entertainment + 
#    On.board.service + Leg.room.service + Baggage.handling + 
#    Checkin.service + Inflight.service + Cleanliness

# RESIDUAL ANALYSIS OF THE STEP MODEL
marginalModelPlots(simpler_model)
```

    ## Warning in mmps(...): Splines and/or polynomials replaced by a fitted linear
    ## combination

![](images/unnamed-chunk-43-1.png)

    ## Warning in mmps(...): Interactions and/or factors skipped

``` r
residualPlots(simpler_model)
```

![](images/unnamed-chunk-43-2.png)

    ## Warning in residualPlots.default(model, ...): No possible lack-of-fit tests

![](images/unnamed-chunk-43-3.png)

``` r
coef(simpler_model)
```

    ##                   (Intercept)   Customer.TypeLoyal Customer 
    ##                   -1.03564176                    3.31470079 
    ##                 poly(Age, 2)1                 poly(Age, 2)2 
    ##                    7.16401654                  -16.36680440 
    ## Type.of.TravelPersonal Travel                       Class.L 
    ##                   -4.31226890                    0.66532681 
    ##                       Class.Q       Inflight.wifi.service.L 
    ##                    0.40638479                   -9.64083910 
    ##       Inflight.wifi.service.Q       Inflight.wifi.service.C 
    ##                   16.12777562                   -6.77751827 
    ##       Inflight.wifi.service^4       Inflight.wifi.service^5 
    ##                    4.50173384                   -1.61232620 
    ##      Ease.of.Online.booking.L      Ease.of.Online.booking.Q 
    ##                    2.03992103                   -1.48821903 
    ##      Ease.of.Online.booking.C      Ease.of.Online.booking^4 
    ##                    0.19758489                   -0.72667026 
    ##      Ease.of.Online.booking^5               Gate.location.L 
    ##                    0.01817586                   -1.03382670 
    ##               Gate.location.Q               Gate.location.C 
    ##                   -0.37346948                    0.17002630 
    ##               Gate.location^4             Online.boarding.L 
    ##                    0.26543174                    0.41774018 
    ##             Online.boarding.Q             Online.boarding.C 
    ##                    3.28758379                   -0.72117990 
    ##             Online.boarding^4             Online.boarding^5 
    ##                    0.28949442                   -0.34684893 
    ##                Seat.comfort.L                Seat.comfort.Q 
    ##                   -0.40820153                    1.09199951 
    ##                Seat.comfort.C                Seat.comfort^4 
    ##                    0.25019060                   -0.54623647 
    ##      Inflight.entertainment.L      Inflight.entertainment.Q 
    ##                   12.42933201                  -11.30028308 
    ##      Inflight.entertainment.C      Inflight.entertainment^4 
    ##                    5.96803949                   -3.14131103 
    ##      Inflight.entertainment^5            On.board.service.L 
    ##                    1.33685916                   -3.33652427 
    ##            On.board.service.Q            On.board.service.C 
    ##                    4.13285638                   -2.29976200 
    ##            On.board.service^4            On.board.service^5 
    ##                    1.32885950                            NA 
    ##            Leg.room.service.L            Leg.room.service.Q 
    ##                   -1.58529430                    2.60943246 
    ##            Leg.room.service.C            Leg.room.service^4 
    ##                   -1.64123340                    0.62113941 
    ##            Leg.room.service^5            Baggage.handling.L 
    ##                   -0.42581597                    0.67274536 
    ##            Baggage.handling.Q            Baggage.handling.C 
    ##                    0.85816572                    0.28059449 
    ##            Baggage.handling^4             Checkin.service.L 
    ##                   -0.38936113                    0.83280523 
    ##             Checkin.service.Q             Checkin.service.C 
    ##                   -0.21494842                    0.08895653 
    ##             Checkin.service^4            Inflight.service.L 
    ##                    0.30475877                    2.44183451 
    ##            Inflight.service.Q            Inflight.service.C 
    ##                   -1.63121611                    1.93513236 
    ##            Inflight.service^4            Inflight.service^5 
    ##                   -1.07571736                            NA 
    ##                 Cleanliness.L                 Cleanliness.Q 
    ##                    0.67560163                    0.32861707 
    ##                 Cleanliness.C                 Cleanliness^4 
    ##                   -0.06749474                    0.16773355

``` r
Anova(simpler_model, test = "LR")
```

    ## Analysis of Deviance Table (Type II tests)
    ## 
    ## Response: satisfaction
    ##                        LR Chisq Df Pr(>Chisq)    
    ## Customer.Type            298.19  1  < 2.2e-16 ***
    ## poly(Age, 2)              12.26  2  0.0021726 ** 
    ## Type.of.Travel           411.44  1  < 2.2e-16 ***
    ## Class                     35.64  2  1.822e-08 ***
    ## Inflight.wifi.service    616.20  5  < 2.2e-16 ***
    ## Ease.of.Online.booking    26.96  5  5.816e-05 ***
    ## Gate.location             33.98  4  7.535e-07 ***
    ## Online.boarding          196.01  5  < 2.2e-16 ***
    ## Seat.comfort              50.50  4  2.842e-10 ***
    ## Inflight.entertainment    49.27  4  5.120e-10 ***
    ## On.board.service          21.51  4  0.0002511 ***
    ## Leg.room.service          34.67  5  1.748e-06 ***
    ## Baggage.handling          48.79  4  6.462e-10 ***
    ## Checkin.service           29.32  4  6.740e-06 ***
    ## Inflight.service          18.63  4  0.0009308 ***
    ## Cleanliness               11.39  4  0.0225099 *  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

``` r
AIC(simpler_model)
```

    ## [1] 1922.224

Next, we explore interaction terms between predictors. Starting with a
base model that includes all significant predictors identified from the
stepwise selection, we systematically examine potential interactions.
This code is commented out due to the high computational cost and time
it takes to execute.Each pair of predictors is tested by fitting a model
that includes their interaction term and comparing it to the base model
using an ANOVA test. Interaction terms that significantly improve model
fit (p-value \< 0.05) are added to the model.

After identifying all significant interactions, we refit the final
logistic regression model, now including these interaction terms. To
further refine the model, a final stepwise selection is performed to
ensure that only the necessary predictors and interactions are retained.
For the subsequent steps, we will maintain both the simpler model
without interactions and the more complex model with all significant
interactions, validating the performance of each.

``` r
# Define the base formula with no interactions
base_formula <- "Customer.Type + poly(Age, 2) + Type.of.Travel + Class + 
                 Inflight.wifi.service + Ease.of.Online.booking + Gate.location + 
                 Online.boarding + Seat.comfort + Inflight.entertainment + 
                 On.board.service + Leg.room.service + Baggage.handling + 
                 Checkin.service + Inflight.service + Cleanliness"

# List of predictors without interaction terms
predictors <- c("Customer.Type", "poly(Age, 2)", "Type.of.Travel", "Class", 
                "Inflight.wifi.service", "Ease.of.Online.booking", "Gate.location", 
                "Online.boarding", "Seat.comfort", "Inflight.entertainment", 
                "On.board.service", "Leg.room.service", "Baggage.handling", 
                "Checkin.service", "Inflight.service", "Cleanliness")

# Initialize an empty list to store significant interaction terms
significant_interactions <- list()

# Fit the base model using logistic regression (binary outcome: satisfaction)
base_model <- glm(as.formula(paste("satisfaction ~", base_formula)), 
                      data = train, family = binomial)

# Store the AIC of the base model
base_aic <- AIC(base_model)

# Start with the base formula for building the new model
new_model <- base_formula

# Loop over all pairs of predictors to check for significant interactions
#for (i in 1:(length(predictors) - 1)) {
#  for (j in (i + 1):length(predictors)) {
#    # Define the interaction term
#    interaction_term <- paste(predictors[i], predictors[j], sep = ":")
#
#    # Fit the model with the interaction term
#    full_model <- glm(as.formula(paste("satisfaction ~", base_formula, "+", interaction_term)), 
#                      data = train, family = binomial)#
#
#    # Perform ANOVA test to compare models
#    anova_result <- anova(base_model, full_model, test = "Chisq")#
#
#    # Check if the interaction term is significant (p-value < 0.05)
#    if (anova_result$`Pr(>Chi)`[2] < 0.05) {
#      # If the interaction is significant, add it to the list of significant interactions
#      significant_interactions[[length(significant_interactions) + 1]] <- interaction_term
#      
#      # Update the model with the significant interaction term
 #     new_model <- paste0(new_model, "+", interaction_term)
#      
#      base_formula <- paste0(base_formula, "+", interaction_term)
#    }
#  }
#}

# Print the list of significant interactions
#cat("Significant interactions:\n")
#print(significant_interactions)

# Fit the final model with all significant interactions
#new_model <- glm(paste0("satisfaction ~", new_model), data = train, family = binomial)

# Perform final stepwise regression
#stepwise_final_model <- step(new_model, direction = "both", trace = 1)

#Step:  AIC=1420.62
#satisfaction ~ Customer.Type + poly(Age, 2) + Type.of.Travel + 
#    Class + Inflight.wifi.service + Ease.of.Online.booking + 
#    Gate.location + Online.boarding + Seat.comfort + Inflight.entertainment + 
#    On.board.service + Leg.room.service + Baggage.handling + 
#    Checkin.service + Inflight.service + Cleanliness + Customer.Type:poly(Age, 
#    2) + Customer.Type:Inflight.wifi.service + Customer.Type:Gate.location + 
#    Customer.Type:Online.boarding + Customer.Type:Seat.comfort + 
#    Customer.Type:Leg.room.service + Customer.Type:Checkin.service + 
#    Customer.Type:Inflight.service + Customer.Type:Cleanliness + 
#    poly(Age, 2):Online.boarding + poly(Age, 2):Inflight.entertainment + 
#    poly(Age, 2):Inflight.service + poly(Age, 2):Cleanliness + 
#    Type.of.Travel:Class + Type.of.Travel:Inflight.wifi.service + 
#    Type.of.Travel:Ease.of.Online.booking + Type.of.Travel:Online.boarding + 
#    Type.of.Travel:Seat.comfort + Type.of.Travel:Inflight.entertainment + 
#    Type.of.Travel:On.board.service + Type.of.Travel:Baggage.handling + 
#    Type.of.Travel:Checkin.service + Type.of.Travel:Inflight.service + 
#    Class:Inflight.wifi.service + Class:Inflight.entertainment + 
#    Class:On.board.service + Class:Leg.room.service + Type.of.Travel:Gate.location

stepwise_model <- glm(satisfaction ~ Customer.Type + poly(Age, 2) + Type.of.Travel + 
    Class + Inflight.wifi.service + Ease.of.Online.booking + 
    Gate.location + Online.boarding + Seat.comfort + Inflight.entertainment + 
    On.board.service + Leg.room.service + Baggage.handling + 
    Checkin.service + Inflight.service + Cleanliness + Customer.Type:poly(Age, 
    2) + Customer.Type:Inflight.wifi.service + Customer.Type:Gate.location + 
   Customer.Type:Leg.room.service + Customer.Type:Checkin.service + 
    Customer.Type:Inflight.service + Customer.Type:Cleanliness + 
    poly(Age, 2):Online.boarding + poly(Age, 2):Inflight.entertainment + 
   poly(Age, 2):Inflight.service + poly(Age, 2):Cleanliness + 
    Type.of.Travel:Class + Type.of.Travel:Inflight.wifi.service + 
    Type.of.Travel:Ease.of.Online.booking + Type.of.Travel:Online.boarding + 
    Type.of.Travel:Seat.comfort + Type.of.Travel:Inflight.entertainment + 
    Type.of.Travel:On.board.service + Type.of.Travel:Baggage.handling + 
    Type.of.Travel:Checkin.service + Type.of.Travel:Inflight.service + 
    Class:Inflight.wifi.service + Class:Inflight.entertainment + 
    Class:On.board.service + Class:Leg.room.service + Type.of.Travel:Gate.location,
   data = train, family = binomial)
```

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

``` r
marginalModelPlots(stepwise_model)
```

    ## Warning in mmps(...): Splines and/or polynomials replaced by a fitted linear
    ## combination
    ## Warning in mmps(...): glm.fit: fitted probabilities numerically 0 or 1 occurred

![](images/unnamed-chunk-44-1.png)

    ## Warning in mmps(...): Interactions and/or factors skipped

``` r
residualPlots(stepwise_model)
```

![](images/unnamed-chunk-44-2.png)

    ## Warning in residualPlots.default(model, ...): No possible lack-of-fit tests

![](images/unnamed-chunk-44-3.png)

``` r
coef(stepwise_model)
```

    ##                                            (Intercept) 
    ##                                            -0.59442872 
    ##                            Customer.TypeLoyal Customer 
    ##                                             8.08018377 
    ##                                          poly(Age, 2)1 
    ##                                          -251.61447974 
    ##                                          poly(Age, 2)2 
    ##                                            57.39387474 
    ##                          Type.of.TravelPersonal Travel 
    ##                                            -6.67663417 
    ##                                                Class.L 
    ##                                            -5.64878163 
    ##                                                Class.Q 
    ##                                             1.89062601 
    ##                                Inflight.wifi.service.L 
    ##                                            -2.23980660 
    ##                                Inflight.wifi.service.Q 
    ##                                            47.10171462 
    ##                                Inflight.wifi.service.C 
    ##                                           -27.72760346 
    ##                                Inflight.wifi.service^4 
    ##                                            28.28560783 
    ##                                Inflight.wifi.service^5 
    ##                                           -12.98532436 
    ##                               Ease.of.Online.booking.L 
    ##                                             2.85478032 
    ##                               Ease.of.Online.booking.Q 
    ##                                            -3.00762188 
    ##                               Ease.of.Online.booking.C 
    ##                                             1.65419873 
    ##                               Ease.of.Online.booking^4 
    ##                                            -1.06346917 
    ##                               Ease.of.Online.booking^5 
    ##                                            -0.54910732 
    ##                                        Gate.location.L 
    ##                                            -0.38202899 
    ##                                        Gate.location.Q 
    ##                                            -0.62926613 
    ##                                        Gate.location.C 
    ##                                             0.95354932 
    ##                                        Gate.location^4 
    ##                                            -0.17042512 
    ##                                      Online.boarding.L 
    ##                                            19.40107738 
    ##                                      Online.boarding.Q 
    ##                                           -10.82017050 
    ##                                      Online.boarding.C 
    ##                                            11.42685982 
    ##                                      Online.boarding^4 
    ##                                            -4.52912094 
    ##                                      Online.boarding^5 
    ##                                             1.63527629 
    ##                                         Seat.comfort.L 
    ##                                             0.15610811 
    ##                                         Seat.comfort.Q 
    ##                                             2.04940904 
    ##                                         Seat.comfort.C 
    ##                                             0.36161578 
    ##                                         Seat.comfort^4 
    ##                                            -0.37316026 
    ##                               Inflight.entertainment.L 
    ##                                            34.02360838 
    ##                               Inflight.entertainment.Q 
    ##                                           -29.58606545 
    ##                               Inflight.entertainment.C 
    ##                                            16.76266727 
    ##                               Inflight.entertainment^4 
    ##                                            -7.28990406 
    ##                               Inflight.entertainment^5 
    ##                                             1.25972080 
    ##                                     On.board.service.L 
    ##                                            -6.83564814 
    ##                                     On.board.service.Q 
    ##                                             8.29373997 
    ##                                     On.board.service.C 
    ##                                            -4.86706277 
    ##                                     On.board.service^4 
    ##                                             3.34825745 
    ##                                     On.board.service^5 
    ##                                                     NA 
    ##                                     Leg.room.service.L 
    ##                                           -42.80015969 
    ##                                     Leg.room.service.Q 
    ##                                            37.45987896 
    ##                                     Leg.room.service.C 
    ##                                           -26.85448650 
    ##                                     Leg.room.service^4 
    ##                                            13.03404033 
    ##                                     Leg.room.service^5 
    ##                                            -3.78814561 
    ##                                     Baggage.handling.L 
    ##                                             1.30311891 
    ##                                     Baggage.handling.Q 
    ##                                             1.90041711 
    ##                                     Baggage.handling.C 
    ##                                             0.06588359 
    ##                                     Baggage.handling^4 
    ##                                            -0.46386494 
    ##                                      Checkin.service.L 
    ##                                             0.33781086 
    ##                                      Checkin.service.Q 
    ##                                            -1.21629305 
    ##                                      Checkin.service.C 
    ##                                            -0.84938477 
    ##                                      Checkin.service^4 
    ##                                             0.35563455 
    ##                                     Inflight.service.L 
    ##                                             7.06322059 
    ##                                     Inflight.service.Q 
    ##                                            -4.20319868 
    ##                                     Inflight.service.C 
    ##                                             4.06009417 
    ##                                     Inflight.service^4 
    ##                                            -3.74823571 
    ##                                     Inflight.service^5 
    ##                                                     NA 
    ##                                          Cleanliness.L 
    ##                                             0.41967869 
    ##                                          Cleanliness.Q 
    ##                                            -1.87552566 
    ##                                          Cleanliness.C 
    ##                                            -0.11041480 
    ##                                          Cleanliness^4 
    ##                                             0.45467524 
    ##              Customer.TypeLoyal Customer:poly(Age, 2)1 
    ##                                            -8.96843499 
    ##              Customer.TypeLoyal Customer:poly(Age, 2)2 
    ##                                           -60.91703650 
    ##    Customer.TypeLoyal Customer:Inflight.wifi.service.L 
    ##                                           -17.37959018 
    ##    Customer.TypeLoyal Customer:Inflight.wifi.service.Q 
    ##                                           -20.08823744 
    ##    Customer.TypeLoyal Customer:Inflight.wifi.service.C 
    ##                                             5.34019717 
    ##    Customer.TypeLoyal Customer:Inflight.wifi.service^4 
    ##                                           -13.55595052 
    ##    Customer.TypeLoyal Customer:Inflight.wifi.service^5 
    ##                                             6.33127814 
    ##            Customer.TypeLoyal Customer:Gate.location.L 
    ##                                             0.36923379 
    ##            Customer.TypeLoyal Customer:Gate.location.Q 
    ##                                             0.25025413 
    ##            Customer.TypeLoyal Customer:Gate.location.C 
    ##                                            -1.51650319 
    ##            Customer.TypeLoyal Customer:Gate.location^4 
    ##                                             1.15176311 
    ##         Customer.TypeLoyal Customer:Leg.room.service.L 
    ##                                             8.27679235 
    ##         Customer.TypeLoyal Customer:Leg.room.service.Q 
    ##                                            -4.14483559 
    ##         Customer.TypeLoyal Customer:Leg.room.service.C 
    ##                                             3.61609583 
    ##         Customer.TypeLoyal Customer:Leg.room.service^4 
    ##                                            -2.03731443 
    ##         Customer.TypeLoyal Customer:Leg.room.service^5 
    ##                                                     NA 
    ##          Customer.TypeLoyal Customer:Checkin.service.L 
    ##                                            14.82634313 
    ##          Customer.TypeLoyal Customer:Checkin.service.Q 
    ##                                            12.93954107 
    ##          Customer.TypeLoyal Customer:Checkin.service.C 
    ##                                             8.13615211 
    ##          Customer.TypeLoyal Customer:Checkin.service^4 
    ##                                             2.74796341 
    ##         Customer.TypeLoyal Customer:Inflight.service.L 
    ##                                            -5.40012275 
    ##         Customer.TypeLoyal Customer:Inflight.service.Q 
    ##                                             4.45153735 
    ##         Customer.TypeLoyal Customer:Inflight.service.C 
    ##                                            -0.82700450 
    ##         Customer.TypeLoyal Customer:Inflight.service^4 
    ##                                             3.30973338 
    ##         Customer.TypeLoyal Customer:Inflight.service^5 
    ##                                                     NA 
    ##              Customer.TypeLoyal Customer:Cleanliness.L 
    ##                                             0.97374580 
    ##              Customer.TypeLoyal Customer:Cleanliness.Q 
    ##                                             0.99787086 
    ##              Customer.TypeLoyal Customer:Cleanliness.C 
    ##                                             0.40034625 
    ##              Customer.TypeLoyal Customer:Cleanliness^4 
    ##                                            -0.30521734 
    ##                        poly(Age, 2)1:Online.boarding.L 
    ##                                           949.67774949 
    ##                        poly(Age, 2)2:Online.boarding.L 
    ##                                            89.76554276 
    ##                        poly(Age, 2)1:Online.boarding.Q 
    ##                                          -952.71047344 
    ##                        poly(Age, 2)2:Online.boarding.Q 
    ##                                           -43.24719200 
    ##                        poly(Age, 2)1:Online.boarding.C 
    ##                                           758.67211006 
    ##                        poly(Age, 2)2:Online.boarding.C 
    ##                                            -3.70244628 
    ##                        poly(Age, 2)1:Online.boarding^4 
    ##                                          -404.68447402 
    ##                        poly(Age, 2)2:Online.boarding^4 
    ##                                           -31.31710559 
    ##                        poly(Age, 2)1:Online.boarding^5 
    ##                                           108.90122685 
    ##                        poly(Age, 2)2:Online.boarding^5 
    ##                                            38.82707788 
    ##                 poly(Age, 2)1:Inflight.entertainment.L 
    ##                                           199.87902379 
    ##                 poly(Age, 2)2:Inflight.entertainment.L 
    ##                                           255.02436278 
    ##                 poly(Age, 2)1:Inflight.entertainment.Q 
    ##                                          -288.17176296 
    ##                 poly(Age, 2)2:Inflight.entertainment.Q 
    ##                                           -89.22955501 
    ##                 poly(Age, 2)1:Inflight.entertainment.C 
    ##                                            37.91861953 
    ##                 poly(Age, 2)2:Inflight.entertainment.C 
    ##                                           215.63165188 
    ##                 poly(Age, 2)1:Inflight.entertainment^4 
    ##                                           -14.46370589 
    ##                 poly(Age, 2)2:Inflight.entertainment^4 
    ##                                           -52.68479149 
    ##                 poly(Age, 2)1:Inflight.entertainment^5 
    ##                                                     NA 
    ##                 poly(Age, 2)2:Inflight.entertainment^5 
    ##                                                     NA 
    ##                       poly(Age, 2)1:Inflight.service.L 
    ##                                          -151.36544096 
    ##                       poly(Age, 2)2:Inflight.service.L 
    ##                                          -470.41814179 
    ##                       poly(Age, 2)1:Inflight.service.Q 
    ##                                           123.78509360 
    ##                       poly(Age, 2)2:Inflight.service.Q 
    ##                                           355.03914732 
    ##                       poly(Age, 2)1:Inflight.service.C 
    ##                                          -148.86293990 
    ##                       poly(Age, 2)2:Inflight.service.C 
    ##                                          -230.57118938 
    ##                       poly(Age, 2)1:Inflight.service^4 
    ##                                            57.12058760 
    ##                       poly(Age, 2)2:Inflight.service^4 
    ##                                            79.34369518 
    ##                       poly(Age, 2)1:Inflight.service^5 
    ##                                                     NA 
    ##                       poly(Age, 2)2:Inflight.service^5 
    ##                                                     NA 
    ##                            poly(Age, 2)1:Cleanliness.L 
    ##                                            25.18795646 
    ##                            poly(Age, 2)2:Cleanliness.L 
    ##                                           -68.09522370 
    ##                            poly(Age, 2)1:Cleanliness.Q 
    ##                                           186.62910287 
    ##                            poly(Age, 2)2:Cleanliness.Q 
    ##                                          -107.10861638 
    ##                            poly(Age, 2)1:Cleanliness.C 
    ##                                           -12.05968655 
    ##                            poly(Age, 2)2:Cleanliness.C 
    ##                                           -14.30672378 
    ##                            poly(Age, 2)1:Cleanliness^4 
    ##                                            31.25587718 
    ##                            poly(Age, 2)2:Cleanliness^4 
    ##                                           -12.06035947 
    ##                  Type.of.TravelPersonal Travel:Class.L 
    ##                                             1.16530054 
    ##                  Type.of.TravelPersonal Travel:Class.Q 
    ##                                             5.19365208 
    ##  Type.of.TravelPersonal Travel:Inflight.wifi.service.L 
    ##                                            25.28494417 
    ##  Type.of.TravelPersonal Travel:Inflight.wifi.service.Q 
    ##                                            82.36197263 
    ##  Type.of.TravelPersonal Travel:Inflight.wifi.service.C 
    ##                                           -15.91096291 
    ##  Type.of.TravelPersonal Travel:Inflight.wifi.service^4 
    ##                                             9.04648618 
    ##  Type.of.TravelPersonal Travel:Inflight.wifi.service^5 
    ##                                            -9.76531886 
    ## Type.of.TravelPersonal Travel:Ease.of.Online.booking.L 
    ##                                             0.87594789 
    ## Type.of.TravelPersonal Travel:Ease.of.Online.booking.Q 
    ##                                           -44.21978792 
    ## Type.of.TravelPersonal Travel:Ease.of.Online.booking.C 
    ##                                            21.46271216 
    ## Type.of.TravelPersonal Travel:Ease.of.Online.booking^4 
    ##                                           -15.07530508 
    ## Type.of.TravelPersonal Travel:Ease.of.Online.booking^5 
    ##                                            -3.07421286 
    ##        Type.of.TravelPersonal Travel:Online.boarding.L 
    ##                                           -30.48776669 
    ##        Type.of.TravelPersonal Travel:Online.boarding.Q 
    ##                                            32.89613126 
    ##        Type.of.TravelPersonal Travel:Online.boarding.C 
    ##                                           -26.32872407 
    ##        Type.of.TravelPersonal Travel:Online.boarding^4 
    ##                                             4.35282520 
    ##        Type.of.TravelPersonal Travel:Online.boarding^5 
    ##                                             7.29389302 
    ##           Type.of.TravelPersonal Travel:Seat.comfort.L 
    ##                                             0.26977522 
    ##           Type.of.TravelPersonal Travel:Seat.comfort.Q 
    ##                                            -4.16101826 
    ##           Type.of.TravelPersonal Travel:Seat.comfort.C 
    ##                                            -0.30692529 
    ##           Type.of.TravelPersonal Travel:Seat.comfort^4 
    ##                                             0.20572449 
    ## Type.of.TravelPersonal Travel:Inflight.entertainment.L 
    ##                                           -14.72529894 
    ## Type.of.TravelPersonal Travel:Inflight.entertainment.Q 
    ##                                            13.57331290 
    ## Type.of.TravelPersonal Travel:Inflight.entertainment.C 
    ##                                            -7.21817754 
    ## Type.of.TravelPersonal Travel:Inflight.entertainment^4 
    ##                                             4.52679485 
    ## Type.of.TravelPersonal Travel:Inflight.entertainment^5 
    ##                                                     NA 
    ##       Type.of.TravelPersonal Travel:On.board.service.L 
    ##                                             5.44749937 
    ##       Type.of.TravelPersonal Travel:On.board.service.Q 
    ##                                            -7.81924322 
    ##       Type.of.TravelPersonal Travel:On.board.service.C 
    ##                                             5.17805686 
    ##       Type.of.TravelPersonal Travel:On.board.service^4 
    ##                                            -1.73769628 
    ##       Type.of.TravelPersonal Travel:On.board.service^5 
    ##                                                     NA 
    ##       Type.of.TravelPersonal Travel:Baggage.handling.L 
    ##                                            -0.24222264 
    ##       Type.of.TravelPersonal Travel:Baggage.handling.Q 
    ##                                            -2.60758616 
    ##       Type.of.TravelPersonal Travel:Baggage.handling.C 
    ##                                             0.37920265 
    ##       Type.of.TravelPersonal Travel:Baggage.handling^4 
    ##                                             0.26173843 
    ##        Type.of.TravelPersonal Travel:Checkin.service.L 
    ##                                           -15.63013662 
    ##        Type.of.TravelPersonal Travel:Checkin.service.Q 
    ##                                           -12.63644219 
    ##        Type.of.TravelPersonal Travel:Checkin.service.C 
    ##                                            -6.93857144 
    ##        Type.of.TravelPersonal Travel:Checkin.service^4 
    ##                                            -3.85387170 
    ##       Type.of.TravelPersonal Travel:Inflight.service.L 
    ##                                           -15.29832485 
    ##       Type.of.TravelPersonal Travel:Inflight.service.Q 
    ##                                            14.12756753 
    ##       Type.of.TravelPersonal Travel:Inflight.service.C 
    ##                                           -12.73081224 
    ##       Type.of.TravelPersonal Travel:Inflight.service^4 
    ##                                             4.00106679 
    ##       Type.of.TravelPersonal Travel:Inflight.service^5 
    ##                                                     NA 
    ##                        Class.L:Inflight.wifi.service.L 
    ##                                             2.56228029 
    ##                        Class.Q:Inflight.wifi.service.L 
    ##                                             7.91665496 
    ##                        Class.L:Inflight.wifi.service.Q 
    ##                                            -6.69391128 
    ##                        Class.Q:Inflight.wifi.service.Q 
    ##                                           -13.55018264 
    ##                        Class.L:Inflight.wifi.service.C 
    ##                                            12.89365107 
    ##                        Class.Q:Inflight.wifi.service.C 
    ##                                            12.33579083 
    ##                        Class.L:Inflight.wifi.service^4 
    ##                                           -10.03296489 
    ##                        Class.Q:Inflight.wifi.service^4 
    ##                                            -5.37000961 
    ##                        Class.L:Inflight.wifi.service^5 
    ##                                             5.06889892 
    ##                        Class.Q:Inflight.wifi.service^5 
    ##                                             4.20914069 
    ##                       Class.L:Inflight.entertainment.L 
    ##                                            -1.00683862 
    ##                       Class.Q:Inflight.entertainment.L 
    ##                                           -53.66510734 
    ##                       Class.L:Inflight.entertainment.Q 
    ##                                             2.10194777 
    ##                       Class.Q:Inflight.entertainment.Q 
    ##                                            45.84923428 
    ##                       Class.L:Inflight.entertainment.C 
    ##                                            -2.19754359 
    ##                       Class.Q:Inflight.entertainment.C 
    ##                                           -23.50240534 
    ##                       Class.L:Inflight.entertainment^4 
    ##                                             2.33375625 
    ##                       Class.Q:Inflight.entertainment^4 
    ##                                             5.94416144 
    ##                       Class.L:Inflight.entertainment^5 
    ##                                                     NA 
    ##                       Class.Q:Inflight.entertainment^5 
    ##                                                     NA 
    ##                             Class.L:On.board.service.L 
    ##                                             3.14010445 
    ##                             Class.Q:On.board.service.L 
    ##                                            12.11923841 
    ##                             Class.L:On.board.service.Q 
    ##                                            -2.14129412 
    ##                             Class.Q:On.board.service.Q 
    ##                                           -10.17914839 
    ##                             Class.L:On.board.service.C 
    ##                                             0.84809081 
    ##                             Class.Q:On.board.service.C 
    ##                                             7.04445721 
    ##                             Class.L:On.board.service^4 
    ##                                            -0.33093609 
    ##                             Class.Q:On.board.service^4 
    ##                                            -5.24865404 
    ##                             Class.L:On.board.service^5 
    ##                                                     NA 
    ##                             Class.Q:On.board.service^5 
    ##                                                     NA 
    ##                             Class.L:Leg.room.service.L 
    ##                                            24.58735294 
    ##                             Class.Q:Leg.room.service.L 
    ##                                            34.30847474 
    ##                             Class.L:Leg.room.service.Q 
    ##                                           -20.67775686 
    ##                             Class.Q:Leg.room.service.Q 
    ##                                           -30.50296920 
    ##                             Class.L:Leg.room.service.C 
    ##                                            15.07913619 
    ##                             Class.Q:Leg.room.service.C 
    ##                                            22.72213104 
    ##                             Class.L:Leg.room.service^4 
    ##                                            -7.23300606 
    ##                             Class.Q:Leg.room.service^4 
    ##                                            -9.85272719 
    ##                             Class.L:Leg.room.service^5 
    ##                                             2.63153868 
    ##                             Class.Q:Leg.room.service^5 
    ##                                             2.75627565 
    ##          Type.of.TravelPersonal Travel:Gate.location.L 
    ##                                            -0.53248501 
    ##          Type.of.TravelPersonal Travel:Gate.location.Q 
    ##                                             1.08641400 
    ##          Type.of.TravelPersonal Travel:Gate.location.C 
    ##                                            -0.33110220 
    ##          Type.of.TravelPersonal Travel:Gate.location^4 
    ##                                            -1.17756877

``` r
AIC(stepwise_model)
```

    ## [1] 1435.286

# 5. Model Validation

Using the training dataset, we have built a logistic regression model to
predict passenger satisfaction. Now, we will evaluate the model’s
performance using the testing dataset.

Let’s start by predicting the satisfaction levels for the testing
dataset.

``` r
Satisfaction.Prob <- predict(stepwise_model, newdata = test, type = "response")
```

Due to the `type="response"` parameter, the new column
“Satisfaction.Prob” contains the predicted probabilities of satisfaction
for each passenger. To decide whether a passenger is satisfied or not, a
threshold must be set. For example, let’s set a threshold of 0.5.

``` r
Predicted.Satisfaction <- ifelse(Satisfaction.Prob > 0.5, "satisfied", "neutral or dissatisfied")
```

And with that, a confusion matrix can be created to evaluate the model’s
performance.

``` r
conf_matrix <- table(test$satisfaction, Predicted.Satisfaction)
```

To calculate the performance metrics, we start from the confusion
matrix’s elements:

``` r
TP <- conf_matrix[2, 2]
TN <- conf_matrix[1, 1]
FP <- conf_matrix[1, 2]
FN <- conf_matrix[2, 1]
```

We define “Accuracy” as the Correct Classification Rate:

$$
CCR = \frac{TP + TN}{TP + TN + FP + FN}
$$

``` r
(TP + TN) / sum(conf_matrix)
```

    ## [1] 0.938

We define “Sensitivity” or “Recall” as the True Positive Rate:

$$
TPR = \frac{TP}{TP + FN}
$$

``` r
TP / (TP + FN)
```

    ## [1] 0.9297297

And we define “Fall-out” as the False Positive Rate:

$$
FPR = \frac{FP}{FP + TN}
$$

``` r
FP / (FP + TN)
```

    ## [1] 0.05488372

The process of designing a model and choosing a threshold consists of a
careful trade-off between sensitivity and fall-out. If we plot these
metrics against different thresholds, we can visualize this trade-off in
what is known as a Receiver Operating Characteristic (ROC) curve:

``` r
par(mfrow = c(1, 1))
roc <- roc(test$satisfaction, Satisfaction.Prob, quiet = TRUE)
plot(roc, main = "ROC Curve", col = "blue")
polygon(c(roc$specificities, 0), c(roc$sensitivities, 0), col = adjustcolor("blue", alpha.f = 0.2), border = NA)
```

![](images/unnamed-chunk-52-1.png)

A model with perfect discrimination has an ROC curve that passes through
the upper-left corner, whilst a model with no discrimination has an ROC
curve that passes through the diagonal. Thus, the area under the ROC
curve (AUC) is a good measure of the model’s performance. The closer the
AUC is to 1, the better the model is at distinguishing between satisfied
and dissatisfied passengers.

``` r
auc(roc)
```

    ## Area under the curve: 0.983

This high AUC value indicates that the model is fairly good at
distinguishing between satisfied and dissatisfied passengers.

## Which model should we choose?

In the previous section, we arrived at two models: `stepwise_model`,
which was constructed by minimizing the Akaike Information Criterion
(AIC) using the `step` function, and `simpler_model`, which was a
manually constructed model based On the most significant variables of
the former.

The following code executes a prediction on the testing dataset for the
latter model, and computes the ROC curve.

``` r
Satisfaction.Prob.Simpler <- predict(simpler_model, newdata = test, type = "response")
roc.Simpler <- roc(test$satisfaction, Satisfaction.Prob.Simpler, quiet = TRUE)
plot(roc.Simpler, col = "red", main = "ROC Curves")
polygon(c(roc.Simpler$specificities, 0), c(roc.Simpler$sensitivities, 0), col = adjustcolor("red", alpha.f = 0.2), border = NA)
lines(roc, col = "blue", lwd = 2)
polygon(c(roc$specificities, 0), c(roc$sensitivities, 0), col = adjustcolor("blue", alpha.f = 0.2), border = NA)
legend(
  "bottomright",
  legend = c("Stepwise Model", "Simpler Model"),
  col = c("blue", "red"), lwd = 2, fill = adjustcolor(c("blue", "red"), alpha.f = 0.2)
)
```

![](images/unnamed-chunk-54-1.png)

When comparing the two curves, we can see that ….

<!-- I cannot run the stepwise model!!!! I cannot compare the two curves -->

And looking at the areas under the ROC curves, it becomes clear that,
indeed, the stepwise model is slightly better, but it’s complexity might
not be worth the small improvement in performance, according to the AUC
criterion.

``` r
print(paste("Area under the stepwise curve:", auc(roc)))
```

    ## [1] "Area under the stepwise curve: 0.983016467630421"

``` r
print(paste("Area under the simpler curve:", auc(roc.Simpler)))
```

    ## [1] "Area under the simpler curve: 0.974748962916405"

To further compare them, let’s first choose a suitable threshold for
both of them. We’ll choose the threshold that maximizes the F1 score,
which is the harmonic mean of precision and recall, computed as follows:

$$
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$

``` r
f1_computer <- function(expected, probs) {
  return(function(threshold) {
    pred <- factor(
      ifelse(probs > threshold, "satisfied", "neutral or dissatisfied"),
      levels = c("satisfied", "neutral or dissatisfied")
    )
    conf <- table(expected, pred)
    TP <- conf["satisfied", "satisfied"]
    FP <- conf["neutral or dissatisfied", "satisfied"]
    FN <- conf["satisfied", "neutral or dissatisfied"]
    precision <- TP / (TP + FP)
    recall <- TP / (TP + FN)
    return(2 * precision * recall / (precision + recall + 1e-10)) # Avoid division by zero
  })
}

thresholds <- seq(0, 1, 0.01)
stepwise_f1 <- sapply(thresholds, f1_computer(test$satisfaction, Satisfaction.Prob))
simpler_f1 <- sapply(thresholds, f1_computer(test$satisfaction, Satisfaction.Prob.Simpler))

plot(thresholds, stepwise_f1, type = "l", col = "blue", xlab = "Threshold", ylab = "F1 Score", main = "F1 Score vs Threshold")
lines(thresholds, simpler_f1, col = "red")
legend("topright", legend = c("Stepwise Model", "Simpler Model"), col = c("blue", "red"), lwd = 2)
```

![](images/unnamed-chunk-56-1.png)

The thresholds are:

``` r
stepwise_threshold <- thresholds[which.max(stepwise_f1)]
simpler_threshold <- thresholds[which.max(simpler_f1)]
print(paste("Stepwise Model Threshold:", stepwise_threshold))
```

    ## [1] "Stepwise Model Threshold: 0.47"

``` r
print(paste("Simpler Model Threshold:", simpler_threshold))
```

    ## [1] "Simpler Model Threshold: 0.52"

Now, using them, here’s their confusion matrices:

``` r
stepwise_pred <- factor(
  ifelse(Satisfaction.Prob > stepwise_threshold, "satisfied", "neutral or dissatisfied"),
  levels = c("satisfied", "neutral or dissatisfied")
)
simpler_pred <- factor(
  ifelse(Satisfaction.Prob.Simpler > simpler_threshold, "satisfied", "neutral or dissatisfied"),
  levels = c("satisfied", "neutral or dissatisfied")
)

stepwise_conf_matrix <- table(test$satisfaction, stepwise_pred)
stepwise_conf_matrix
```

    ##                          stepwise_pred
    ##                           satisfied neutral or dissatisfied
    ##   neutral or dissatisfied        62                    1013
    ##   satisfied                     866                      59

``` r
simpler_conf_matrix <- table(test$satisfaction, simpler_pred)
simpler_conf_matrix
```

    ##                          simpler_pred
    ##                           satisfied neutral or dissatisfied
    ##   neutral or dissatisfied        51                    1024
    ##   satisfied                     845                      80

And their accuracies:

``` r
stepwise_accuracy <- sum(diag(stepwise_conf_matrix)) / sum(stepwise_conf_matrix)
stepwise_accuracy
```

    ## [1] 0.0605

``` r
simpler_accuracy <- sum(diag(simpler_conf_matrix)) / sum(simpler_conf_matrix)
simpler_accuracy
```

    ## [1] 0.0655

As we can see…
<!-- Comment on which model has the highest threshold, and the highest accuracy -->

# 6. Conclusion

<!-- Interpret the results :) -->
