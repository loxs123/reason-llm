import math

# Function to calculate double factorial
def double_factorial(n):
    if n <= 0:
        return 1
    return n * double_factorial(n - 2)

# Total number of players
total_players = 4048

# Total number of ways to pair 4048 players
total_ways = double_factorial(total_players - 1)

# Number of ways Fred and George are paired together
# If Fred and George are paired together, we need to pair the remaining 4046 players
ways_fred_george_together = double_factorial(total_players - 3)

# Number of ways Fred and George do not play each other
ways_fred_george_not_together = total_ways - ways_fred_george_together

print(ways_fred_george_not_together)