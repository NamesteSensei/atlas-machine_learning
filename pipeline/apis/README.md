# Data Collection - APIs

## Project Overview

This project focuses on retrieving and transforming data from external APIs using Python.

The objective is to build foundational backend skills by:
- Making HTTP requests
- Handling pagination
- Handling rate limits
- Parsing JSON responses
- Performing API chaining
- Aggregating and sorting data

All scripts are written for Ubuntu 20.04 using Python 3.9 and follow PEP8 style guidelines.

---

## Technologies Used

- Python 3.9
- requests library
- REST APIs
- Git & GitHub
- Bash (for CLI testing)

---

## Tasks

### Task 0 - Available Ships

**File:** `0-passengers.py`

Retrieve starships from the Star Wars API (SWAPI) that can carry at least a given number of passengers.

Concepts:
- HTTP GET requests
- Pagination handling
- Numeric filtering
- Data cleaning (removing commas)

---

### Task 1 - Sentient Planets

**File:** `1-sentience.py`

Retrieve home planets of all sentient species from SWAPI.

Concepts:
- Pagination
- String filtering
- API chaining (species → planet)

---

### Task 2 - GitHub User Location

**File:** `2-user_location.py`

CLI script that retrieves the location of a GitHub user.

Handles:
- 200 (Success)
- 404 (Not Found)
- 403 (Rate limit exceeded)

Concepts:
- Command-line arguments
- HTTP headers
- Rate limit handling
- UNIX timestamp calculation

---

### Task 3 - First Upcoming Launch

**File:** `3-first_launch.py`

Displays the first upcoming SpaceX launch including:
- Launch name
- Date (local time)
- Rocket name
- Launchpad name and locality

Concepts:
- Sorting by `date_unix`
- API chaining (launch → rocket → launchpad)
- Structured output formatting

---

### Task 4 - Rocket Launch Frequency

**File:** `4-rocket_frequency.py`

Displays the number of launches per rocket.

Requirements:
- All launches considered
- Sorted by number of launches (descending)
- Alphabetical tie-breaker

Concepts:
- Aggregation using dictionaries
- Grouping logic
- Multi-key sorting

---

## How to Run

Make scripts executable:

Run example:
/0-passengers.py
./1-sentience.py
./2-user_location.py <GitHub_API_URL>
./3-first_launch.py
./4-rocket_frequency.py

## Key Concepts Learned

- REST API consumption
- Pagination handling
- API chaining
- Rate limit handling
- Data aggregation
- Efficient querying patterns
- Multi-condition sorting
- CLI script structure

---

## Author

Christopher  
Atlas Machine Learning
