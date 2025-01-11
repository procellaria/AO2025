# Tennis Tournament Simulator

A Monte Carlo simulation tool for predicting outcomes of the Australian Open tennis tournament. The simulator uses player rankings and other relevant data to calculate win probabilities, possible matchups, and tournament progression statistics.

## Features

- Monte Carlo simulation of complete tournament brackets
- Calculation of winning probabilities for each player
- Prediction of most likely finals and semi-finals
- Wilson confidence intervals for all statistical calculations
- Detailed statistics for each player's tournament progression
- Web interface for easy interaction

## Requirements

- Python 3.8 or higher
- Required Python packages:
  - streamlit
  - pandas
  - numpy
  - scipy
  - openpyxl

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/tennis-tournament-simulator.git
cd tennis-tournament-simulator
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Local Development

To run the application locally:

```bash
streamlit run app.py
```

Then open your web browser and go to `http://localhost:8501`

### Input Data Format

The application expects an Excel file (.xlsx) with the following structure:

- Column 1: Player names in tournament bracket order
  - **IMPORTANT**: Players must be listed in the exact order of the tournament draw
  - The order determines the tournament bracket structure and matchups
  - First two players will meet in first round, next two players will meet in first round, and so on
  - This ordering ensures the simulation follows the actual tournament bracket structure
- Column 2: Base strength coefficients (ATP points)
- Column 3: Bonus points
- Column 4: Player status (1 = active, 0 = eliminated)

Example ordering for a simplified 8-player tournament:
```
Player1 (meets Player2 in first round)
Player2 (meets Player1 in first round)
Player3 (meets Player4 in first round)
Player4 (meets Player3 in first round)
Player5 (meets Player6 in first round)
Player6 (meets Player5 in first round)
Player7 (meets Player8 in first round)
Player8 (meets Player7 in first round)
```

Winner of matches 1-2 will meet winner of matches 3-4, and winner of matches 5-6 will meet winner of matches 7-8 in the next round.

### Using the Web Interface

1. Upload your Excel file using the file uploader
2. Adjust the number of simulations using the slider (default: 1000)
3. Click "Start Simulation" to run the analysis
4. View the results and download the complete statistics file

## Output

The simulator provides:
- Win probabilities for each player with confidence intervals
- Most likely final and semi-final matchups
- Detailed progression statistics for each player
- Downloadable text file with complete statistics

## Deploy to Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Deploy your forked repository

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Authors

- Procellaria

## Acknowledgments

- Based on Monte Carlo simulation techniques
- Uses Wilson score interval for confidence calculations
- Inspired by professional tennis tournament structures