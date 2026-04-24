# RFP Bid Response Evaluator

RFP Bid Response Evaluator is a Streamlit-based web application that utilizes a local Large Language Model (LLM) to evaluate draft bid responses against Request for Proposal (RFP) requirements, win themes, and Shipley best practices. It employs a multi-agent system combined with Python-native lexical analysis.

## Features

This application evaluates your draft response using three distinct AI agents working alongside a Python heuristic engine:

### 1. Python Lexical Analysis Engine
Before the agents analyze the text, a pure Python heuristic engine runs to calculate basic metrics:
- **Win Theme Counter:** Counts the occurrences of specific win themes in your text.
- **Shipley Metrics:** Checks for customer focus by calculating the ratio of client name mentions to inward-facing pronouns ("we", "us", "our").

### 2. The AI Agents
The application uses a sequential multi-agent pipeline using an LLM to evaluate the text:

* **Agent 1: Compliance Agent**
  * **Role:** Analyzes if the draft response meets ALL the functional requirements asked in the RFP question.
  * **Output:** Provides a brief explanation and a definite status of `PASS`, `FAIL`, or `PARTIAL`.

* **Agent 2: Theme Qualifier Agent**
  * **Role:** Takes the lexical occurrences of the win themes and conceptually evaluates if they are effectively woven into the narrative, rather than just being name-dropped.
  * **Output:** Provides a brief paragraph explanation and a "Theme Quality Score" percentage.

* **Agent 3: Shipley Grader Agent**
  * **Role:** Evaluates the draft for active voice, clarity, and structural Shipley customer-focus based on the lexical statistics.
  * **Output:** Provides feedback on active voice and clarity, followed by a final "Shipley Score" out of 10.

## Prerequisites

- **Python 3.8+**
- **LM Studio:** You need to have LM Studio installed and running a local server to provide the LLM capabilities. The app is optimized for smaller models to ensure low CPU utilization and fast responses.

## Installation & Setup

1. **Clone the repository:**
   (If you haven't already, clone this directory to your local machine).

2. **Install the dependencies:**
   Open a terminal in the project directory and run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start your Local LLM Server:**
   * Open **LM Studio**.
   * Load your preferred model (e.g., `Nemotron-3-nano-4b` or any other model you prefer).
   * Start the Local Server. Ensure it is running on `http://localhost:1234/v1` (the default for LM Studio).

4. **Run the Application:**
   In your terminal, within the project directory, run:
   ```bash
   streamlit run app.py
   ```

5. **Using the App:**
   * The app will open in your default web browser.
   * Adjust the "Local Server Config" in the sidebar if your LM Studio uses a different endpoint or model name.
   * Provide your "Win Themes" and "Target Client Name" in the sidebar.
   * Paste the "RFP Question / Requirement" and your "Draft Bid Response" into the respective text areas.
   * Click **Evaluate Response with AI Pipeline (Sequential)** to run the analysis.
