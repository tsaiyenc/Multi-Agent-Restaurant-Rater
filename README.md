# Restaurant Review Analysis System

This is a system that uses PyAutoGen and OpenAI API to analyze restaurant reviews. The system can analyze reviews for specific restaurants and provide overall ratings based on user queries.

## System Requirements

- Python 3.8 or higher
- OpenAI API Key

## Installation Steps

1. Clone the repository:
```bash
git clone [your-repository-url]
cd [project-directory]
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up OpenAI API Key:
   - Set `OPENAI_API_KEY` in environment variables
   - Or set it before running:
     ```bash
     # Linux/Mac
     export OPENAI_API_KEY='your-api-key'
     
     # Windows
     set OPENAI_API_KEY=your-api-key
     ```

## Usage

1. Basic usage:
```bash
python test.py
```

2. Specify data file:
```bash
python test.py [data-file-path]
```

3. Direct query using main.py:
```bash
python main.py [data-file-path] "your query question"
```

### Query Examples
- "How good is the restaurant McDonald's overall?"
- "What is the overall score for Starbucks?"
- "How good is the restaurant Chick-fil-A overall?"

## File Description

- `main.py`: Main program code
- `test.py`: Test program
- `restaurant-data.txt`: Restaurant review dataset
- `requirements.txt`: Project dependencies list

## Notes

1. Make sure you have a valid OpenAI API key
2. Internet connection is required to use OpenAI API
3. It is recommended to use a Python virtual environment for package installation