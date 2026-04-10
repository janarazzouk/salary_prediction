1.Project Overview:

This project takes user input (job title, experience level, company size, company location,employment type and work arrangment), predicts a salary using a trained ML model, and enhances the result with AI-generated insights and visualizations.

It combines:
Machine Learning
API development
Interactive dashboard
Database storage
LLM-based analysis

2.Project Structure:
salary-prediction-project/
├── api/
│   ├── app.py
│   └── preprocess.py
│ 
├── dashboard/
│   └── app.py 
│
├── data/
│   ├── raw/
│   │   └── ds_salaries.csv
│   └── processed/
│
├── models/
│   ├── salary_model.pkl
│   ├── model_columns.pkl
│   ├── experience_map.pkl
│   ├── size_map.pkl
│   ├── top_locations.pkl
│      
│
├── notebooks/
│   └── data_preprocessing.ipynb
│
├── .env
├── requirements.txt
└── README.md
│
└── requirements.txt

3.Data Processing
Cleaned and explored the dataset inside the notebook
selected relevant features.
Grouped job titles into meaningful categories using custom logic
Encoded categorical variables (experience, company size, etc.)
Prepared final dataset for training

Tool: Pandas, Jupyter Notebook

4.Model Training
Trained a regression model to predict salaries [used the random forest alogorithm]
Applied log transformation on the target variable
Evaluated performance using metrics [the average r squared is 0.48, couldn't get a better measure due to how small the dataset was ]
Used cross-validation to validate the model [k=5]
Saved model and artifacts for later use [ in the models folder]

Tool: Scikit-learn

5.Model Artifacts
salary_model.pkl → trained model
model_columns.pkl → expected input structure
Other .pkl files → encoding mappings (experience, company size, locations)

These ensure the API uses the exact same structure as training

6.API Development
Built a REST API with a /predict endpoint
Receives user input and converts it into model-ready format
Uses a preprocessing pipeline to match training structure
Returns predicted salary (converted back from log scale)

Tool: FastAPI
flow:

Receive input
Preprocess input
Predict using model
Return result

7.Preprocessing Pipeline
Implemented in api/preprocess.py
Handles:
Encoding ordinal values (experience, company size)
Grouping job titles
Handling locations (top vs others)
One-hot encoding
Aligning columns with model expectations

This ensures consistency between training and prediction

8.Dashboard
Built an interactive UI for user input
Sends data to the API and displays results
Shows:
Predicted salary
Input summary
Visual charts
History

Tool: Streamlit

9.Data Visualization
Displayed insights using charts:
verage Salary by Country for the job title selected (it makes the user observe how his salary would differ based on the coutry he's working in)
Top Job Titles by Average Salary(it makes the user see where does he stand between the top most earning jobs)
Used dataset separately from prediction to avoid affecting model results

 Tool: Matplotlib + Pandas


10.AI Insights (Gemini)
NB: i installed ollama and i connected it and used it , but it was taking forever to run so it slowed the whole process so i switched to gemini api 
Integrated an LLM to generate human-readable analysis
Explains:
What the salary means
Impact of experience and role
Effect of location and remote work

Tool: Google Gemini API

11.Database Integration
Stored predictions and inputs in a database
Enabled history tracking inside the dashboard

 Tool: Supabase



12.End-to-End Flow
User enters data in dashboard
Dashboard sends request to API
API preprocesses input and predicts salary
Result is returned to dashboard
Gemini generates insights
Data is stored in Supabase
Charts visualize trends



Tech Stack:
Python
Pandas / NumPy
Scikit-learn
FastAPI
Streamlit
Supabase
Google Gemini API
Matplotlib



How to Run
1. Start the API
uvicorn app:app --reload
2. Run the dashboard
streamlit run app.py
3. Set environment variables
Create a .env file with:

FASTAPI_URL=http://127.0.0.1:8000/predict
SUPABASE_URL=
SUPABASE_KEY=
GEMINI_API_KEY=