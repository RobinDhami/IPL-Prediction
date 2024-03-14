import pickle
import pandas as pd
import base64
import streamlit as st
from logisticregression import LogisticRegression

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()
img = get_img_as_base64("background.jpg")
# data:image/png;base64,{img}
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/png;base64,{img}");
width: 100%;
height:100%
background-repeat: no-repeat;
background-attachment: fixed;
background-size: cover;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""
teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Punjab Kings',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals',
    'Gujarat Titans',
    'Lucknow Super Giants'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Kolkata', 'Delhi', 'Chennai',
    'Jaipur', 'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion',
    'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein',
    'Ahmedabad', 'Cuttack', 'Nagpur', 'Visakhapatnam', 'Pune',
    'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah', 'Bengaluru'
]
# Load the pipeline from the file
with open('pipe.pkl', 'rb') as file:
    loaded_pipe = pickle.load(file)

# Now you can use loaded_pipe for predictions or further processing
#loaded_pipe.predict(X_test)
st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown("""
    # **IPL VICTORY PREDICTOR**            
""")
# Assuming you get these values from user input or some other source
col1, col2 = st.columns(2)

with col1:
   
   batting_team =  st.selectbox('Select Batting Team',sorted(teams))

with col2:
    if batting_team == '--- select ---':
        bowling_team = st.selectbox('Select Bowling Team', sorted(teams))
    else:
        filtered_teams = [team for team in teams if team != batting_team]
        bowling_team = st.selectbox('Select Bowling Team', sorted(filtered_teams))
selected_city = st.selectbox('Select Host city', sorted(cities))
target = st.number_input('Target')

col1,col2,col3 = st.columns(3)

with col1:
    score = st.number_input('Score')
with col2:
    overs = st.number_input("Over Completed")
with col3:
    wickets = st.number_input("wicktes down")


# Rest of your code remains unchanged from here

if st.button('Predict Productivity'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets = 10 - wickets
    crr = score / overs
    rrr = (runs_left * 6) / balls_left
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    #st.table(input_df)
    result = loaded_pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + ": " + str(round(win*100)) + "%")
    st.header(bowling_team + ": " + str(round(loss*100)) + "%")