import streamlit as st
import pickle
import pandas as pd
st.title('ipl win predictor')

col1,col2 = st.columns(2)

teams = ['Royal Challengers Bengalore','Kings XI Punjab','Delhi Capitals','Mumbai Indians','Kolkata Knight Riders', 'Rajasthan Royals',
       'Sunrisers Hyderabad', 'Chennai Super Kings','Gujarat Titans',
       ]

pipe = pickle.load(open('pipe.pkl','rb'))
with col1:
   batting_team = st.selectbox('select the batting team',sorted(teams))
with col2:
   bowling_team = st.selectbox('select the bowling team',sorted(teams))

city = ['Bangalore', 'Chandigarh', 'Delhi', 'Kolkata', 'Jaipur',
       'Hyderabad', 'Chennai', 'Mumbai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Kochi', 'Indore', 'Visakhapatnam', 'Pune', 'Raipur', 'Abu Dhabi',
        'Ranchi', 'Rajkot', 'Kanpur', 'Bengaluru', 'Dubai', 'Sharjah',
       'Navi Mumbai', 'Lucknow', 'Guwahati', 'Mohali']

selected_city = st.selectbox('select host city',city)
target = st.number_input('target')


col3,col4,col5 = st.columns(3)

with col3:
   score = st.number_input('score')
with col4:
   overs = st.number_input('over')
with col5:
   wickets = st.number_input('wickets')

if st.button('predict probability'):
   runs_left = target - score
   balls_left = 120-(overs*6)
   wickets = 10-wickets
   curr = score/overs
   rrr = (runs_left*6)/balls_left

   input_df = pd.DataFrame({
      'city':[selected_city],
      'batting_team':[batting_team],
      'bowling_team':[bowling_team],
      'total_runs_x':[target],
      'runs_left':[runs_left],
      'balls_left':[balls_left],
      'wickets_left':[wickets],
      'crr':[curr],
      'rrr':[rrr]
      
   })
   result = pipe.predict_proba(input_df)
   loss = result[0][0]
   win = result[0][1]
   st.header(batting_team + " - "+str(round(win*100)) + "%")
   st.header(bowling_team + " - "+str(round(loss*100))+"%")


