import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(page_title="Cricket Score Predictions", layout="wide")

# App title and description
st.title("Cricket Score Predictions")
st.write("A simple app that provides AI-powered cricket match predictions.")

# Sample teams data
teams = [
    "India", "Australia", "England", "New Zealand", "Pakistan", 
    "South Africa", "West Indies", "Sri Lanka", "Bangladesh", "Afghanistan",
    "Zimbabwe", "Ireland"
]

# Team strength ratings (1-100)
team_ratings = {
    team: random.randint(70, 95) for team in teams
}

# Match formats and team preferences
formats = ["T20", "ODI", "Test"]
format_preferences = {}

# Generate random format preferences for each team
for team in teams:
    preferences = {}
    for format_type in formats:
        # Base preference is team rating with some random adjustment
        preferences[format_type] = max(60, min(100, team_ratings[team] + random.randint(-10, 10)))
    format_preferences[team] = preferences

# Function to generate match prediction
def predict_match(team1, team2, format_type):
    """
    Generate a match prediction based on team ratings and format preferences.
    
    Args:
        team1: Name of the first team
        team2: Name of the second team
        format_type: Type of cricket match
        
    Returns:
        Dictionary with prediction details
    """
    # Get team ratings adjusted for format
    t1_rating = format_preferences.get(team1, {}).get(format_type, team_ratings.get(team1, 80))
    t2_rating = format_preferences.get(team2, {}).get(format_type, team_ratings.get(team2, 80))
    
    # Calculate win probabilities
    total_rating = t1_rating + t2_rating
    t1_win_prob = t1_rating / total_rating
    t2_win_prob = t2_rating / total_rating
    
    # Generate score predictions based on format
    if format_type == "T20":
        t1_score = random.randint(120, 220)
        t1_wickets = random.randint(3, 10)
        t2_score = random.randint(120, 220)
        t2_wickets = random.randint(3, 10)
        overs = 20
    elif format_type == "ODI":
        t1_score = random.randint(200, 350)
        t1_wickets = random.randint(5, 10)
        t2_score = random.randint(200, 350)
        t2_wickets = random.randint(5, 10)
        overs = 50
    else:  # Test
        # For Test matches, we'll simulate innings scores
        t1_innings1 = random.randint(200, 500)
        t1_innings2 = random.randint(150, 350)
        t2_innings1 = random.randint(200, 500)
        t2_innings2 = random.randint(150, 350)
        t1_score = t1_innings1 + t1_innings2
        t2_score = t2_innings1 + t2_innings2
        t1_wickets = 20  # All out in both innings
        t2_wickets = 20
        overs = None
    
    # Adjust scores based on team ratings
    rating_factor = t1_rating / t2_rating
    if format_type != "Test":
        t1_score = int(t1_score * (rating_factor * 0.2 + 0.9))
        t2_score = int(t2_score * (1 / rating_factor * 0.2 + 0.9))
    else:
        t1_innings1 = int(t1_innings1 * (rating_factor * 0.2 + 0.9))
        t2_innings1 = int(t2_innings1 * (1 / rating_factor * 0.2 + 0.9))
        t1_innings2 = int(t1_innings2 * (rating_factor * 0.2 + 0.9))
        t2_innings2 = int(t2_innings2 * (1 / rating_factor * 0.2 + 0.9))
        t1_score = t1_innings1 + t1_innings2
        t2_score = t2_innings1 + t2_innings2
    
    # Format the score
    if format_type != "Test":
        t1_score_str = f"{t1_score}/{t1_wickets}"
        t2_score_str = f"{t2_score}/{t2_wickets}"
        score_str = f"{team1}: {t1_score_str} vs {team2}: {t2_score_str}"
    else:
        score_str = f"{team1}: {t1_innings1} & {t1_innings2} vs {team2}: {t2_innings1} & {t2_innings2}"
    
    # Determine winner
    if format_type != "Test":
        winner = team1 if t1_score > t2_score else team2 if t2_score > t1_score else "Draw"
    else:
        team1_total = t1_innings1 + t1_innings2
        team2_total = t2_innings1 + t2_innings2
        winner = team1 if team1_total > team2_total else team2 if team2_total > team1_total else "Draw"
    
    # Calculate result probabilities
    result_probs = {
        f"{team1} Win": round(t1_win_prob * 100, 1),
        f"{team2} Win": round(t2_win_prob * 100, 1),
        "Draw": round(100 - (t1_win_prob * 100) - (t2_win_prob * 100), 1) if format_type == "Test" else 0
    }
    
    # If not Test match, redistribute Draw probability
    if format_type != "Test":
        t1_win_prob = result_probs[f"{team1} Win"] / 100
        t2_win_prob = result_probs[f"{team2} Win"] / 100
        total = t1_win_prob + t2_win_prob
        result_probs[f"{team1} Win"] = round((t1_win_prob / total) * 100, 1)
        result_probs[f"{team2} Win"] = round((t2_win_prob / total) * 100, 1)
        result_probs.pop("Draw")
    
    return {
        "team1": team1,
        "team2": team2,
        "format": format_type,
        "predicted_score": score_str,
        "winner": winner,
        "result_probabilities": result_probs,
        "team1_rating": t1_rating,
        "team2_rating": t2_rating,
        "analysis": generate_analysis(team1, team2, t1_rating, t2_rating, format_type, result_probs, winner)
    }

# Function to generate match analysis
def generate_analysis(team1, team2, t1_rating, t2_rating, format_type, result_probs, winner):
    """
    Generate a text analysis of the match prediction.
    
    Args:
        team1: Name of the first team
        team2: Name of the second team
        t1_rating: Rating of the first team
        t2_rating: Rating of the second team
        format_type: Type of cricket match
        result_probs: Dictionary of result probabilities
        winner: Predicted winner
        
    Returns:
        String with match analysis
    """
    # Determine team form descriptions
    t1_form = "excellent" if t1_rating > 90 else "good" if t1_rating > 80 else "average" if t1_rating > 70 else "poor"
    t2_form = "excellent" if t2_rating > 90 else "good" if t2_rating > 80 else "average" if t2_rating > 70 else "poor"
    
    # Determine format specialist
    t1_format_rating = format_preferences.get(team1, {}).get(format_type, t1_rating)
    t2_format_rating = format_preferences.get(team2, {}).get(format_type, t2_rating)
    
    format_specialist = None
    if t1_format_rating > t1_rating + 5 and t1_format_rating > t2_format_rating:
        format_specialist = team1
    elif t2_format_rating > t2_rating + 5 and t2_format_rating > t1_format_rating:
        format_specialist = team2
    
    # Determine match difficulty
    rating_diff = abs(t1_rating - t2_rating)
    if rating_diff < 5:
        match_desc = "This looks to be a very close match between evenly matched teams."
    elif rating_diff < 10:
        stronger_team = team1 if t1_rating > t2_rating else team2
        match_desc = f"{stronger_team} has a slight edge in this contest."
    else:
        stronger_team = team1 if t1_rating > t2_rating else team2
        weaker_team = team2 if t1_rating > t2_rating else team1
        match_desc = f"{stronger_team} is strongly favored against {weaker_team}."
    
    # Generate analysis text
    analysis = f"{team1} is in {t1_form} form, while {team2} is showing {t2_form} form. "
    
    if format_specialist:
        analysis += f"{format_specialist} is particularly strong in {format_type} cricket. "
    
    analysis += match_desc + " "
    
    # Add probability insight
    most_likely = max(result_probs, key=result_probs.get)
    analysis += f"Our AI model suggests a {most_likely} is the most likely outcome at {result_probs[most_likely]}%."
    
    return analysis

# Sidebar for team selection
st.sidebar.header("Select Match Details")

team1 = st.sidebar.selectbox("Team 1", teams, index=0)
team2 = st.sidebar.selectbox("Team 2", teams, index=1)
format_type = st.sidebar.selectbox("Format", formats, index=0)

# Prevent same team selection
if team1 == team2:
    st.error("Please select different teams.")
else:
    # Generate prediction when button is clicked
    if st.sidebar.button("Generate Prediction"):
        with st.spinner("AI analyzing match data..."):
            # Simulate AI processing time
            import time
            time.sleep(1)
            
            # Get prediction
            prediction = predict_match(team1, team2, format_type)
            
            # Display match header
            st.header(f"{team1} vs {team2} ({format_type})")
            
            # Display prediction in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Match Prediction")
                st.markdown(f"**Predicted Score:** {prediction['predicted_score']}")
                st.markdown(f"**Predicted Winner:** {prediction['winner']}")
            
            with col2:
                st.subheader("Win Probabilities")
                for outcome, prob in prediction['result_probabilities'].items():
                    st.markdown(f"**{outcome}:** {prob}%")
                
                # Create a simple chart for probabilities
                probs_df = pd.DataFrame({
                    'Outcome': list(prediction['result_probabilities'].keys()),
                    'Probability': list(prediction['result_probabilities'].values())
                })
                st.bar_chart(probs_df.set_index('Outcome'))
            
            with col3:
                st.subheader("Team Ratings")
                st.markdown(f"**{team1}:** {prediction['team1_rating']} (in {format_type})")
                st.markdown(f"**{team2}:** {prediction['team2_rating']} (in {format_type})")
                
                # Rating comparison
                ratings_df = pd.DataFrame({
                    'Team': [team1, team2],
                    'Rating': [prediction['team1_rating'], prediction['team2_rating']]
                })
                st.bar_chart(ratings_df.set_index('Team'))
            
            # Display analysis
            st.subheader("AI Analysis")
            st.info(prediction['analysis'])
            
            # Disclaimer
            st.caption("Disclaimer: These predictions are for entertainment purposes only and are generated using a simplified model.")

# Add upcoming matches section
st.sidebar.divider()
st.sidebar.header("Upcoming Matches")

# Generate some random matches
def generate_matches(num_matches=5):
    """
    Generate random upcoming matches.
    
    Args:
        num_matches: Number of matches to generate
        
    Returns:
        List of match dictionaries
    """
    matches = []
    used_teams = set()
    
    # Start date for matches (tomorrow)
    today = datetime.now()
    next_day = today + timedelta(days=1)
    
    for i in range(num_matches):
        # Find teams not yet used
        available_teams = [team for team in teams if team not in used_teams]
        
        # If we don't have enough teams, reset the used teams
        if len(available_teams) < 2:
            used_teams = set()
            available_teams = teams
        
        # Select random teams
        team1 = random.choice(available_teams)
        used_teams.add(team1)
        available_teams = [team for team in available_teams if team != team1]
        team2 = random.choice(available_teams)
        used_teams.add(team2)
        
        # Generate match date
        match_date = next_day + timedelta(days=i*2)
        
        # Random format
        match_format = random.choice(formats)
        
        matches.append({
            "team1": team1,
            "team2": team2,
            "format": match_format,
            "date": match_date.strftime("%d %b %Y"),
            "time": f"{random.choice([10, 12, 14, 16, 18])}:00"
        })
    
    return matches

# Display matches
matches = generate_matches()
for match in matches:
    st.sidebar.markdown(f"**{match['date']} at {match['time']}**")
    st.sidebar.write(f"{match['team1']} vs {match['team2']} ({match['format']})")
    if st.sidebar.button(f"Predict {match['team1']} vs {match['team2']}", key=f"{match['team1']}_{match['team2']}"):
        # Set the selectboxes to these teams and format
        st.session_state.team1 = match['team1']
        st.session_state.team2 = match['team2']
        st.session_state.format_type = match['format']
        st.rerun()

# Footer
st.divider()
st.caption("Cricket Score Predictions - Powered by AI Insights")