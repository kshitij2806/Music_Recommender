import pandas as pd
import pickle
import sqlite3
import os

def load_recommendation_system():
    """Load the pre-trained recommendation system"""
    with open('models/recommendation_system.pkl', 'rb') as f:
        components = pickle.load(f)
    return components

def setup_music_database():
    """Setup SQLite database for music data"""
    components = load_recommendation_system()
    df = components['df']

    conn = sqlite3.connect('music_data.db')
    df.to_sql('songs', conn, if_exists='replace', index=False)

    # Create index for faster searches
    conn.execute("CREATE INDEX IF NOT EXISTS idx_track_name ON songs(track_name_clean)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_popularity ON songs(popularity)")

    conn.close()
    return "Database setup complete"

def search_songs_database(query, limit=10):
    """Search songs using SQLite database - auto-initializes if needed"""
    # Check if database exists, if not create it
    if not os.path.exists('music_data.db'):
        print("Initializing music database for the first time...")
        setup_music_database()

    conn = sqlite3.connect('music_data.db')

    # Use parameterized query to prevent SQL injection
    result = pd.read_sql("""
        SELECT * FROM songs 
        WHERE track_name_clean LIKE ? 
        ORDER BY popularity DESC 
        LIMIT ?
    """, conn, params=[f'%{query}%', limit])

    conn.close()
    return result

def get_mood_based_recommendations(target_mood, num_recommendations=10):
    """Get songs based on mood (happy, sad, energetic, calm)"""
    components = load_recommendation_system()
    df = components['df']

    mood_profiles = {
        'happy': {'valence': 0.7, 'energy': 0.6, 'danceability': 0.6},
        'sad': {'valence': 0.3, 'energy': 0.3, 'danceability': 0.3},
        'energetic': {'energy': 0.8, 'danceability': 0.7, 'valence': 0.6},
        'calm': {'energy': 0.2, 'acousticness': 0.7, 'valence': 0.5}
    }

    if target_mood in mood_profiles:
        target_profile = mood_profiles[target_mood]
        scores = []

        for idx, row in df.iterrows():
            score = 0
            for feature, target_value in target_profile.items():
                score += 1 - abs(row[feature] - target_value)
            scores.append(score)

        df_temp = df.copy()
        df_temp['mood_score'] = scores
        return df_temp.nlargest(num_recommendations, 'mood_score')[['track_name_clean', 'artists_clean', 'mood_score']]

    return pd.DataFrame()

def create_playlist_from_seeds(seed_songs, playlist_length=20):
    """Create a playlist from multiple seed songs"""
    all_recommendations = []

    for song in seed_songs:
        recs = get_recommendations(song, playlist_length//len(seed_songs))
        if isinstance(recs, pd.DataFrame):
            all_recommendations.append(recs)

    if all_recommendations:
        combined = pd.concat(all_recommendations)
        # Remove duplicates and sort by similarity
        combined = combined.drop_duplicates('track_name').nlargest(playlist_length, 'similarity_score')
        return combined

    return pd.DataFrame()

def explain_recommendation(original_song, recommended_song):
    """Explain why a song was recommended"""
    components = load_recommendation_system()
    df = components['df']

    orig_data = df[df['track_name_clean'] == original_song].iloc[0]
    rec_data = df[df['track_name_clean'] == recommended_song].iloc[0]

    similarities = []
    for feature in ['danceability', 'energy', 'valence', 'tempo']:
        similarity = 1 - abs(orig_data[feature] - rec_data[feature])
        if similarity > 0.7:
            similarities.append(f"{feature} ({similarity:.1%} similar)")

    return f"Recommended because of similar: {', '.join(similarities)}" if similarities else "Similar in overall audio characteristics"

def get_recommendations(track_name, num_recommendations=10):
    """
    Main recommendation function for Streamlit app
    """
    # Load components
    components = load_recommendation_system()
    nn_model = components['nn_model']
    feature_matrix_minimal = components['feature_matrix_minimal']
    df = components['df']

    try:
        # Check if song exists
        if track_name not in df['track_name_clean'].values:
            similar_names = df[df['track_name_clean'].str.contains(track_name, case=False, na=False)]
            if len(similar_names) > 0:
                suggestion = similar_names.iloc[0]['track_name_clean']
                return f"Song not found. Did you mean: '{suggestion}'?"
            else:
                return "Song not found in dataset."

        # Get song index and features
        song_idx = df[df['track_name_clean'] == track_name].index[0]
        song_features = feature_matrix_minimal.iloc[song_idx:song_idx+1]

        # Get recommendations
        distances, indices = nn_model.kneighbors(song_features, n_neighbors=num_recommendations + 1)

        recommendations = []
        for i in range(1, len(indices[0])):
            neighbor_idx = indices[0][i]
            distance = distances[0][i]

            # Normalize similarity score
            max_dist = distances[0][1:].max() if len(distances[0]) > 1 else 1
            similarity = max(0.1, 1 - (distance / max_dist)) if max_dist > 0 else 0.5

            neighbor_data = df.iloc[neighbor_idx]
            recommendations.append({
                'track_name': neighbor_data['track_name_clean'],
                'artists': neighbor_data['artists_clean'],
                'genre': neighbor_data['track_genre_clean'],
                'popularity': neighbor_data['popularity'],
                'similarity_score': round(similarity, 3)
            })

        return pd.DataFrame(recommendations)

    except Exception as e:
        return f"Error getting recommendations: {str(e)}"
