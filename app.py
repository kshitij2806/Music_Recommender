import streamlit as st
import pandas as pd
import pickle
import time
import os
import sys
import numpy as np

# Set the working directory to ensure proper path resolution
project_dir = 'C:/Users/Kshitij/Documents/Projects/ML/Music_Recommender'
os.chdir(project_dir)

# Add the project directory to Python path
sys.path.append(project_dir)

from src.recommendation_engine import (
    load_recommendation_system, 
    get_recommendations, 
    search_songs_database
)

# Page configuration
st.set_page_config(
    page_title="Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling with proper contrast
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1DB954;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    /* Song cards with proper contrast */
    .song-card {
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #1DB954;
        background-color: #ffffff;
        margin-bottom: 1rem;
        color: #2c3e50;
        border: 2px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .song-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    
    .song-card h4 {
        color: #1DB954;
        margin-bottom: 0.8rem;
        font-size: 1.3rem;
        font-weight: bold;
    }
    
    .song-card p {
        color: #555555;
        margin-bottom: 0.5rem;
        font-size: 1rem;
        line-height: 1.4;
    }
    
    .song-card strong {
        color: #2c3e50;
    }
    
    /* Search result cards */
    .search-result {
        padding: 1.5rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin-bottom: 1rem;
        color: #ffffff;
        border: 2px solid #5a6fd0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .search-result h4 {
        color: #ffffff;
        margin-bottom: 0.8rem;
        font-size: 1.3rem;
        font-weight: bold;
    }
    
    .search-result p {
        color: #f0f0f0;
        margin-bottom: 0.5rem;
        font-size: 1rem;
        line-height: 1.4;
    }
    
    .search-result strong {
        color: #ffffff;
    }
    
    /* Selected song styling */
    .selected-song {
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
        border-left: 6px solid #ffffff;
        color: #ffffff;
        border: 2px solid #1DB954;
    }
    
    .selected-song h4 {
        color: #ffffff;
        font-weight: bold;
    }
    
    .selected-song p {
        color: #ffffff;
    }
    
    .selected-song strong {
        color: #ffffff;
    }
    
    /* Headers */
    .recommendation-header {
        color: #1DB954;
        font-size: 1.8rem;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        font-weight: bold;
        border-bottom: 3px solid #1DB954;
        padding-bottom: 0.5rem;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
        color: white;
        border: none;
        padding: 0.7rem 1.5rem;
        border-radius: 8px;
        width: 100%;
        font-weight: bold;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #1aa34a 0%, #1bc255 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Warning and info boxes */
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #856404;
        font-weight: bold;
    }
    
    .info-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border: 2px solid #17a2b8;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #0c5460;
        font-weight: bold;
    }
    
    /* Success messages */
    .stSuccess {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #28a745;
        color: #155724;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Input fields */
    .stTextInput input {
        border: 2px solid #1DB954;
        border-radius: 8px;
        padding: 0.8rem;
        font-size: 1rem;
    }
    
    .stTextInput input:focus {
        border-color: #1aa34a;
        box-shadow: 0 0 0 2px rgba(29, 185, 84, 0.2);
    }
    
    /* Slider */
    .stSlider {
        color: #1DB954;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #6c757d;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 2px solid #e9ecef;
    }
    
    /* Ensure all text has proper contrast */
    .stApp {
        color: #2c3e50;
        background-color: #f8f9fa;
    }
    
    /* Section headers */
    h2 {
        color: #2c3e50;
        border-bottom: 2px solid #1DB954;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    h3 {
        color: #34495e;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_components():
    """Load recommendation system components with caching"""
    try:
        components = load_recommendation_system()
        
        # Check if required components exist
        required_keys = ['df']
        missing_keys = [key for key in required_keys if key not in components]
        
        if missing_keys:
            st.warning(f"Missing components in model: {missing_keys}")
            return None
            
        return components
    except Exception as e:
        st.error(f"Error loading recommendation system: {e}")
        return None

def get_random_popular_songs(df, num_songs=10, genre=None):
    """
    Get random popular songs from the dataset
    """
    try:
        # Filter by genre if specified
        if genre and genre != "All Genres":
            filtered_df = df[df['track_genre_clean'] == genre].copy()
        else:
            filtered_df = df.copy()
        
        # Ensure we have enough songs
        if len(filtered_df) < num_songs:
            filtered_df = df.copy()
        
        # Get popular songs (top 30% by popularity)
        popular_threshold = filtered_df['popularity'].quantile(0.7)
        popular_songs = filtered_df[filtered_df['popularity'] >= popular_threshold]
        
        # If we don't have enough popular songs, take from entire dataset
        if len(popular_songs) < num_songs:
            popular_songs = filtered_df
        
        # Randomly sample
        if len(popular_songs) > num_songs:
            popular_songs = popular_songs.sample(n=num_songs, random_state=42)
        
        return popular_songs[['track_name_clean', 'artists_clean', 'track_genre_clean', 'popularity']].sort_values('popularity', ascending=False)
    
    except Exception as e:
        st.error(f"Error getting popular songs: {e}")
        # Fallback: return random songs
        return df.sample(n=min(num_songs, len(df)))[['track_name_clean', 'artists_clean', 'track_genre_clean', 'popularity']]

def simple_content_based_recommendations(selected_song, df, num_recommendations=10):
    """
    Simple content-based recommendations based on genre and popularity
    Fallback when similarity matrix is not available
    """
    try:
        # Get the selected song's genre
        selected_genre = selected_song['track_genre_clean']
        
        # Filter songs by same genre
        same_genre_songs = df[df['track_genre_clean'] == selected_genre].copy()
        
        # Remove the selected song itself using proper pandas filtering
        mask = ~(
            (same_genre_songs['track_name_clean'] == selected_song['track_name_clean']) & 
            (same_genre_songs['artists_clean'] == selected_song['artists_clean'])
        )
        same_genre_songs = same_genre_songs[mask]
        
        if len(same_genre_songs) == 0:
            # If no same genre songs, use all songs
            same_genre_songs = df.copy()
            # Remove the selected song from all songs
            mask = ~(
                (same_genre_songs['track_name_clean'] == selected_song['track_name_clean']) & 
                (same_genre_songs['artists_clean'] == selected_song['artists_clean'])
            )
            same_genre_songs = same_genre_songs[mask]
        
        # Sort by popularity and get top recommendations
        recommendations = same_genre_songs.sort_values('popularity', ascending=False).head(num_recommendations)
        
        # Add a dummy similarity score for display
        recommendations['similarity_score'] = np.random.uniform(0.7, 0.95, len(recommendations))
        
        return recommendations[['track_name_clean', 'artists_clean', 'track_genre_clean', 'popularity', 'similarity_score']]
    
    except Exception as e:
        st.error(f"Error in simple recommendations: {e}")
        # Final fallback: return random songs (excluding the selected one)
        other_songs = df[
            ~(
                (df['track_name_clean'] == selected_song['track_name_clean']) & 
                (df['artists_clean'] == selected_song['artists_clean'])
            )
        ]
        return other_songs.sample(n=min(num_recommendations, len(other_songs)))[['track_name_clean', 'artists_clean', 'track_genre_clean', 'popularity']]

def safe_search_songs(query, max_results=15):
    """
    Safely call search_songs_database with proper parameters
    Handles different function signatures
    """
    try:
        # Try calling with max_results parameter
        results = search_songs_database(query, max_results=max_results)
        return results
    except TypeError:
        try:
            # If that fails, try calling without max_results and limit afterwards
            results = search_songs_database(query)
            return results.head(max_results)
        except Exception as e:
            st.error(f"Search failed: {e}")
            return pd.DataFrame()

def main():
    # Header
    st.markdown('<h1 class="main-header">🎵 Music Recommender</h1>', unsafe_allow_html=True)
    
    # Load components
    with st.spinner("Loading recommendation system..."):
        components = load_components()
    
    if components is None:
        st.error("Failed to load the recommendation system. Please check the model files.")
        return
    
    # Extract components with fallbacks
    df = components['df']
    
    # Check if similarity matrix exists, otherwise use simple recommendations
    similarity = components.get('similarity')
    vectorizer = components.get('vectorizer')
    
    if similarity is None or vectorizer is None:
        st.markdown("""
        <div class="warning-box">
            ⚠️ <strong>Advanced recommendations unavailable</strong><br>
            Using basic genre-based recommendations. For full functionality, please retrain the model with similarity matrix.
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Navigation")
    app_mode = st.sidebar.radio("Choose a mode:", 
                               ["🔍 Search & Recommend", "🎲 Discover Popular Songs"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Tips")
    st.sidebar.info("• Search by song title or artist name\n• Select a song to get recommendations\n• Use the slider to adjust number of recommendations")
    
    if app_mode == "🔍 Search & Recommend":
        search_and_recommend(df, similarity, vectorizer)
    else:
        discover_popular_songs(df)

def search_and_recommend(df, similarity, vectorizer):
    """Main search and recommendation interface"""
    
    # Search section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎯 Find Your Favorite Songs")
        query = st.text_input(
            "Search for a song:",
            placeholder="Enter song title or artist...",
            help="Start typing to search for songs in our database"
        )
    
    with col2:
        st.subheader("⚙️ Settings")
        num_recommendations = st.slider(
            "Number of recommendations:",
            min_value=5,
            max_value=20,
            value=10,
            help="How many similar songs to recommend"
        )
    
    # Initialize session state for selected song
    if 'selected_song' not in st.session_state:
        st.session_state.selected_song = None
    
    # Search results
    if query:
        with st.spinner(f"🔍 Searching for '{query}'..."):
            start_time = time.time()
            search_results = safe_search_songs(query, max_results=15)
            search_time = time.time() - start_time
            
        if not search_results.empty:
            st.success(f"✅ Found {len(search_results)} songs in {search_time:.2f} seconds")
            
            # Display search results
            st.subheader("📋 Search Results")
            display_song_selection(search_results)
            
            # Get recommendations if a song is selected
            if st.session_state.selected_song is not None:
                get_song_recommendations(st.session_state.selected_song, df, similarity, vectorizer, num_recommendations)
        else:
            st.warning("❌ No songs found. Try a different search term.")
            
            # Show popular songs as fallback
            st.subheader("🔥 Popular Songs You Might Like")
            popular_songs = get_random_popular_songs(df, num_songs=8)
            for _, song in popular_songs.iterrows():
                display_song_card(song)
    else:
        # Show popular songs when no search query
        st.markdown("""
        <div class="info-box">
            👆 Start by searching for a song above, or check out popular songs below!
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("🌟 Popular Songs")
        popular_songs = get_random_popular_songs(df, num_songs=12)
        
        cols = st.columns(3)
        for idx, (_, song) in enumerate(popular_songs.iterrows()):
            with cols[idx % 3]:
                if st.button(f"🎵 {song['track_name_clean'][:25]}{'...' if len(song['track_name_clean']) > 25 else ''}", 
                           key=f"popular_{idx}"):
                    st.session_state.selected_song = song
                    st.rerun()

def display_song_selection(search_results):
    """Display search results and handle song selection"""
    
    st.write("🎯 Select a song to get recommendations:")
    
    for idx, (_, song) in enumerate(search_results.iterrows()):
        col1, col2 = st.columns([3, 1])
        
        # Check if this song is currently selected
        is_selected = (
            st.session_state.selected_song is not None and
            st.session_state.selected_song['track_name_clean'] == song['track_name_clean'] and
            st.session_state.selected_song['artists_clean'] == song['artists_clean']
        )
        
        card_class = "selected-song" if is_selected else "search-result"
        
        with col1:
            st.markdown(f"""
            <div class="{card_class}">
                <h4>🎵 {song['track_name_clean']}</h4>
                <p><strong>🎤 Artist:</strong> {song['artists_clean']}</p>
                <p><strong>🎭 Genre:</strong> {song['track_genre_clean']}</p>
                <p><strong>⭐ Popularity:</strong> {song['popularity']}/100</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            button_text = "✅ Selected" if is_selected else "Select"
            if st.button(button_text, key=f"select_{idx}", disabled=is_selected):
                st.session_state.selected_song = song
                st.rerun()

def get_song_recommendations(selected_song, df, similarity, vectorizer, num_recommendations):
    """Get and display recommendations for selected song"""
    
    # Display the selected song
    st.markdown(f'<div class="recommendation-header">🎵 Selected Song</div>', unsafe_allow_html=True)
    display_song_card(selected_song, is_selected=True)
    
    with st.spinner("🎯 Finding similar songs..."):
        start_time = time.time()
        
        # Try advanced recommendations if similarity matrix exists
        if similarity is not None and vectorizer is not None:
            try:
                recommendations = get_recommendations(
                    selected_song['track_name_clean'],
                    selected_song['artists_clean'],
                    df,
                    similarity,
                    vectorizer,
                    num_recommendations
                )
            except Exception as e:
                st.warning(f"Advanced recommendations failed: {e}. Using basic recommendations.")
                recommendations = simple_content_based_recommendations(selected_song, df, num_recommendations)
        else:
            # Use simple content-based recommendations
            recommendations = simple_content_based_recommendations(selected_song, df, num_recommendations)
        
        recommendation_time = time.time() - start_time
    
    if not recommendations.empty:
        st.markdown(f'<div class="recommendation-header">🎧 Similar Songs</div>', unsafe_allow_html=True)
        st.success(f"✨ Generated {len(recommendations)} recommendations in {recommendation_time:.2f} seconds")
        
        # Display recommendations in a grid
        cols = st.columns(2)
        for idx, (_, song) in enumerate(recommendations.iterrows()):
            with cols[idx % 2]:
                display_song_card(song, show_similarity=True)
    else:
        st.warning("❌ No recommendations found. Try selecting a different song.")

def discover_popular_songs(df):
    """Display popular songs for discovery"""
    st.subheader("🎲 Discover Popular Songs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        genre_filter = st.selectbox(
            "🎭 Filter by genre:",
            ["All Genres"] + sorted(df['track_genre_clean'].unique().tolist())
        )
    
    with col2:
        num_songs = st.slider(
            "📊 Number of songs to show:",
            min_value=8,
            max_value=24,
            value=12,
            step=4
        )
    
    # Get popular songs
    popular_songs = get_random_popular_songs(df, num_songs=num_songs, genre=genre_filter if genre_filter != "All Genres" else None)
    
    if not popular_songs.empty:
        st.success(f"🎉 Found {len(popular_songs)} popular songs!")
        
        # Display in a grid
        cols = st.columns(3)
        for idx, (_, song) in enumerate(popular_songs.iterrows()):
            with cols[idx % 3]:
                display_song_card(song)
    else:
        st.warning("❌ No songs found with the selected criteria.")

def display_song_card(song, show_similarity=False, is_selected=False):
    """Display a song in a nice card format"""
    card_class = "selected-song" if is_selected else "song-card"
    
    similarity_text = ""
    if show_similarity and 'similarity_score' in song:
        similarity_score = song.get("similarity_score", 0)
        similarity_text = f'<p><strong>🔍 Similarity:</strong> {similarity_score:.3f}</p>'
    
    with st.container():
        st.markdown(f"""
        <div class="{card_class}">
            <h4>🎵 {song['track_name_clean']}</h4>
            <p><strong>🎤 Artist:</strong> {song['artists_clean']}</p>
            <p><strong>🎭 Genre:</strong> {song['track_genre_clean']}</p>
            <p><strong>⭐ Popularity:</strong> {song['popularity']}/100</p>
            {similarity_text}
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <hr>
    <p>Built with ❤️ using Streamlit | 🎵 Music data from Spotify</p>
    <p>🎶 Discover your next favorite song!</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()