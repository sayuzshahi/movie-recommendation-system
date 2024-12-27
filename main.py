import streamlit as st
import streamlit_option_menu
from streamlit_extras.stoggle import stoggle

# Ensure your custom processing module is correctly imported
try:
    from processing import preprocess
    from processing.display import Main
except ImportError as e:
    st.error("Processing module is missing or not installed. Please verify.")
    raise e

# Setting the wide mode as default
st.set_page_config(layout="wide")

# Session State Initialization
if 'movie_number' not in st.session_state:
    st.session_state['movie_number'] = 0
if 'selected_movie_name' not in st.session_state:
    st.session_state['selected_movie_name'] = ""
if 'user_menu' not in st.session_state:
    st.session_state['user_menu'] = ""
if 'new_df' not in st.session_state:
    st.session_state['new_df'] = None  # Ensure new_df is available

def main():
    """Main function to control the Streamlit app."""

    def initial_options():
        st.session_state.user_menu = streamlit_option_menu.option_menu(
            menu_title='What type of movie are you looking for?',
            options=['Recommend me a similar movie', 'Auto-suggest a movie'],
            icons=['film', 'film'],
            menu_icon='list',
            orientation="horizontal",
        )
        if st.session_state.user_menu == 'Recommend me a similar movie':
            recommend_display()
        elif st.session_state.user_menu == 'Auto-suggest a movie':
            auto_suggest_movie()

    def recommend_display():
        st.title('Similar Movie Recommender System')
        try:
            selected_movie_name = st.selectbox('Select a Movie...', st.session_state.new_df['title'].values)
        except KeyError:
            st.error("Movies dataset is not loaded properly.")
            return

        if st.button('Recommend'):
            st.session_state.selected_movie_name = selected_movie_name
            recommendation_tags(st.session_state.new_df, selected_movie_name, "Files/similarity_tags_tags.pkl", "based on similarity")

    def recommendation_tags(new_df, selected_movie_name, pickle_file_path, label):
        try:
            movies, posters = preprocess.recommend(new_df, selected_movie_name, pickle_file_path)
            st.subheader(f'Best Recommendations {label}...')
            
            # Create a row of columns for the recommended movies
            cols = st.columns(5)  # Display 5 movies in a row
            
            # Loop through the recommendations and display each movie with genres, release date, and overview
            for i, (movie, poster) in enumerate(zip(movies[:5], posters[:5])):
                movie_details = preprocess.get_details(movie)  # Fetch the movie details for each recommendation
                genres = " . ".join(movie_details[2])  # Extract genres
                release_date = movie_details[4]  # Extract release date
                overview = movie_details[3]  # Extract overview

                with cols[i]:  # Display the movie in the ith column
                    st.text(movie)
                    st.image(poster if poster else "placeholder.png")
                    st.text(f"Genres: {genres}")
                    st.text(f"Release Date: {release_date}")
                    st.text(f"Overview: {overview}")

        except Exception as e:
            st.error("Error while fetching recommendations.")
            st.exception(e)
            return

    def auto_suggest_movie():
        """Auto-suggest a random movie and display its details (genres, release date, overview)."""
        st.title("Auto-suggested Movie")

        try:
            random_movie_name = st.session_state.new_df.sample()['title'].values[0]  # Get a random movie
            movie_details = preprocess.get_details(random_movie_name)

            genres = " . ".join(movie_details[2])  # Extract genres
            release_date = movie_details[4]  # Extract release date
            overview = movie_details[3]  # Extract overview

            st.text(f"Movie Name: {random_movie_name}")
            
            st.text(f"Genres: {genres}")
            st.text(f"Release Date: {release_date}")
            st.text(f"Overview: {overview}")
        except Exception as e:
            st.error("Error fetching movie details.")
            st.exception(e)

    # Main Loop
    try:
        with Main() as bot:
            bot.main_()
            new_df, movies, movies2 = bot.getter()
            st.session_state.new_df = new_df  # Store new_df in session state to be accessed later
            initial_options()

    except Exception as e:
        st.error("Error initializing Main class.")
        st.exception(e)

if __name__ == '__main__':
    main()





