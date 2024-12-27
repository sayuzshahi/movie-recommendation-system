import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from processing import preprocess


class Main:
    def __enter__(self):
        # Initialization code, if needed
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Cleanup code, if needed
        pass

    def __init__(self):
        self.new_df = None
        self.movies = None
        self.movies2 = None

    def getter(self):
        """Returns the dataframes."""
        return self.new_df, self.movies, self.movies2

    def get_df(self):
        """Loads or creates the required dataframes."""
        pickle_paths = {
            "new_df": r"Files/new_df_dict.pkl",
            "movies": r"Files/movies_dict.pkl",
            "movies2": r"Files/movies2_dict.pkl",
        }

        if all(os.path.exists(path) for path in pickle_paths.values()):
            # Load preprocessed dataframes from pickle files
            self.new_df = pd.DataFrame.from_dict(self._load_pickle(pickle_paths["new_df"]))
            self.movies = pd.DataFrame.from_dict(self._load_pickle(pickle_paths["movies"]))
            self.movies2 = pd.DataFrame.from_dict(self._load_pickle(pickle_paths["movies2"]))
        else:
            # Preprocess the data and save it to pickle files
            self.movies, self.new_df, self.movies2 = preprocess.read_csv_to_df()

            self._save_pickle(self.new_df.to_dict(), pickle_paths["new_df"])
            self._save_pickle(self.movies.to_dict(), pickle_paths["movies"])
            self._save_pickle(self.movies2.to_dict(), pickle_paths["movies2"])

    def _load_pickle(self, file_path):
        """Helper method to load pickle files."""
        try:
            with open(file_path, "rb") as pickle_file:
                return pickle.load(pickle_file)
        except Exception as e:
            raise RuntimeError(f"Error loading pickle file {file_path}: {e}")

    def _save_pickle(self, data, file_path):
        """Helper method to save data to pickle files."""
        try:
            with open(file_path, "wb") as pickle_file:
                pickle.dump(data, pickle_file)
        except Exception as e:
            raise RuntimeError(f"Error saving to pickle file {file_path}: {e}")

    def vectorise(self, col_name):
        """Vectorizes the specified column using CountVectorizer."""
        try:
            cv = CountVectorizer(max_features=5000, stop_words="english")
            vec_tags = cv.fit_transform(self.new_df[col_name]).toarray()
            return cosine_similarity(vec_tags)
        except KeyError:
            raise ValueError(f"Column '{col_name}' not found in dataframe.")
        except Exception as e:
            raise RuntimeError(f"Error during vectorization: {e}")

    def get_similarity(self, col_name):
        """Computes or loads the similarity matrix for a column."""
        pickle_file_path = fr"Files/similarity_tags_{col_name}.pkl"
        if os.path.exists(pickle_file_path):
            return  # Already computed and saved
        else:
            try:
                similarity_tags = self.vectorise(col_name)
                self._save_pickle(similarity_tags, pickle_file_path)
            except Exception as e:
                raise RuntimeError(f"Error processing similarity for {col_name}: {e}")

    def main_(self):
        """Main method to prepare all required resources."""
        try:
            self.get_df()
            self.get_similarity("tags")
            self.get_similarity("genres")
            self.get_similarity("keywords")
            self.get_similarity("tcast")
            self.get_similarity("tprduction_comp")
        except Exception as e:
            raise RuntimeError(f"Error in Main class initialization: {e}")
