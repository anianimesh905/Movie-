{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "78182705-c926-408b-a3b2-b18f91f48df7",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package punkt to\n",
      "[nltk_data]     C:\\Users\\anian\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package punkt is already up-to-date!\n",
      "[nltk_data] Downloading package stopwords to\n",
      "[nltk_data]     C:\\Users\\anian\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Unzipping corpora\\stopwords.zip.\n",
      "No model was supplied, defaulted to distilbert/distilbert-base-uncased-finetuned-sst-2-english and revision 714eb0f (https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english).\n",
      "Using a pipeline without specifying a model name and revision in production is not recommended.\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "4624cc8ac24442c48971385a852ef706",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "config.json:   0%|          | 0.00/629 [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\anian\\AppData\\Local\\Programs\\Python\\Python313\\Lib\\site-packages\\huggingface_hub\\file_download.py:140: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\\Users\\anian\\.cache\\huggingface\\hub\\models--distilbert--distilbert-base-uncased-finetuned-sst-2-english. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.\n",
      "To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development\n",
      "  warnings.warn(message)\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "20e16783fbb042fba37794e9cb77c649",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "model.safetensors:   0%|          | 0.00/268M [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "5c69312563b740d2bdee8d0b2516d5af",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "tokenizer_config.json:   0%|          | 0.00/48.0 [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "ee8eb5f406224be5805894e15db503ff",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "vocab.txt:   0%|          | 0.00/232k [00:00<?, ?B/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Device set to use cpu\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mood: POSITIVE\n",
      "Keywords: ['want', 'watch', 'happy', 'comedy', 'movie', 'action']\n"
     ]
    }
   ],
   "source": [
    "from transformers import pipeline\n",
    "import nltk\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.tokenize import word_tokenize\n",
    "\n",
    "nltk.download(\"punkt\")\n",
    "nltk.download(\"stopwords\")\n",
    "\n",
    "# Sentiment Analysis using a pre-trained transformer model\n",
    "sentiment_pipeline = pipeline(\"sentiment-analysis\")\n",
    "\n",
    "def extract_mood(prompt):\n",
    "    \"\"\"Extracts mood using sentiment analysis\"\"\"\n",
    "    result = sentiment_pipeline(prompt)[0]\n",
    "    return result[\"label\"]\n",
    "\n",
    "def extract_keywords(prompt):\n",
    "    \"\"\"Extracts keywords from the user input to determine genre and meaning.\"\"\"\n",
    "    words = word_tokenize(prompt.lower())\n",
    "    filtered_words = [word for word in words if word.isalnum() and word not in stopwords.words(\"english\")]\n",
    "    return filtered_words\n",
    "\n",
    "# Example usage\n",
    "if __name__ == \"__main__\":\n",
    "    prompt = \"I want to watch a happy comedy movie with some action!\"\n",
    "    mood = extract_mood(prompt)\n",
    "    keywords = extract_keywords(prompt)\n",
    "    print(f\"Mood: {mood}\")\n",
    "    print(f\"Keywords: {keywords}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "52e6fdd0-6801-4b07-8f2a-0bb035e4cd2d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Top movies in Action:\n"
     ]
    }
   ],
   "source": [
    "import requests\n",
    "\n",
    "API_KEY = \"your_tmdb_api_key\"  # Replace with your TMDb API Key\n",
    "\n",
    "def get_movies_by_genre(genre):\n",
    "    \"\"\"Fetches movies based on genre from TMDb API\"\"\"\n",
    "    url = f\"https://api.themoviedb.org/3/discover/movie?api_key={API_KEY}&with_genres={genre}\"\n",
    "    response = requests.get(url)\n",
    "    if response.status_code == 200:\n",
    "        movies = response.json().get(\"results\", [])\n",
    "        return movies[:10]  # Return top 10 movies\n",
    "    return []\n",
    "\n",
    "def search_movies(query):\n",
    "    \"\"\"Searches for movies based on user query\"\"\"\n",
    "    url = f\"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={query}\"\n",
    "    response = requests.get(url)\n",
    "    if response.status_code == 200:\n",
    "        movies = response.json().get(\"results\", [])\n",
    "        return movies[:5]  # Return top 5 matches\n",
    "    return []\n",
    "\n",
    "# Example usage\n",
    "if __name__ == \"__main__\":\n",
    "    genre = \"Action\"\n",
    "    movies = get_movies_by_genre(genre)\n",
    "    print(f\"Top movies in {genre}:\")\n",
    "    for movie in movies:\n",
    "        print(movie[\"title\"])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "121e5dec-070a-457f-ad04-ea7e5b54edf5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predicted Genre: Animation\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.pipeline import Pipeline\n",
    "import pickle\n",
    "\n",
    "# Sample dataset (expand this with real data)\n",
    "data = {\n",
    "    \"prompt\": [\"I want a scary horror movie\", \"A fun animated movie\", \"A thrilling action-packed film\"],\n",
    "    \"genre\": [\"Horror\", \"Animation\", \"Action\"]\n",
    "}\n",
    "\n",
    "df = pd.DataFrame(data)\n",
    "\n",
    "# Training a simple model\n",
    "X = df[\"prompt\"]\n",
    "y = df[\"genre\"]\n",
    "\n",
    "pipeline = Pipeline([\n",
    "    (\"tfidf\", TfidfVectorizer()),\n",
    "    (\"clf\", LogisticRegression())\n",
    "])\n",
    "\n",
    "pipeline.fit(X, y)\n",
    "\n",
    "# Save model\n",
    "with open(\"genre_model.pkl\", \"wb\") as f:\n",
    "    pickle.dump(pipeline, f)\n",
    "\n",
    "# Function to predict genre\n",
    "def predict_genre(prompt):\n",
    "    with open(\"genre_model.pkl\", \"rb\") as f:\n",
    "        model = pickle.load(f)\n",
    "    return model.predict([prompt])[0]\n",
    "\n",
    "# Example usage\n",
    "if __name__ == \"__main__\":\n",
    "    user_prompt = \"I want a fun animated movie\"\n",
    "    predicted_genre = predict_genre(user_prompt)\n",
    "    print(f\"Predicted Genre: {predicted_genre}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "3a6695ee-7d2c-44c9-aa8c-06698f632506",
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'streamlit'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[16], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;21;01mstreamlit\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;28;01mas\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;21;01mst\u001b[39;00m\n\u001b[0;32m      2\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;21;01mmood_extractor\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;28;01mimport\u001b[39;00m extract_mood, extract_keywords\n\u001b[0;32m      3\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;21;01mfetch_movies\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;28;01mimport\u001b[39;00m get_movies_by_genre\n",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'streamlit'"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "from mood_extractor import extract_mood, extract_keywords\n",
    "from fetch_movies import get_movies_by_genre\n",
    "from genre_classifier import predict_genre\n",
    "\n",
    "st.title(\"🎬 Movie Recommendation System\")\n",
    "st.write(\"Enter a description of the movie you feel like watching, and we'll recommend the best match!\")\n",
    "\n",
    "# User input\n",
    "user_input = st.text_area(\"Enter your movie preference (e.g., 'I want a happy comedy movie with some action!')\")\n",
    "\n",
    "if st.button(\"Get Recommendations\"):\n",
    "    if user_input:\n",
    "        # Extract mood, keywords, and genre\n",
    "        mood = extract_mood(user_input)\n",
    "        keywords = extract_keywords(user_input)\n",
    "        genre = predict_genre(user_input)\n",
    "\n",
    "        # Fetch movie recommendations\n",
    "        movies = get_movies_by_genre(genre)\n",
    "\n",
    "        # Display results\n",
    "        st.subheader(f\"Mood: {mood}\")\n",
    "        st.subheader(f\"Predicted Genre: {genre}\")\n",
    "        \n",
    "        if movies:\n",
    "            st.subheader(\"🎥 Recommended Movies:\")\n",
    "            for movie in movies:\n",
    "                st.write(f\"**{movie['title']}** (⭐ {movie['vote_average']})\")\n",
    "        else:\n",
    "            st.write(\"No recommendations found. Try a different prompt!\")\n",
    "\n",
    "    else:\n",
    "        st.warning(\"Please enter a movie preference.\")\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
