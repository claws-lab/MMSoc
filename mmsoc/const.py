import os.path as osp
import platform
from string import punctuation

if platform.system() == "Linux":
    if osp.exists("/home/ahren"):
        MODEL_PATH = "/home/ahren/Workspace/models_hf"

    elif osp.exists("/nethome/yjin328"):
        MODEL_PATH = "/nethome/yjin328/Workspace/models_hf"

    else:
        MODEL_PATH = ""

else:
    MODEL_PATH = ""


def get_root_dir():
    if platform.system() == "Darwin":

        ROOT = "/Users/ahren/Workspace/data/"


    elif platform.system() == "Linux":
        if osp.exists("/home/ahren"):
            ROOT = "/media/ahren/Dataset/data/"

        elif osp.exists("/workingdir/yjin328/"):
            ROOT = "/workingdir/yjin328/data/"

        elif osp.exists("/nethome/yjin328"):
            ROOT = "/nethome/yjin328/Workspace/data/"

        else:
            ROOT = "."

    else:
        ROOT = "."

    return ROOT


LABELS_DICT = {
    "hatefulmemes": {
        1: "hateful",
        0: "not hateful"
    },
    "politi": {
        1: "fake",
        0: "true"
    }
}

LABELS_DICT['gossip'] = LABELS_DICT["politi"]



TOPICS2QUESTION = {
    "Description": "Describe the scene, such as its major subjects, colors, and texture.",
    "Background": "Describe the background of the scene.",
    "Foreground": "Describe the foreground of the scene.",
    "Social_Context": "Describe the cultural and social context of the image.",
    "Target": "This is a meme on social media. Assume this memes is hateful. Which ethnic or demographic "
                         "groups is this meme targeting at?",
    "Audience": "What particular groups is the image and text targeting at?",
    "Relevance_Short_Response": "How is the given image relevant to the text?",
    "Hateful": "Why do you think this meme is potentially hateful?",
    "Misinformation": "Why do you think this image text pair is potentially "
                                                  "misinformative?",
    "True": "Why do you think this image text pair is potentially "
                                        "true news?",
    "Why is this meme potentially humorous": "Why is this meme potentially humorous?",
    "Why is this meme potentially not humorous": "Why is this meme potentially not humorous?",
    "Why is this meme potentially sarcastic": "Why is this meme potentially sarcastic?",
    "Why is this meme potentially not sarcastic": "Why is this meme potentially not sarcastic?",
    "Why is this meme potentially offensive": "Why is this meme potentially offensive?",
    "Why is this meme potentially not offensive": "Why is this meme potentially not offensive?",
    "Why is this meme potentially motivational": "Why is this meme potentially motivational?",
    "Why is this meme potentially not motivational": "Why is this meme potentially not motivational",
    "Why is the sentiment potentially positive": "Why is the sentiment potentially positive?",
    "Why is the sentiment potentially negative": "Why is the sentiment potentially negative?"
}


TASK2QUESTION = {
    "misinformation": "Is this news (image-text pair) misinformation? Answer \'Yes\' or \'No\'.",
    "hatespeech": "Is this meme hateful? Answer \'Yes\' or \'No\'.",

    "relevance": "Is the given image relevant to the text? Answer \'Very Relevant\' or \'Relevant\' or "
                 "\'Somewhat Relevant\' or \'Not Relevant\'.",
    "humor": "Is this meme humorous? Answer \'humorous\' or \'not_humorous\'.",
    "sarcasm": "Is this meme sarcastic? Answer \'sarcastic\' or \'not_sarcastic\'.",
    "sentiment": "What is the overall sentiment expressed through this meme? Answer \'positive\' or "
                         "\'neutral\' or \'negative\'",
    "offensive": "Is this meme offensive? Answer \'offensive\' or \'not_offensive\'.",
    "OCR": "What is the text in the image? Use double quotes to enclose your answer.",
    "tag": ("Predict the tags of the following online video given its title, description, and thumbnail image. "
              "Different tags must be separated by commas. For example, \"sports,tennis,french-open\"")
}

YOUTUBE_TOPICS = ['Action-adventure_game', 'Action_game', 'American_football', 'Association_football', 'Baseball',
                  'Basketball', 'Boxing', 'Business', 'Casual_game', 'Christian_music', 'Classical_music',
                  'Country_music', 'Cricket', 'Electronic_music', 'Entertainment', 'Fashion', 'Film', 'Food', 'Golf',
                  'Health', 'Hip_hop_music', 'Hobby', 'Humour', 'Ice_hockey', 'Independent_music', 'Jazz', 'Knowledge',
                  'Lifestyle', 'Military', 'Mixed_martial_arts', 'Motorsport', 'Music', 'Music_of_Asia',
                  'Music_of_Latin_America', 'Music_video_game', 'Performing_arts', 'Pet', 'Physical_attractiveness',
                  'Physical_fitness', 'Politics', 'Pop_music', 'Professional_wrestling', 'Puzzle_video_game',
                  'Racing_video_game', 'Reggae', 'Religion', 'Rhythm_and_blues', 'Rock_music',
                  'Role-playing_video_game', 'Simulation_video_game', 'Society', 'Soul_music', 'Sport', 'Sports_game',
                  'Strategy_video_game', 'Technology', 'Television_program', 'Tennis', 'Tourism', 'Vehicle',
                  'Video_game_culture', 'Volleyball']

YOUTUBE_TOPICS_LOWER = ['action-adventure_game', 'action_game', 'american_football', 'association_football', 'baseball',
                        'basketball', 'boxing', 'business', 'casual_game', 'christian_music', 'classical_music',
                        'country_music', 'cricket', 'electronic_music', 'entertainment', 'fashion', 'film', 'food',
                        'golf',
                        'health', 'hip_hop_music', 'hobby', 'humour', 'ice_hockey', 'independent_music', 'jazz',
                        'knowledge',
                        'lifestyle', 'military', 'mixed_martial_arts', 'motorsport', 'music', 'music_of_asia',
                        'music_of_latin_america', 'music_video_game', 'performing_arts', 'pet',
                        'physical_attractiveness',
                        'physical_fitness', 'politics', 'pop_music', 'professional_wrestling', 'puzzle_video_game',
                        'racing_video_game', 'reggae', 'religion', 'rhythm_and_blues', 'rock_music',
                        'role-playing_video_game', 'simulation_video_game', 'society', 'soul_music', 'sport',
                        'sports_game',
                        'strategy_video_game', 'technology', 'television_program', 'tennis', 'tourism', 'vehicle',
                        'video_game_culture', 'volleyball']

labels_d = {
    "true": 0,
    "false": 1,
    "fake": 1,
}
TRUE_WORDS = ["true", "real"]
FAKE_WORDS = ["false", "fake"]

labels_d_humor = {
    "very_funny": 1,
    "hilarious": 1,
    "funny": 1,
    "not_funny": 0,
    "2": 1,
    "2.": 1,
    2.: 1,
}

MEMOTION_LABELS = {
    "humor": {'hilarious': "humorous",
               'not_funny': "not_humorous",
               'very_funny': "humorous",
               'funny': "humorous"},
    "sarcasm": {'not_sarcastic': "not_sarcastic",
                'general': "sarcastic",
                'twisted_meaning': "sarcastic",
                'very_twisted': "sarcastic"},
    "sentiment": {'very_negative': "negative",
                          'negative': "negative",
                          'neutral': "neutral",
                          'positive': "positive",
                          'very_positive': "positive"},

    "offensive": {
        'not_offensive': "not_offensive",
        'slight': "offensive",
        'very_offensive': "offensive",
        'hateful_offensive': "offensive",
    },

    "motivational": {
        'motivational': "motivational",
        'not_motivational': "not_motivational",
    },

}

labels_d_sarcasm = {
    "sarcastic": 1,
    "general": 1,
    "not_sarcastic": 0,
    'twisted_meaning': 1,
    'very_twisted': 1,
    '1': 1,
    1: 1,
    '0': 0,
    0: 0
}

labels_d_sentiment = {
    'very_negative': 0,
    'negative': 0,
    'neutral': 1,
    'positive': 2,
    'very_positive': 2,
}

labels_d_offensive = {
    "not_offensive": 0,
    "slight": 1,
    "very_offensive": 1,
    "hateful_offensive": 1,
}

separators = punctuation + '\n '
