import base64
import io
from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import requests
from textblob import TextBlob
from googleapiclient.discovery import build
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session
from flask_pymongo import PyMongo
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import matplotlib.pyplot as plt

# Loading the Spacy model with pre-trained word vectors
nlp = spacy.load('en_core_web_md')

def glove_encode(word):
    """
    Encode a word using pre-trained Spacy word vectors.

    Parameters:
    word (str): The word to encode.

    Returns:
    numpy array: The word vector.
    """
    word = word.lower()  # Convert word to lowercase
    doc = nlp(word)
    if doc.has_vector:
        return doc.vector #numpy array
    else:
        print(f"'{word}' does not have a vector.")
        return None
  
    
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# MongoDB Configuration
app.config['MONGO_URI'] = 'mongodb://localhost:27017/sponsor_spotter_db'
mongo = PyMongo(app)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Sign-up route for sponsors
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        brand_name = request.form['brand_name']
        product_type = request.form['product_type']

        sponsors = mongo.db.sponsors
        sponsors.insert_one({'username': username, 'password': password, 'brand_name': brand_name, 'product_type': product_type})

        return redirect(url_for('login'))

    return render_template('signup.html')

# Sign-up route for youtubers
@app.route('/ysignup', methods=['GET', 'POST'])
def ysignup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        channel_name = request.form['channel_name']
        content_type = request.form['content_type']

        youtubers = mongo.db.youtubers
        youtubers.insert_one({'username': username, 'password': password, 'channel_name': channel_name, 'content_type': content_type})

        return redirect(url_for('ylogin'))

    return render_template('ysignup.html')

# Login route for sponsors
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        sponsors = mongo.db.sponsors
        user = sponsors.find_one({'username': username, 'password': password})

        if user:
            session['logged_in'] = True
            session['username'] = user['username']
            return redirect(url_for('profile'))
        else:
            return render_template('login.html', error='Invalid credentials. Please try again.')

    return render_template('login.html')

# Login route for youtubers
@app.route('/ylogin', methods=['GET', 'POST'])
def ylogin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        youtubers = mongo.db.youtubers
        user = youtubers.find_one({'username': username, 'password': password})

        if user:
            session['logged_in'] = True
            session['username'] = user['username']
            return redirect(url_for('yprofile'))
        else:
            return render_template('ylogin.html', error='Invalid credentials. Please try again.')

    return render_template('ylogin.html')

# Profile route for sponsors
@app.route('/profile')
def profile():
    if 'logged_in' in session:
        return render_template('profile.html', username=session['username'])
    else:
        return redirect(url_for('login'))

# Profile route for youtubers
@app.route('/yprofile')
def yprofile():
    if 'logged_in' in session:
        username = session['username']

        youtubers = mongo.db.youtubers
        user_data = youtubers.find_one({'username': username})

        if user_data:
            channel_name = user_data['channel_name']
            content_type = user_data['content_type']
            return render_template('yprofile.html', username=username, channel_name=channel_name, content_type=content_type)
        else:
            return "User data not found."  # You can handle this case as per your application's logic
    else:
        return redirect(url_for('ylogin'))

# Logout route
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# YouTube Data API Key
# api_key = 'AIzaSyCqgAuD1tK7c4OGTSvQP_87DpWONlDRwzo'
api_key = 'AIzaSyAaL-IuAv-_AAIK5C_nk4n8_uMLKqXGOw4'
youtube = build('youtube', 'v3', developerKey=api_key)

#comments
def get_channel_id_by_name(channel_name):
    request = youtube.search().list(
        part='snippet',
        q=channel_name,
        type='channel',
        maxResults=1
    )
    response = request.execute()
    channel_id = response.get('items', [])[0]['snippet']['channelId'] if 'items' in response else None
    return channel_id

def fetch_video_ids(channel_id, max_results=10):
    url = 'https://www.googleapis.com/youtube/v3/search'
    params = {
        'key': api_key,
        'channelId': channel_id,
        'part': 'id',
        'order': 'date',
        'type': 'video',
        'maxResults': max_results
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        videos_data = response.json()
        video_ids = [item['id']['videoId'] for item in videos_data.get('items', [])]
        return video_ids
    else:
        print('Error:', response.status_code)
        return []

def fetch_comments(video_id, max_results=100):
    url = 'https://www.googleapis.com/youtube/v3/commentThreads'
    params = {
        'part': 'snippet',
        'videoId': video_id,
        'key': api_key,
        'maxResults': max_results
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        comments_data = response.json()
        comments = [item['snippet']['topLevelComment']['snippet']['textDisplay'] for item in comments_data.get('items', [])]
        return comments
    else:
        print('Error:', response.status_code)
        return []

def analyze_sentiment(comment):
    analysis = TextBlob(comment)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return 'positive'
    elif polarity == 0:
        return 'neutral'
    else:
        return 'negative'

def get_video_sentiments(channel_name, max_results=10):
    channel_id = get_channel_id_by_name(channel_name)
    video_ids = fetch_video_ids(channel_id, max_results)
    sentiments_list = []
    for video_id in video_ids:
        comments = fetch_comments(video_id)
        sentiments_list.extend([analyze_sentiment(comment) for comment in comments])
    sentiment_counts = pd.Series(sentiments_list).value_counts()
    return sentiment_counts

#other
def get_channel_states(youtube, channel_ids):
    all_data = []
    request = youtube.channels().list(
        part='snippet,contentDetails,statistics',
        id=','.join(channel_ids)
    )
    response = request.execute()

    for i in range(len(response['items'])):
        data = dict(Channel_name=response['items'][i]['snippet']['title'],
                    Subscribers=response['items'][i]['statistics']['subscriberCount'],
                    Views=response['items'][i]['statistics']['viewCount'],
                    Total_videos=response['items'][i]['statistics']['videoCount'],
                    playlist_id=response['items'][i]['contentDetails']['relatedPlaylists']['uploads']
        )
        all_data.append(data)
    return all_data

##Function to get video ids
def get_video_ids(youtube,playlist_id):
  request=youtube.playlistItems().list(
      part='contentDetails',
      playlistId=playlist_id,
      maxResults=50
  )
  response=request.execute()
  video_ids=[]
  for i in range(len(response['items'])):
    video_ids.append(response['items'][i]['contentDetails']['videoId'])
  next_page_token=response['nextPageToken']
  more_pages=True
  while more_pages:
    if next_page_token is None:
      more_pages=False
    else:
      request=youtube.playlistItems().list(
          part='contentDetails',
          playlistId=playlist_id,
          maxResults=50,
          pageToken=next_page_token)
      response=request.execute()

      for i in range(len(response['items'])):
        video_ids.append(response['items'][i]['contentDetails']['videoId'])
      next_page_token=response.get('nextPageToken')
  return video_ids

##Function to get video details

def get_video_details(youtube, video_ids):
    all_video_stats = []
    for i in range(0, len(video_ids), 50):
        request = youtube.videos().list(
                part="snippet,statistics",
                id=",".join(video_ids[i:i+50])
        )
        response = request.execute()
 
        for video in response['items']:
            video_stats = dict(
                Title= video['snippet']['title'],
                Published_date= video['snippet']['publishedAt'],
                # Views= video['statistics']['viewCount'],
                Views = video['statistics'].get('viewCount', '0'),
                Likes= video['statistics'].get('likeCount'),
                Comments= video['statistics'].get('commentCount'),
            )
            all_video_stats.append(video_stats)
    return all_video_stats

def generate_stacked_bar_chart(data, title):
    fig = plt.figure(figsize=(7, 3))
    ax = fig.add_subplot(111)
    data.plot(kind='bar', stacked=True, ax=ax)
    plt.title(title)
    plt.xlabel('Video IDs')
    plt.ylabel('Counts')
    plt.legend(title='Metrics')
    plt.tight_layout()

    # Convert plot to base64 for embedding in HTML
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    bar_chart_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return bar_chart_url

@app.route('/result/<final_y>')
def result(final_y):
    return render_template('result.html',result=final_y)

@app.route('/submit',methods=['POST',"GET"])
def submit():
    ytuber = ""
    brand = ""

    if 'logged_in' in session:
        username = session['username']
        sponsors = mongo.db.sponsors
        sponsor_data = sponsors.find_one({'username': username})
        if sponsor_data:
            brand = sponsor_data.get('brand_name', '')

    if request.method == 'POST':
        y1 = request.form['y1']
        y2 = request.form['y2']

        youtubers = mongo.db.youtubers
        y1_data = youtubers.find_one({'channel_name': y1})
        y2_data = youtubers.find_one({'channel_name': y2})

        if y1_data and y2_data:
            y1_content_type = y1_data.get('content_type', '')
            y2_content_type = y2_data.get('content_type', '')
            print("inside.")
            if y1_content_type and y2_content_type:
                embedding1 = glove_encode(y1_content_type)
                embedding2 = glove_encode(y2_content_type)
                
                # Assuming p represents the GloVe embedding for the product type
                product = sponsors.find_one({'brand_name': brand})
                product_type = product.get('product_type', '')
                p = glove_encode(product_type)

                if embedding1 is not None and embedding2 is not None and p is not None:
                    # Reshape embeddings to be 2D arrays with a single row
                    embedding1_2d = np.array(embedding1).reshape(1, -1)
                    embedding2_2d = np.array(embedding2).reshape(1, -1)
                    p_2d = np.array(p).reshape(1, -1)
                    
                    # Calculate cosine similarity between content types
                    similarity1 = cosine_similarity(embedding1_2d, p_2d)
                    similarity2 = cosine_similarity(embedding2_2d, p_2d)

                    if similarity1[0][0] < 0.4 and similarity2[0][0] < 0.4:
                        ytuber = "None of the channels you entered are suitable for advertising your product."
                        # return redirect(url_for('result',final_y=y))
                        return render_template('result2.html', ytuber=ytuber)

                    elif similarity1[0][0] < 0.4:
                        ytuber = f"{y1} is not suitable for advertising your product. You can consider {y2} for sponsorship."
                        # return redirect(url_for('result',final_y=y))
                        return render_template('result2.html', ytuber=ytuber)
                    elif similarity2[0][0] < 0.4:
                        ytuber = f"{y2} is not suitable for advertising your product. You can consider {y1} for sponsorship."
                        # return redirect(url_for('result',final_y=y))
                        return render_template('result2.html', ytuber=ytuber)

    channel_names = [y1,y2]
    visualization1 = pd.DataFrame()
    visualization1["Channel_Names"] = channel_names
    
    avg_comment_ratio=[]
    for channel_name in channel_names:
        print(f"Sentiment analysis for {channel_name}:")
        sentiment_counts = get_video_sentiments(channel_name)
        visualization1.loc[visualization1['Channel_Names'] == channel_name, 'neutral'] = sentiment_counts["neutral"]
        visualization1.loc[visualization1['Channel_Names'] == channel_name, 'positive'] = sentiment_counts["positive"]
        visualization1.loc[visualization1['Channel_Names'] == channel_name, 'negative'] = sentiment_counts["negative"]

        print(sentiment_counts)
        avg_comment_ratio.append(sentiment_counts.get('positive', 0) / max(sentiment_counts.get('negative', 1), 1))
        print("Average positive-to-negative ratio:", sentiment_counts.get('positive', 0) / max(sentiment_counts.get('negative', 1), 1))
    ans=""
    if avg_comment_ratio[0]>avg_comment_ratio[1]:
        ans=avg_comment_ratio[0]
    else:
        ans=avg_comment_ratio[1]

    channel_ids1 = []
    channel_ids1.append(get_channel_id_by_name(y1))
    channel_ids2 = []
    channel_ids2.append(get_channel_id_by_name(y2))
    channel_statistics1 = get_channel_states(youtube, channel_ids1)
    channel_data1 = pd.DataFrame(channel_statistics1)
    channel_statistics2 = get_channel_states(youtube, channel_ids2)
    channel_data2 = pd.DataFrame(channel_statistics2)
    channel_data = pd.concat([channel_data1, channel_data2], ignore_index=True)
    channel_data['Avg Comment ratio'] = avg_comment_ratio
    channel_data['Subscribers']=pd.to_numeric(channel_data['Subscribers'])
    channel_data['views']=pd.to_numeric(channel_data['Views'])
    channel_data['Total_videos']=pd.to_numeric(channel_data['Total_videos'])


    playlist_id1 = channel_data.iloc[0, 4]
    playlist_id2 = channel_data.iloc[1, 4]
    video_ids1=get_video_ids(youtube,playlist_id1)
    video_ids2=get_video_ids(youtube,playlist_id2)
    video_details1=get_video_details(youtube,video_ids1)
    video_details2=get_video_details(youtube,video_ids2)
    video_data1=pd.DataFrame(video_details1)
    video_data2=pd.DataFrame(video_details2)

    video_data1['Published_date']=pd.to_datetime(video_data1['Published_date']).dt.date
    video_data1['Views']=pd.to_numeric(video_data1['Views'])
    video_data1['Likes']=pd.to_numeric(video_data1['Likes'])
    video_data1['Views']=pd.to_numeric(video_data1['Views'])

    video_data2['Published_date']=pd.to_datetime(video_data2['Published_date']).dt.date
    video_data2['Views']=pd.to_numeric(video_data2['Views'])
    video_data2['Likes']=pd.to_numeric(video_data2['Likes'])
    video_data2['Views']=pd.to_numeric(video_data2['Views'])

    # Feature Engineering

    # Convert 'Published_date' to datetime format
    video_data1['Published_date'] = pd.to_datetime(video_data1['Published_date'])
    video_data1['Month'] = video_data1['Published_date'].dt.month
    video_data1['Year'] = video_data1['Published_date'].dt.year
    video_data1['Likes_to_Views_Ratio'] = video_data1['Likes'] / video_data1['Views']

    # Convert 'Published_date' to datetime format
    video_data2['Published_date'] = pd.to_datetime(video_data2['Published_date'])
    video_data2['Month'] = video_data2['Published_date'].dt.month
    video_data2['Year'] = video_data2['Published_date'].dt.year
    video_data2['Likes_to_Views_Ratio'] = video_data2['Likes'] / video_data2['Views']
    
    average1 = video_data1['Likes_to_Views_Ratio'].mean()
    average2 = video_data2['Likes_to_Views_Ratio'].mean()
    
    AvgLikesPerViews=[average1,average2]
    channel_data['Avg likes per views']=AvgLikesPerViews

    parameters = pd.DataFrame()
    parameters['Channel_name'] = channel_data['Channel_name']
    parameters['Views'] = channel_data['Views']
    parameters['Subscribers'] = channel_data['Subscribers']
    parameters['Avg Comment ratio'] = channel_data['Avg Comment ratio']
    parameters['Avg likes per views'] = channel_data['Avg likes per views']
    print(parameters)

    scaler = MinMaxScaler()
    df = pd.DataFrame(parameters)
    df = df.drop(columns=['Channel_name'])
    
    normalized_df = pd.DataFrame()
    normalized_df['Views'] = (parameters['Views'] == parameters['Views'].max()).astype(int)
    normalized_df['Subscribers'] = (parameters['Subscribers'] == parameters['Subscribers'].max()).astype(int)
    normalized_df['Avg Comment ratio'] = (parameters['Avg Comment ratio'] == parameters['Avg Comment ratio'].max()).astype(int)
    normalized_df['Avg likes per views'] = (parameters['Avg likes per views'] == parameters['Avg likes per views'].max()).astype(int)
    print(normalized_df)

    visualization1['Views'] = parameters['Views']
    visualization1['Subscribers'] = parameters['Subscribers']
    print('visualization1')
    print(visualization1)

    video_data1_10 = pd.DataFrame()
    video_data1_10['Likes'] = video_data1.head(10)['Likes']
    video_data1_10['Views'] = video_data1.head(10)['Views']
    print('video_data1_10')
    print(video_data1_10)
    
    video_data2_10 = pd.DataFrame()
    video_data2_10['Likes'] = video_data2.head(10)['Likes']
    video_data2_10['Views'] = video_data2.head(10)['Views']
    print('video_data2_10')
    print(video_data2_10)

    score = normalized_df.sum(axis=1)
    print(score)
    
    ytuber = ""
    if score[0] == score[1] : 
        ytuber = "Both channels have similar performance"
        print(ytuber)
    else:
        index = score.idxmax()
        ytuber = parameters.loc[index, 'Channel_name']
        print("Youtuber chosen is",ytuber )
    
# #8e1719 212121 cb342d
    pie_charts = []
    for idx, row in visualization1.iterrows():
        labels = ['Neutral', 'Positive', 'Negative']
        sizes = [row['neutral'], row['positive'], row['negative']]
        # colors = ['#8e1719', 'grey', '#cb342d']
        fig = plt.figure(figsize=(5, 3))
        ax = fig.add_subplot(111)        
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

        # Convert plot to base64 for embedding in HTML
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        pie_chart_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        pie_charts.append(pie_chart_url)

    bar_chart1_url = generate_stacked_bar_chart(video_data1_10, 'Stacked Bar Plot - Video Data 1')
    bar_chart2_url = generate_stacked_bar_chart(video_data2_10, 'Stacked Bar Plot - Video Data 2')
    return render_template('result.html',ytuber=ytuber, visualization1=visualization1, pie_charts=pie_charts, bar_chart1=bar_chart1_url, bar_chart2=bar_chart2_url)


if __name__ == '__main__':
    app.run(debug=True)


