# SponsorSpotter

SponsorSpotter is a Flask-based web application that analyzes YouTube channel growth using Data Science and Natural Language Processing (NLP) techniques. It visualizes growth metrics and offers sponsorship recommendations based on the analysis.

## Index
1. [Project Overview](#project-overview)
2. [Objectives](#objectives)
3. [Methodology](#methodology)
   - [Content and Product Alignment Analysis](#1-content-and-product-alignment-analysis)
   - [Audience Sentiment Analysis](#2-audience-sentiment-analysis)
   - [Likes-to-Views Ratio Assessment](#3-likes-to-views-ratio-assessment)
   - [Subscriber Count Evaluation](#4-subscriber-count-evaluation)
   - [Channel View Count Analysis](#5-channel-view-count-analysis)
   - [Final Evaluation](#final-evaluation)
4. [Installation and Setup](#installation-and-setup)
5. [Directory Structure](#directory-structure)
6. [Contributing](#contributing)
7. [Acknowledgments](#acknowledgments)

## Project Overview

This project aims to help sponsors find the best YouTube channels for their products by analyzing various metrics and audience sentiment.

## Objectives

- To collect, preprocess data, and extract relevant features from the data to represent different aspects of YouTube channel growth, such as views, likes growth rate, audience demographics, and content types, among others.
- To determine how the type of content youtuber creates is suitable to the product using GloVe encoding technique.
- To implement NLP techniques for analyzing video comments and audience feedback in order to gain insights into audience sentiment, engagement, and preferences.
- To build DS models and analyze the data to identify patterns in channel growth metrics.
- To design a recommendation system that utilizes the analyzed data to suggest optimal sponsorship choices for potential advertisers based on channel performance and audience characteristics.

## Methodology

### 1. Content and Product Alignment Analysis
- Collect content type input from the YouTuber and product type input from the sponsor.
- Apply GloVe encoding to both inputs to facilitate comparison.
- Retrieve channel data using the YouTube API based on the sponsor-provided channel name.
- Conduct comprehensive data analysis and derive statistical insights.

### 2. Audience Sentiment Analysis
- Extract comments from YouTube videos associated with the target channel.
- Use NLP techniques to analyze sentiment within the comments.
- Calculate the ratio of positive to negative sentiments to gauge audience engagement.

### 3. Likes-to-Views Ratio Assessment
- Compute the ratio of likes to views for all videos within the targeted YouTube channel.

### 4. Subscriber Count Evaluation
- Obtain the subscriber count for the channel under consideration.

### 5. Channel View Count Analysis
- Determine the total number of views accumulated by the YouTube channel.

### Final Evaluation
- Normalize all values to ensure fair comparison across parameters.
- Aggregate scores derived from each parameter to compute a comprehensive numeric score for each channel.
- Recommend the channel with the highest final score for sponsorship, indicating its potential suitability for the sponsoring product.

## Installation and Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/SponsorSpotter.git
   cd SponsorSpotter
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.x installed. Then install the required packages using:
   ```bash
   pip install -r requirements.txt
   ```

3. **API Keys**:
   Obtain YouTube API keys and update.

4. **Run the Application**:
   Start the web application using:
   ```bash
   python app.py
   ```

## Directory Structure

```
SponsorSpotter/
│
├── static/
│   └── images/
│
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── profile.html
│   ├── result.html
│   ├── result2.html
│   ├── resultold.html
│   ├── signup.html
│   ├── ylogin.html
│   ├── yprofile.html
│   └── ysignup.html
│
└── app.py
```

## Contributing

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a pull request.

## Acknowledgments

- [YouTube API](https://developers.google.com/youtube/v3)
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- [NLTK: Natural Language Toolkit](https://www.nltk.org/)

