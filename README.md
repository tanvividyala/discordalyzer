# Convolyzer: Message History Analysis & Visualization Tool

This is a personal project I created to analyze my best friend and I's conversation history over the past 7 years and get the hang of basic data science tools. I used pandas to clean my data and find trends, plotly to visualize trends, NTLK to tokenize and preprocess text, and scikit-learn (K-Means, TF-idf) to get started with clustering similar messages. Then, I used Streamlit to build and deploy a functional web app so you can do it too! Check out this video demo:

[![Watch the video demo](https://img.youtube.com/vi/jQuLD7CuNTA/0.jpg)](https://www.youtube.com/watch?v=jQuLD7CuNTA)

**Finished Features**
- Visualization of messages sent over time
- Visualization of trends in word usage
- Visualization of the proportion of messages sent by each member of the conversation
- Word clouds for frequently used words
- Tool to analyze sentiment of messages over time.
- Tool to provide summaries of a day of conversation (with the OpenAI API)

**Works in Progress**
- Tool to cluster messages with similar themes.

## Web App
Access the Web App [Here](https://convolyzer.streamlit.app/).

## Usage Instructions for Jupyter Notebook
1. **Clone the Repository**:
   ```
   git clone https://github.com/tanvividyala/discord-analyzer-tool.git
   cd discord-analyzer-tool
   ```
2. **Install Dependencies**: Install the required Python libraries using the `requirements.txt` file:
   ```
   pip install -r requirements.txt
   ```

4. **Download Your Discord Conversation**: I used this [repository to download my Discord conversation as a CSV File](https://github.com/Tyrrrz/DiscordChatExporter).

6. **Open the Jupyter Notebook**: Start the Jupyter Notebook server and open `discordalyzer.ipynb`. From there you can edit commented cells with information pertaining to your data:
   ```
   jupyter notebook discordalyzer.ipynb
   ```

## Useful Things
- [Awesome tool I used to download my Discord conversation as a CSV File](https://github.com/Tyrrrz/DiscordChatExporter)
- [Awesome tool I used to download my iMessage conversations as a TXT File](https://github.com/reagentx/imessage-exporter)
- [Cool Article about Clustering text](https://towardsdatascience.com/a-friendly-introduction-to-text-clustering-fa996bcefd04)
