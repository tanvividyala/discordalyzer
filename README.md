# Discord Analyzer Tool

This is a personal project I created to analyze my best friend and I's Discord conversation history over the past 6 years and get the hang of basic data science tools. I used pandas to clean my data and find trends, matplotlib to visualize trends, NTLK to tokenize and preprocess text, and scikit-learn (K-Means, TF-idf) to get started with clustering similar messages. 

**Finished Features**
- Visualization of messages sent over time
- Visualization of trends in word usage
- Visualization of the proportion of messages sent by each member of the conversation
- Word clouds for frequently used words
- Tool to provide summaries of a day of conversation with the OpenAI API

**Works in Progress**
- Tool to cluster messages with similar themes.
- Tool to analyze sentiment of messages over time.

You can download this Jupyter Notebook and edit wherever there are comments to analyze your own Discord DM/

## Usage Instructions
1. **Clone the Repository**
   ```
   git clone https://github.com/tanvividyala/discord-analyzer-tool.git
   cd discord-analyzer-tool
   ```
2. **Install Dependencies**
   Install the required Python libraries using the `requirements.txt` file:
   ```
   pip install -r requirements.txt
   ```

3. **Download Your Discord Conversation**
   I used this [repository to download my Discord conversation as a CSV File](https://github.com/Tyrrrz/DiscordChatExporter).

4. **Open the Jupyter Notebook**
   Start the Jupyter Notebook server and open `discordalyzer.ipynb`. From there you can edit commented cells with information pertaining to your data:
   ```
   jupyter notebook discordalyzer.ipynb
   ```

## Useful Things
- [Awesome tool I used to download my Discord conversation as a CSV File](https://github.com/Tyrrrz/DiscordChatExporter)
- [Cool Article about Clustering text](https://towardsdatascience.com/a-friendly-introduction-to-text-clustering-fa996bcefd04)
